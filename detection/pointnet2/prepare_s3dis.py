import argparse
import json
import shutil
import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
TSDF_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

S3DIS_LABELS = [
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "chair",
    "table",
    "bookcase",
    "sofa",
    "board",
    "clutter",
]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(S3DIS_LABELS)}
ARCHIVE_EXTS = [".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz"]
DEFAULT_FULL_URL = "https://cvg-data.inf.ethz.ch/s3dis/Stanford3dDataset_v1.2_Aligned_Version.zip"

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    if tqdm is None:
        urlretrieve(url, destination)
        print(f"Downloaded to {destination}")
        return

    progress_bar = None

    def reporthook(block_num, block_size, total_size):
        nonlocal progress_bar
        if progress_bar is None:
            total = total_size if total_size > 0 else None
            progress_bar = tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=destination.name,
                dynamic_ncols=True,
            )
        downloaded = block_num * block_size
        if total_size > 0:
            downloaded = min(downloaded, total_size)
        progress_bar.update(downloaded - progress_bar.n)

    try:
        urlretrieve(url, destination, reporthook=reporthook)
    finally:
        if progress_bar is not None:
            progress_bar.close()
    print(f"Downloaded to {destination}")


def write_labels(path):
    with open(path, "w", encoding="utf-8") as handle:
        for label in S3DIS_LABELS:
            handle.write(f"{label}\n")


def find_area_dir(raw_root, area_name):
    candidates = [
        raw_root / area_name,
        raw_root / "Stanford3dDataset_v1.2_Aligned_Version" / area_name,
        raw_root / "Stanford3dDataset_v1.2" / area_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def extract_archive(archive_path, extract_root):
    print(f"Extracting {archive_path} -> {extract_root}")
    extract_root.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(archive_path), str(extract_root))


def find_local_archive(archive_dir, area_name):
    if archive_dir is None:
        return None
    archive_dir = Path(archive_dir)
    for ext in ARCHIVE_EXTS:
        matches = sorted(archive_dir.glob(f"*{area_name}*{ext}"))
        if matches:
            return matches[0]
    return None


def maybe_download_area(area_name, archive_dir, url_template, skip_existing):
    if archive_dir is None or url_template is None:
        return None
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)

    existing = find_local_archive(archive_dir, area_name)
    if existing is not None and skip_existing:
        print(f"Using existing archive for {area_name}: {existing}")
        return existing

    url = url_template.format(area=area_name, area_lower=area_name.lower())
    destination = archive_dir / f"{area_name}.zip"
    download(url, destination)
    return destination


def find_full_archive(archive_dir):
    if archive_dir is None:
        return None
    archive_dir = Path(archive_dir)
    candidates = [
        archive_dir / "Stanford3dDataset_v1.2_Aligned_Version.zip",
        archive_dir / "Stanford3dDataset_v1.2.zip",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def ensure_full_dataset_available(raw_root, archive_dir, full_url, skip_existing):
    aligned_root = raw_root / "Stanford3dDataset_v1.2_Aligned_Version"
    if aligned_root.exists():
        return aligned_root

    archive_dir = Path(archive_dir) if archive_dir is not None else (raw_root / "archives")
    archive_dir.mkdir(parents=True, exist_ok=True)

    archive_path = find_full_archive(archive_dir)
    if archive_path is None:
        archive_path = archive_dir / Path(full_url).name
        if archive_path.exists() and skip_existing:
            print(f"Using existing full archive: {archive_path}")
        else:
            download(full_url, archive_path)

    extract_archive(archive_path, raw_root)
    if not aligned_root.exists():
        raise RuntimeError(f"已下载并解压 {archive_path}，但没有找到 {aligned_root}")
    return aligned_root


def ensure_area_available(area_name, raw_root, archive_dir=None, url_template=None, full_url=DEFAULT_FULL_URL, skip_existing=True):
    area_dir = find_area_dir(raw_root, area_name)
    if area_dir is not None:
        return area_dir

    archive_path = find_local_archive(archive_dir, area_name)
    if archive_path is None and url_template is not None:
        archive_path = maybe_download_area(area_name, archive_dir, url_template, skip_existing)

    if archive_path is None:
        full_root = ensure_full_dataset_available(
            raw_root=raw_root,
            archive_dir=archive_dir,
            full_url=full_url,
            skip_existing=skip_existing,
        )
        area_dir = find_area_dir(raw_root, area_name)
        if area_dir is not None:
            return area_dir
        raise FileNotFoundError(f"已准备完整 S3DIS 数据，但仍然找不到 {area_name}，检查解压结构: {full_root}")

    extract_archive(archive_path, raw_root)
    area_dir = find_area_dir(raw_root, area_name)
    if area_dir is None:
        raise RuntimeError(f"已解压 {archive_path}，但仍然找不到 {area_name} 目录。")
    return area_dir


def parse_annotation_file(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line_idx, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 6:
                continue
            try:
                row = [float(value) for value in parts[:6]]
            except ValueError:
                continue
            if not np.all(np.isfinite(row)):
                continue
            rows.append(row)

    if not rows:
        raise ValueError(f"{path} 没有可用的 xyzrgb 行。")

    points = np.asarray(rows, dtype=np.float32)

    name = path.stem
    if "_" in name:
        raw_label = name.split("_")[0]
    else:
        raw_label = name
    raw_label = raw_label.lower()
    label = raw_label if raw_label in LABEL_TO_INDEX else "clutter"
    label_idx = LABEL_TO_INDEX[label]
    labels = np.full(len(points), label_idx, dtype=np.int64)
    return points, labels


def build_room_npz(area_dir, room_dir, output_path):
    annotations_dir = room_dir / "Annotations"
    if not annotations_dir.exists():
        raise FileNotFoundError(f"缺少 Annotations 目录: {annotations_dir}")

    all_points = []
    all_labels = []
    for txt_path in sorted(annotations_dir.glob("*.txt")):
        points, labels = parse_annotation_file(txt_path)
        all_points.append(points)
        all_labels.append(labels)

    if not all_points:
        raise RuntimeError(f"{annotations_dir} 下没有可用标注文件。")

    points = np.concatenate(all_points, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, points=points, labels=labels)
    return len(points)


def process_area(area_dir, output_root, val_areas):
    area_name = area_dir.name
    split = "val" if area_name in val_areas else "train"
    room_dirs = sorted(path for path in area_dir.iterdir() if path.is_dir())
    summaries = []

    for room_dir in room_dirs:
        output_name = f"{area_name}_{room_dir.name}.npz"
        output_path = output_root / split / output_name
        num_points = build_room_npz(area_dir, room_dir, output_path)
        summaries.append(
            {
                "area": area_name,
                "room": room_dir.name,
                "split": split,
                "points": int(num_points),
                "output": str(output_path),
            }
        )
        print(f"Prepared {output_name} -> split={split} | points={num_points}")
    return summaries


def main():
    parser = argparse.ArgumentParser(
        description="按需准备 S3DIS 为 PointNet++ 分割训练格式，不下载全量无关数据。"
    )
    parser.add_argument(
        "--raw-root",
        default=str(TSDF_ROOT / "data" / "S3DIS_raw"),
        help="S3DIS 解压根目录。",
    )
    parser.add_argument(
        "--archive-dir",
        default=None,
        help="可选，本地 S3DIS 压缩包目录。可放整包或 Area 压缩包。",
    )
    parser.add_argument(
        "--url-template",
        default=None,
        help="可选，Area 下载 URL 模板，例如 https://.../{area}.zip",
    )
    parser.add_argument(
        "--full-url",
        default=DEFAULT_FULL_URL,
        help="整包 S3DIS 对齐版下载地址。默认使用 ETHZ 公开镜像。",
    )
    parser.add_argument(
        "--areas",
        nargs="+",
        default=["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"],
        help="需要准备的 Area 列表。",
    )
    parser.add_argument(
        "--val-areas",
        nargs="+",
        default=["Area_6"],
        help="作为验证集的 Area，默认 Area_6。",
    )
    parser.add_argument(
        "--output-root",
        default=str(TSDF_ROOT / "data" / "S3DIS_seg"),
        help="输出根目录，生成 train/ 和 val/ 的 .npz。",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果 Area 已存在压缩包或已解压目录，则直接复用。",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    output_root = Path(args.output_root)
    (output_root / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "val").mkdir(parents=True, exist_ok=True)

    summaries = []
    for area_name in args.areas:
        area_dir = ensure_area_available(
            area_name=area_name,
            raw_root=raw_root,
            archive_dir=args.archive_dir,
            url_template=args.url_template,
            full_url=args.full_url,
            skip_existing=args.skip_existing,
        )
        summaries.extend(process_area(area_dir, output_root, set(args.val_areas)))

    labels_path = output_root / "labels.txt"
    write_labels(labels_path)

    with open(output_root / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)

    train_count = sum(1 for item in summaries if item["split"] == "train")
    val_count = sum(1 for item in summaries if item["split"] == "val")
    print(f"Prepared train rooms: {train_count}")
    print(f"Prepared val rooms: {val_count}")
    print(f"Labels file: {labels_path}")
    print(f"Summary: {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
