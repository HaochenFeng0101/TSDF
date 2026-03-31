import argparse
import json
import shutil
import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

'''
python3 detection/pointnet2/prepare_s3dis.py

https://cvg-data.inf.ethz.ch/s3dis/
'''
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
DEFAULT_RAW_ROOT = Path("data") / "S3DIS_raw"
DEFAULT_OUTPUT_ROOT = Path("data") / "S3DIS_seg"
DEFAULT_AREAS = ["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"]
DEFAULT_VAL_AREAS = ["Area_6"]

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def resolve_repo_path(path_str):
    path = Path(path_str)
    if not path.is_absolute():
        path = TSDF_ROOT / path
    return path


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
        raise RuntimeError(f"Downloaded and extracted {archive_path}, but {aligned_root} was not found.")
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
        raise FileNotFoundError(f"Prepared the full S3DIS dataset, but {area_name} is still missing. Check the extracted structure under {full_root}.")

    extract_archive(archive_path, raw_root)
    area_dir = find_area_dir(raw_root, area_name)
    if area_dir is None:
        raise RuntimeError(f"Extracted {archive_path}, but the {area_name} directory was not found.")
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
        raise ValueError(f"{path} contains no usable xyzrgb rows.")

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


def build_room_data(room_dir):
    annotations_dir = room_dir / "Annotations"
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Missing Annotations directory: {annotations_dir}")

    all_points = []
    all_labels = []
    for txt_path in sorted(annotations_dir.glob("*.txt")):
        points, labels = parse_annotation_file(txt_path)
        all_points.append(points)
        all_labels.append(labels)

    if not all_points:
        raise RuntimeError(f"No usable annotation files found under {annotations_dir}.")

    points = np.concatenate(all_points, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    points[:, 3:6] = np.clip(points[:, 3:6] / 255.0, 0.0, 1.0)
    return points, labels


def list_room_dirs(area_dir):
    return sorted(
        path for path in area_dir.iterdir() if path.is_dir() and (path / "Annotations").is_dir()
    )


def compute_room_normalized_xyz(points):
    xyz = points[:, :3].astype(np.float32)
    mins = xyz.min(axis=0, keepdims=True)
    maxs = xyz.max(axis=0, keepdims=True)
    span = np.maximum(maxs - mins, 1e-6)
    return (xyz - mins) / span


def slice_room_blocks(points, labels, block_size, stride, min_points, max_blocks_per_room=0):
    xyz = points[:, :3]
    x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()
    x_max, y_max = xyz[:, 0].max(), xyz[:, 1].max()
    room_norm_xyz = compute_room_normalized_xyz(points)

    x_starts = np.arange(x_min, x_max + 1e-6, stride, dtype=np.float32)
    y_starts = np.arange(y_min, y_max + 1e-6, stride, dtype=np.float32)

    blocks = []
    for x0 in x_starts:
        for y0 in y_starts:
            x1 = x0 + block_size
            y1 = y0 + block_size
            mask = (
                (xyz[:, 0] >= x0)
                & (xyz[:, 0] <= x1)
                & (xyz[:, 1] >= y0)
                & (xyz[:, 1] <= y1)
            )
            indices = np.where(mask)[0]
            if len(indices) < min_points:
                continue

            block_points = points[indices]
            block_labels = labels[indices]
            block_room_norm = room_norm_xyz[indices]
            block_features = np.concatenate([block_points, block_room_norm], axis=1).astype(np.float32)
            blocks.append(
                {
                    "points": block_features,
                    "labels": block_labels.astype(np.int64),
                    "num_points": int(len(indices)),
                    "bounds": [float(x0), float(y0), float(x1), float(y1)],
                }
            )
            if max_blocks_per_room > 0 and len(blocks) >= max_blocks_per_room:
                return blocks

    if not blocks:
        full_features = np.concatenate([points, room_norm_xyz], axis=1).astype(np.float32)
        blocks.append(
            {
                "points": full_features,
                "labels": labels.astype(np.int64),
                "num_points": int(len(labels)),
                "bounds": None,
            }
        )
    return blocks


def cleanup_split_outputs(output_root, areas):
    area_prefixes = tuple(f"{area}_" for area in areas)
    removed = 0
    for split in ("train", "val"):
        split_dir = output_root / split
        if not split_dir.exists():
            continue
        for path in split_dir.glob("*.npz"):
            if path.name.startswith(area_prefixes):
                path.unlink(missing_ok=True)
                removed += 1
    return removed


def process_area(
    area_dir,
    output_root,
    val_areas,
    block_size,
    stride,
    min_block_points,
    max_blocks_per_room,
    progress_bar=None,
):
    area_name = area_dir.name
    split = "val" if area_name in val_areas else "train"
    room_dirs = list_room_dirs(area_dir)
    summaries = []

    for room_dir in room_dirs:
        room_points, room_labels = build_room_data(room_dir)
        num_room_points = len(room_points)
        blocks = slice_room_blocks(
            room_points,
            room_labels,
            block_size=block_size,
            stride=stride,
            min_points=min_block_points,
            max_blocks_per_room=max_blocks_per_room,
        )

        for block_idx, block in enumerate(blocks):
            output_name = f"{area_name}_{room_dir.name}_block_{block_idx:03d}.npz"
            output_path = output_root / split / output_name
            np.savez_compressed(output_path, points=block["points"], labels=block["labels"])
            summaries.append(
                {
                    "area": area_name,
                    "room": room_dir.name,
                    "split": split,
                    "room_points": int(num_room_points),
                    "block_points": int(block["num_points"]),
                    "block_index": int(block_idx),
                    "bounds": block["bounds"],
                    "output": str(output_path),
                }
            )
        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix_str(
                f"{area_name}/{room_dir.name} blocks={len(blocks)}",
                refresh=False,
            )
    return summaries


def main():
    parser = argparse.ArgumentParser(
        description="Prepare S3DIS for PointNet++ segmentation training on demand without downloading unrelated large datasets."
    )
    parser.add_argument(
        "--raw-root",
        default=str(DEFAULT_RAW_ROOT),
        help="Root directory for extracted S3DIS data. Defaults to data/S3DIS_raw under the repository root.",
    )
    parser.add_argument(
        "--archive-dir",
        default=None,
        help="Optional local S3DIS archive directory. Can contain the full archive or per-Area archives.",
    )
    parser.add_argument(
        "--url-template",
        default=None,
        help="Optional Area download URL template, for example https://.../{area}.zip",
    )
    parser.add_argument(
        "--full-url",
        default=DEFAULT_FULL_URL,
        help="Download URL for the full aligned S3DIS archive. Defaults to the ETHZ public mirror.",
    )
    parser.add_argument(
        "--areas",
        nargs="+",
        default=DEFAULT_AREAS,
        help="List of Areas to prepare.",
    )
    parser.add_argument(
        "--val-areas",
        nargs="+",
        default=DEFAULT_VAL_AREAS,
        help="Areas to use for validation. Defaults to Area_6.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output root directory. Creates train/ and val/ .npz files. Defaults to data/S3DIS_seg under the repository root.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing archives or extracted Areas when available.",
    )
    parser.add_argument(
        "--block-size",
        type=float,
        default=1.5,
        help="XY block size in meters for room slicing.",
    )
    parser.add_argument(
        "--stride",
        type=float,
        default=0.5,
        help="XY stride in meters for room slicing.",
    )
    parser.add_argument(
        "--min-block-points",
        type=int,
        default=1024,
        help="Minimum number of points required to keep a block.",
    )
    parser.add_argument(
        "--max-blocks-per-room",
        type=int,
        default=0,
        help="Optional cap on the number of blocks per room. Use 0 for no cap.",
    )
    parser.add_argument(
        "--skip-clean-output",
        action="store_true",
        help="Keep existing processed .npz files for the selected Areas instead of clearing them first.",
    )
    args = parser.parse_args()

    raw_root = resolve_repo_path(args.raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    archive_dir = resolve_repo_path(args.archive_dir) if args.archive_dir is not None else None
    output_root = resolve_repo_path(args.output_root)
    (output_root / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "val").mkdir(parents=True, exist_ok=True)

    if not args.skip_clean_output:
        removed = cleanup_split_outputs(output_root, args.areas)
        if removed:
            print(f"Removed {removed} existing processed files before regeneration.")

    area_dirs = []
    for area_name in args.areas:
        area_dirs.append(
            ensure_area_available(
                area_name=area_name,
                raw_root=raw_root,
                archive_dir=archive_dir,
                url_template=args.url_template,
                full_url=args.full_url,
                skip_existing=args.skip_existing,
            )
        )

    total_rooms = sum(len(list_room_dirs(area_dir)) for area_dir in area_dirs)
    progress_bar = None
    if tqdm is not None:
        progress_bar = tqdm(total=total_rooms, desc="prepare_s3dis", dynamic_ncols=True)

    summaries = []
    try:
        for area_dir in area_dirs:
            summaries.extend(
                process_area(
                    area_dir,
                    output_root,
                    set(args.val_areas),
                    block_size=args.block_size,
                    stride=args.stride,
                    min_block_points=args.min_block_points,
                    max_blocks_per_room=args.max_blocks_per_room,
                    progress_bar=progress_bar,
                )
            )
    finally:
        if progress_bar is not None:
            progress_bar.close()
    labels_path = output_root / "labels.txt"
    write_labels(labels_path)

    with open(output_root / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)

    train_count = sum(1 for item in summaries if item["split"] == "train")
    val_count = sum(1 for item in summaries if item["split"] == "val")
    print(f"Prepared train blocks: {train_count}")
    print(f"Prepared val blocks: {val_count}")
    print(f"Labels file: {labels_path}")
    print(f"Summary: {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
