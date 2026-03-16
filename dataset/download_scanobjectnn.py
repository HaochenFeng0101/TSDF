import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


REPO_ROOT = Path(__file__).resolve().parents[2]
TSDF_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


DEFAULT_URL = "http://hkust-vgd.ust.hk/scanobjectnn/h5_files.zip"


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    if tqdm is None:
        urlretrieve(url, destination)
        print("tqdm is not installed, downloaded without a progress bar.")
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


def extract(zip_path, extract_dir):
    print(f"Extracting {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)
    print("Extraction complete")


def maybe_flatten_h5_dir(root):
    root = Path(root)
    nested = root / "h5_files"
    if nested.exists():
        return root

    for child in root.iterdir():
        if child.is_dir() and (child / "main_split").exists():
            target = root / "h5_files"
            if target.exists():
                return root
            shutil.move(str(child), str(target))
            return root
    return root


def write_labels(root):
    labels_path = Path(root) / "labels.txt"
    labels = [
        "bag",
        "bin",
        "box",
        "cabinet",
        "chair",
        "desk",
        "display",
        "door",
        "shelf",
        "table",
        "bed",
        "pillow",
        "sink",
        "sofa",
        "toilet",
    ]
    with open(labels_path, "w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(f"{label}\n")
    return labels_path


def main():
    parser = argparse.ArgumentParser(
        description="Download and unpack the ScanObjectNN h5 dataset."
    )
    parser.add_argument(
        "--output-dir",
        default=str(TSDF_ROOT / "data" / "ScanObjectNN"),
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Download URL for the official h5 archive.",
    )
    parser.add_argument(
        "--archive-name",
        default="h5_files.zip",
        help="Local archive filename.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload even if the archive already exists.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the zip archive after extraction.",
    )
    parser.add_argument(
        "--write-example",
        action="store_true",
        help="Print a ready-to-use dataloader example after download.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    archive_path = output_dir / args.archive_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.force_download or not archive_path.exists():
        download(args.url, archive_path)
    else:
        print(f"Archive already exists: {archive_path}")

    extract(archive_path, output_dir)
    maybe_flatten_h5_dir(output_dir)
    labels_path = write_labels(output_dir)

    if not args.keep_archive and archive_path.exists():
        archive_path.unlink()
        print(f"Removed archive {archive_path}")

    print(f"Dataset ready under {output_dir}")
    print(f"Labels file written to {labels_path}")

    if args.write_example:
        print()
        print("Example:")
        print("from TSDF.dataset.scanobjectnn_data import get_scanobjectnn_dataloaders")
        print(
            f"train_ds, test_ds, train_loader, test_loader = get_scanobjectnn_dataloaders(root='{output_dir}')"
        )
        print("points, label = train_ds[0]")
        print("print(points.shape, label)")


if __name__ == "__main__":
    main()
