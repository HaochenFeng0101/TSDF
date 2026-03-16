import argparse
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


SEQUENCES = {
    "fr1_desk": "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz",
    "fr2_xyz": "https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz",
    "fr3_office": "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz",
}

DEFAULT_SAMPLES = ["fr1_desk", "fr2_xyz", "fr3_office"]


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


def extract(archive_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path.name} -> {output_dir}")
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(output_dir)
    print("Extraction complete")


def main():
    parser = argparse.ArgumentParser(
        description="Download a few indoor TUM RGB-D sample sequences."
    )
    parser.add_argument(
        "--output-dir",
        default="data/tum",
        help="Directory where the TUM sequences will be stored.",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        choices=sorted(SEQUENCES.keys()),
        default=DEFAULT_SAMPLES,
        help="Which sample sequences to download.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the .tgz archives after extraction.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload archives even if they already exist.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample in args.samples:
        url = SEQUENCES[sample]
        archive_name = url.split("/")[-1]
        archive_path = output_dir / archive_name

        if args.force_download or not archive_path.exists():
            download(url, archive_path)
        else:
            print(f"Archive already exists: {archive_path}")

        extract(archive_path, output_dir)

        if not args.keep_archive and archive_path.exists():
            archive_path.unlink()
            print(f"Removed archive {archive_path}")

    print(f"TUM RGB-D samples are ready under {output_dir}")


if __name__ == "__main__":
    main()
