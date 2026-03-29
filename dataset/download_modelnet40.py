import argparse
import socket
import shutil
import sys
import time
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


REPO_ROOT = Path(__file__).resolve().parents[3]
TSDF_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


DEFAULT_URL = "http://modelnet.cs.princeton.edu/ModelNet40.zip"


def download(url, destination, timeout=60, retries=3):
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    print(f"Destination: {destination}")
    print(f"Timeout per attempt: {timeout}s | Retries: {retries}")
    if tqdm is None:
        last_error = None
        for attempt in range(1, retries + 1):
            print(f"Attempt {attempt}/{retries} ...")
            try:
                previous_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(timeout)
                urlretrieve(url, destination)
                print("tqdm is not installed, downloaded without a progress bar.")
                print(f"Downloaded to {destination}")
                return
            except Exception as exc:
                last_error = exc
                if destination.exists():
                    destination.unlink()
                print(f"Attempt {attempt} failed: {exc}")
                if attempt < retries:
                    time.sleep(2)
            finally:
                socket.setdefaulttimeout(previous_timeout)
        raise RuntimeError(f"Download failed after {retries} retries. Last error: {last_error}")

    last_error = None

    for attempt in range(1, retries + 1):
        progress_bar = None
        print(f"Attempt {attempt}/{retries} ...")

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

        previous_timeout = socket.getdefaulttimeout()
        try:
            socket.setdefaulttimeout(timeout)
            urlretrieve(url, destination, reporthook=reporthook)
            print(f"Downloaded to {destination}")
            return
        except Exception as exc:
            last_error = exc
            if destination.exists():
                destination.unlink()
            print(f"Attempt {attempt} failed: {exc}")
            if attempt < retries:
                time.sleep(2)
        finally:
            socket.setdefaulttimeout(previous_timeout)
            if progress_bar is not None:
                progress_bar.close()

    raise RuntimeError(f"Download failed after {retries} retries. Last error: {last_error}")


def extract(zip_path, extract_dir):
    print(f"Extracting {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)
    print("Extraction complete")


def maybe_flatten(root):
    root = Path(root)
    nested = root / "modelnet40_ply_hdf5_2048"
    if nested.exists():
        return nested

    for child in root.iterdir():
        if child.is_dir() and (child / "shape_names.txt").exists():
            target = root / "modelnet40_ply_hdf5_2048"
            if child != target:
                if target.exists():
                    shutil.rmtree(target)
                shutil.move(str(child), str(target))
            return target
    return nested


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract the original OFF-mesh version of ModelNet40."
    )
    parser.add_argument(
        "--output-dir",
        default=str(TSDF_ROOT / "data" / "ModelNet40"),
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="ModelNet40 archive URL.",
    )
    parser.add_argument(
        "--archive-name",
        default="ModelNet40.zip",
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
        "--timeout",
        type=int,
        default=60,
        help="Timeout in seconds for each download attempt.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries after a failed download attempt.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    archive_path = output_dir / args.archive_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.force_download or not archive_path.exists():
        download(args.url, archive_path, timeout=args.timeout, retries=args.retries)
    else:
        print(f"Archive already exists: {archive_path}")

    extract(archive_path, output_dir)
    dataset_dir = maybe_flatten(output_dir)
    if not dataset_dir.exists():
        fallback_dir = output_dir / "ModelNet40"
        if fallback_dir.exists():
            dataset_dir = fallback_dir
        else:
            raise RuntimeError("No usable ModelNet40 data directory was found after extraction.")

    if not args.keep_archive and archive_path.exists():
        archive_path.unlink()
        print(f"Removed archive {archive_path}")

    print(f"Dataset ready under {dataset_dir}")


if __name__ == "__main__":
    main()
