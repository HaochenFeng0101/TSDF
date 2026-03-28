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
        raise RuntimeError(f"下载失败，已重试 {retries} 次。最后错误: {last_error}")

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

    raise RuntimeError(f"下载失败，已重试 {retries} 次。最后错误: {last_error}")


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
        description="下载并解压 ModelNet40 原始 OFF 网格版本。"
    )
    parser.add_argument(
        "--output-dir",
        default=str(TSDF_ROOT / "data" / "ModelNet40"),
        help="数据集根目录。",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="ModelNet40 压缩包地址。",
    )
    parser.add_argument(
        "--archive-name",
        default="ModelNet40.zip",
        help="本地压缩包文件名。",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="即使压缩包已存在也重新下载。",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="解压后保留 zip 压缩包。",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="单次下载尝试的超时时间，单位秒。",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="下载失败后的重试次数。",
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
            raise RuntimeError("解压后没有找到可用的 ModelNet40 数据目录。")

    if not args.keep_archive and archive_path.exists():
        archive_path.unlink()
        print(f"Removed archive {archive_path}")

    print(f"Dataset ready under {dataset_dir}")


if __name__ == "__main__":
    main()
