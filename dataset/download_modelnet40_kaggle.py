import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TSDF_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DATASET_SLUG = "balraj98/modelnet40-princeton-3d-object-dataset"


def find_kaggle_cli():
    for candidate in ("kaggle", "kaggle.exe"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError(
        "Could not find the Kaggle CLI. Install it with `pip install kaggle` and "
        "configure credentials via kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY."
    )

def load_kaggle_credentials():  
    env_username = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")
    if env_username and env_key:
        return {"username": env_username, "key": env_key}
    candidate_paths = [
        TSDF_ROOT / "kaggle.json",
        REPO_ROOT / "kaggle.json",
        Path.cwd() / "kaggle.json",
        Path.home() / ".kaggle" / "kaggle.json",
    ]
    visited = set()
    for path in candidate_paths:
        resolved = path.resolve()
        if resolved in visited or not resolved.exists():
            continue
        visited.add(resolved)
        with open(resolved, "r", encoding="utf-8") as handle:
            credentials = json.load(handle)
        if "username" not in credentials or "key" not in credentials:
            raise KeyError(
                f"Malformed Kaggle credentials file at {resolved}: expected "
                "'username' and 'key'."
            )
        return {
            "username": str(credentials["username"]).strip(),
            "key": str(credentials["key"]).strip(),
        }
    return None


def build_kaggle_env():
    env = os.environ.copy()
    credentials = load_kaggle_credentials()
    if credentials is not None:
        env["KAGGLE_USERNAME"] = credentials["username"]
        env["KAGGLE_KEY"] = credentials["key"]
    return env
    

def run_download(kaggle_cli, dataset_slug, output_dir, force_download):
    command = [
        kaggle_cli,
        "datasets",
        "download",
        "-d",
        dataset_slug,
        "-p",
        str(output_dir),
    ]
    if force_download:
        command.append("--force")
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, env=build_kaggle_env())


def find_downloaded_archive(output_dir):
    archives = sorted(
        output_dir.glob("*.zip"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not archives:
        raise FileNotFoundError(f"No zip archive found in {output_dir} after Kaggle download.")
    return archives[0]


def extract(zip_path, output_dir):
    print(f"Extracting {zip_path.name} -> {output_dir}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)
    print("Extraction complete")


def flatten_modelnet_root(output_dir):
    output_dir = Path(output_dir)
    for candidate in (output_dir / "ModelNet40", output_dir / "modelnet40"):
        if candidate.exists():
            return candidate

    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        for nested in (child / "ModelNet40", child / "modelnet40"):
            if nested.exists():
                target = output_dir / nested.name
                if target.exists():
                    return target
                shutil.move(str(nested), str(target))
                return target
    return output_dir


def write_labels(root):
    root = Path(root)
    labels = sorted(path.name for path in root.iterdir() if path.is_dir())
    if not labels:
        return None
    labels_path = root / "labels.txt"
    with open(labels_path, "w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(f"{label}\n")
    return labels_path


def main():
    parser = argparse.ArgumentParser(
        description="Download and unpack the ModelNet40 dataset from Kaggle."
    )
    parser.add_argument(
        "--output-dir",
        default=str(TSDF_ROOT / "data" / "ModelNet40"),
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--dataset-slug",
        default=DATASET_SLUG,
        help="Kaggle dataset slug.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload even if an archive already exists.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the zip archive after extraction.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kaggle_cli = find_kaggle_cli()
    run_download(kaggle_cli, args.dataset_slug, output_dir, args.force_download)
    archive_path = find_downloaded_archive(output_dir)
    extract(archive_path, output_dir)

    dataset_root = flatten_modelnet_root(output_dir)
    labels_path = write_labels(dataset_root)

    if not args.keep_archive and archive_path.exists():
        archive_path.unlink()
        print(f"Removed archive {archive_path}")

    print(f"Dataset ready under {dataset_root}")
    if labels_path is not None:
        print(f"Labels file written to {labels_path}")
    print()
    print("Example:")
    print(
        "python detection/train_pointnet_cls.py "
        f"--dataset-type modelnet40 --modelnet40-root {dataset_root}"
    )


if __name__ == "__main__":
    main()
