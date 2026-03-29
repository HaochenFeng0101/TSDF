import argparse
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


AREA_PREFIX_RE = re.compile(r"^a\d+_")


def read_track_entries(track_list_path):
    entries = []
    with open(track_list_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"Expected '<timestamp> <relative_mask_path>' in {track_list_path}, got: {stripped}"
                )
            entries.append((parts[0], parts[1]))
    return entries


def resolve_relpath(track_list_path, relpath):
    return track_list_path.parent / Path(relpath)


def compute_mask_area(mask_path):
    with Image.open(mask_path) as image:
        array = np.array(image)
    if array.ndim == 3:
        array = array[..., 0]
    return int(np.count_nonzero(array))


def build_prefixed_name(original_name, area):
    clean_name = AREA_PREFIX_RE.sub("", original_name)
    return f"a{area:06d}_{clean_name}"


def rename_masks_for_track(track_list_path, apply_changes=False, backup_suffix=".bak"):
    track_list_path = Path(track_list_path)
    entries = read_track_entries(track_list_path)
    if not entries:
        raise RuntimeError(f"Track list is empty: {track_list_path}")

    rename_map = {}
    updated_entries = []

    for timestamp, relpath in entries:
        relpath_obj = Path(relpath)
        abs_path = resolve_relpath(track_list_path, relpath_obj)
        if not abs_path.is_file():
            raise FileNotFoundError(f"Mask file not found: {abs_path}")

        if relpath not in rename_map:
            area = compute_mask_area(abs_path)
            new_name = build_prefixed_name(abs_path.name, area)
            new_abs_path = abs_path.with_name(new_name)
            new_relpath = str(relpath_obj.with_name(new_name)).replace("\\", "/")
            rename_map[relpath] = {
                "area": area,
                "old_abs_path": abs_path,
                "new_abs_path": new_abs_path,
                "old_relpath": relpath,
                "new_relpath": new_relpath,
            }
        updated_entries.append((timestamp, rename_map[relpath]["new_relpath"]))

    print(f"\nTrack list: {track_list_path}")
    print(f"Unique masks: {len(rename_map)}")
    for item in sorted(rename_map.values(), key=lambda payload: payload["old_relpath"]):
        print(
            f"  area={item['area']:6d}  {item['old_relpath']} -> {item['new_relpath']}"
        )

    if not apply_changes:
        print("Dry run only. Re-run with --apply to rename files and rewrite the track list.")
        return

    backup_path = track_list_path.with_suffix(track_list_path.suffix + backup_suffix)
    if not backup_path.exists():
        shutil.copy2(track_list_path, backup_path)

    for item in rename_map.values():
        old_abs_path = item["old_abs_path"]
        new_abs_path = item["new_abs_path"]
        if old_abs_path == new_abs_path:
            continue
        if new_abs_path.exists():
            raise FileExistsError(f"Target file already exists: {new_abs_path}")
        old_abs_path.rename(new_abs_path)

    with open(track_list_path, "w", encoding="utf-8") as handle:
        for timestamp, relpath in updated_entries:
            handle.write(f"{timestamp} {relpath}\n")

    print(f"Updated track list: {track_list_path}")
    print(f"Backup written to: {backup_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prefix each mask filename with its pixel area and update mask_track_*.txt accordingly."
    )
    parser.add_argument(
        "--track-list",
        action="append",
        required=True,
        help="Path to a mask_track_*.txt file. Repeat for multiple tracks.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename files and rewrite the track list. Without this flag, only print a preview.",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".bak",
        help="Suffix appended to the original track list backup when --apply is used.",
    )
    args = parser.parse_args()

    for track_list in args.track_list:
        rename_masks_for_track(
            track_list_path=track_list,
            apply_changes=args.apply,
            backup_suffix=args.backup_suffix,
        )


if __name__ == "__main__":
    main()
