import argparse
import shutil
from pathlib import Path


def read_entries(track_list_path):
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


def resolve_path(track_list_path, relpath):
    return track_list_path.parent / Path(relpath)


def sync_track_list(track_list_path, apply_changes=False, backup_suffix=".bak"):
    track_list_path = Path(track_list_path)
    entries = read_entries(track_list_path)
    kept = []
    removed = []

    for timestamp, relpath in entries:
        abs_path = resolve_path(track_list_path, relpath)
        if abs_path.is_file():
            kept.append((timestamp, relpath))
        else:
            removed.append((timestamp, relpath))

    print(f"\nTrack list: {track_list_path}")
    print(f"Total entries: {len(entries)}")
    print(f"Kept entries: {len(kept)}")
    print(f"Removed entries: {len(removed)}")
    if removed:
        print("Missing files removed from track list:")
        for timestamp, relpath in removed:
            print(f"  {timestamp} {relpath}")

    if not apply_changes:
        print("Dry run only. Re-run with --apply to rewrite the track list.")
        return

    backup_path = track_list_path.with_suffix(track_list_path.suffix + backup_suffix)
    if not backup_path.exists():
        shutil.copy2(track_list_path, backup_path)

    with open(track_list_path, "w", encoding="utf-8") as handle:
        for timestamp, relpath in kept:
            handle.write(f"{timestamp} {relpath}\n")

    print(f"Updated track list: {track_list_path}")
    print(f"Backup written to: {backup_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove entries from mask_track_*.txt whose mask files were deleted from disk."
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
        help="Actually rewrite the track list. Without this flag, only print a preview.",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".bak",
        help="Suffix appended to the original track list backup when --apply is used.",
    )
    args = parser.parse_args()

    for track_list in args.track_list:
        sync_track_list(
            track_list_path=track_list,
            apply_changes=args.apply,
            backup_suffix=args.backup_suffix,
        )


if __name__ == "__main__":
    main()
