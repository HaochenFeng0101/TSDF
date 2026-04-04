import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


TSDF_ROOT = Path(__file__).resolve().parents[1]

SCENE_TO_CONFIG = {
    "fr1_desk": TSDF_ROOT / "configs" / "rgbd" / "tum" / "fr1_desk.yaml",
    "fr2_xyz": TSDF_ROOT / "configs" / "rgbd" / "tum" / "fr2_xyz.yaml",
    "fr3_office": TSDF_ROOT / "configs" / "rgbd" / "tum" / "fr3_office.yaml",
}

OBJECT_SPECS = {
    "bed": {"mask_class": "bed", "expected_label": "bed"},
    "chair": {"mask_class": "chair", "expected_label": "chair"},
    "display": {"mask_class": "tv", "expected_label": "monitor"},
    "keyboard": {"mask_class": "keyboard", "expected_label": "keyboard"},
    "sink": {"mask_class": "sink", "expected_label": "sink"},
    "sofa": {"mask_class": "couch", "expected_label": "sofa"},
    "table": {"mask_class": "dining table", "expected_label": "table"},
    "toilet": {"mask_class": "toilet", "expected_label": "toilet"},
}

OBJECT_ALIASES = {
    "couch": "sofa",
    "dining_table": "table",
    "monitor": "display",
    "tv": "display",
}


def update_recursive(base, extra):
    for key, value in extra.items():
        if key not in base:
            base[key] = {}
        if isinstance(value, dict):
            update_recursive(base[key], value)
        else:
            base[key] = value


def load_config(path):
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg_special = yaml.full_load(handle)

    inherit_from = cfg_special.get("inherit_from")
    if inherit_from is not None:
        inherit_path = Path(inherit_from)
        if not inherit_path.is_absolute():
            inherit_path = (config_path.parent / inherit_path).resolve()
        cfg = load_config(inherit_path)
    else:
        cfg = {}

    update_recursive(cfg, cfg_special)
    return cfg


def canonicalize_object_class(name):
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    key = OBJECT_ALIASES.get(key, key)
    if key not in OBJECT_SPECS:
        supported = ", ".join(sorted(OBJECT_SPECS))
        raise ValueError(
            f"Unsupported object class '{name}'. Supported classes: {supported}"
        )
    return key


def run_command(command, cwd):
    print()
    print("[run_tum_scene_object_flow] " + " ".join(str(part) for part in command))
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="" if completed.stderr.endswith("\n") else "\n")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}"
        )
    return completed


def parse_prediction_output(stdout):
    predicted_match = re.search(r"^predicted:\s+(.+)$", stdout, flags=re.MULTILINE)
    confidence_match = re.search(r"^confidence:\s+([0-9.]+)$", stdout, flags=re.MULTILINE)
    top_predictions = []
    top_pred_pattern = re.compile(r"^\s*(\d+)\.\s+(.+?)\s+\(([0-9.]+)\)$", flags=re.MULTILINE)
    for match in top_pred_pattern.finditer(stdout):
        top_predictions.append(
            {
                "rank": int(match.group(1)),
                "label": match.group(2),
                "score": float(match.group(3)),
            }
        )

    return {
        "predicted_label": predicted_match.group(1).strip() if predicted_match else None,
        "confidence": float(confidence_match.group(1)) if confidence_match else None,
        "top_predictions": top_predictions,
    }


def select_track(metadata_path, requested_track_id=None):
    with Path(metadata_path).open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    tracks = metadata.get("tracks", [])
    if not tracks:
        raise RuntimeError(
            f"No instance tracks found in {metadata_path}. "
            "Make sure mask generation ran with --separate-instances."
        )

    if requested_track_id is not None:
        for track in tracks:
            if int(track["track_id"]) == requested_track_id:
                return track, metadata
        raise ValueError(
            f"Track id {requested_track_id} was not found. "
            f"Available track ids: {[int(track['track_id']) for track in tracks]}"
        )

    best_track = max(tracks, key=lambda item: int(item.get("frames_saved", 0)))
    return best_track, metadata


def ensure_exists(path, description):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def build_argparser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the TUM end-to-end flow for one scene and one supported object class: "
            "TSDF reconstruction -> Mask R-CNN masks -> refined object fusion -> point-cloud recognition."
        )
    )
    parser.add_argument("--scene", choices=sorted(SCENE_TO_CONFIG), required=True)
    parser.add_argument("--object-class", required=True, help="Supported classes: " + ", ".join(sorted(OBJECT_SPECS)))
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run the internal scripts.",
    )
    parser.add_argument("--output-root", default=str(TSDF_ROOT / "integration_runs"))
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--mask-device", default="cuda")
    parser.add_argument("--predict-device", default="cuda")
    parser.add_argument("--mask-score-threshold", type=float, default=0.6)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--track-id", type=int, default=None, help="Optional instance track id. Defaults to the longest track.")
    parser.add_argument("--num-votes", type=int, default=5)
    parser.add_argument(
        "--checkpoint",
        default=str(TSDF_ROOT / "model" / "pointnet2" / "pointnet2_best.pth"),
        help="Classifier checkpoint used for the final recognition step.",
    )
    parser.add_argument(
        "--labels",
        default=str(TSDF_ROOT / "model" / "pointnet2" / "labels.txt"),
        help="Label file aligned with --checkpoint.",
    )
    parser.add_argument(
        "--model",
        default="auto",
        help="Model name for detection/predict_pointcloud.py. Defaults to auto.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing stage outputs inside the run directory instead of recomputing them.",
    )
    return parser


def main():
    args = build_argparser().parse_args()

    scene = args.scene
    canonical_object = canonicalize_object_class(args.object_class)
    object_spec = OBJECT_SPECS[canonical_object]
    config_path = SCENE_TO_CONFIG[scene]
    config = load_config(config_path)
    dataset_path = Path(config["Dataset"]["dataset_path"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"{scene}_{canonical_object}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tsdf_output = run_dir / f"{scene}_tsdf_scene.pcd"
    mask_output_dir = run_dir / "masks"
    fused_output = run_dir / f"{scene}_{canonical_object}_object_refined.pcd"
    summary_path = run_dir / "integration_report.json"

    ensure_exists(config_path, "Config file")
    ensure_exists(dataset_path, "Dataset directory")
    ensure_exists(args.checkpoint, "Classifier checkpoint")
    ensure_exists(args.labels, "Classifier labels")

    report = {
        "scene": scene,
        "config": str(config_path),
        "dataset": str(dataset_path),
        "requested_object_class": args.object_class,
        "canonical_object_class": canonical_object,
        "mask_class": object_spec["mask_class"],
        "expected_classifier_label": object_spec["expected_label"],
        "run_dir": str(run_dir),
        "timings_sec": {},
        "outputs": {
            "tsdf_scene_pcd": str(tsdf_output),
            "mask_output_dir": str(mask_output_dir),
            "fused_object_pcd": str(fused_output),
        },
    }

    t0 = time.perf_counter()

    stage_start = time.perf_counter()
    if args.reuse_existing and tsdf_output.exists():
        print(f"[run_tum_scene_object_flow] Reusing existing reconstruction: {tsdf_output}")
    else:
        run_command(
            [
                args.python,
                str(TSDF_ROOT / "3d_construction" / "run_tum_rgbd_tsdf.py"),
                "--config",
                str(config_path),
                "--output",
                str(tsdf_output),
            ],
            cwd=TSDF_ROOT,
        )
    report["timings_sec"]["reconstruction"] = time.perf_counter() - stage_start

    metadata_path = mask_output_dir / "metadata.json"
    stage_start = time.perf_counter()
    if args.reuse_existing and metadata_path.exists():
        print(f"[run_tum_scene_object_flow] Reusing existing mask output: {mask_output_dir}")
    else:
        run_command(
            [
                args.python,
                str(TSDF_ROOT / "mask_generation" / "generate_tum_masks_maskrcnn.py"),
                "--config",
                str(config_path),
                "--target-class",
                object_spec["mask_class"],
                "--output-dir",
                str(mask_output_dir),
                "--separate-instances",
                "--save-preview",
                "--frame-stride",
                str(args.frame_stride),
                "--device",
                args.mask_device,
                "--score-threshold",
                str(args.mask_score_threshold),
                "--mask-threshold",
                str(args.mask_threshold),
                *([] if args.max_frames is None else ["--max-frames", str(args.max_frames)]),
            ],
            cwd=TSDF_ROOT,
        )
    report["timings_sec"]["mask_generation"] = time.perf_counter() - stage_start

    selected_track, mask_metadata = select_track(metadata_path, requested_track_id=args.track_id)
    selected_track_id = int(selected_track["track_id"])
    selected_mask_list = Path(selected_track["mask_list"])
    if not selected_mask_list.is_absolute():
        selected_mask_list = (mask_output_dir / selected_mask_list.name).resolve()
    selected_mask_list = ensure_exists(selected_mask_list, "Selected mask track list")

    report["mask_metadata"] = {
        "frames_total": int(mask_metadata.get("frames_total", 0)),
        "frames_with_detection": int(mask_metadata.get("frames_with_detection", 0)),
        "frames_without_detection": int(mask_metadata.get("frames_without_detection", 0)),
        "frames_saved": int(mask_metadata.get("frames_saved", 0)),
        "selected_track_id": selected_track_id,
        "selected_track_frames_saved": int(selected_track.get("frames_saved", 0)),
        "selected_mask_list": str(selected_mask_list),
        "available_tracks": [
            {
                "track_id": int(track["track_id"]),
                "frames_saved": int(track.get("frames_saved", 0)),
            }
            for track in mask_metadata.get("tracks", [])
        ],
    }

    stage_start = time.perf_counter()
    if args.reuse_existing and fused_output.exists():
        print(f"[run_tum_scene_object_flow] Reusing existing fused object cloud: {fused_output}")
    else:
        run_command(
            [
                args.python,
                str(TSDF_ROOT / "3d_construction" / "fuse_tum_mask_object_pcd_refined.py"),
                "--config",
                str(config_path),
                "--mask-dir",
                str(mask_output_dir / "masks"),
                "--mask-list",
                str(selected_mask_list),
                "--output",
                str(fused_output),
                "--largest-component",
                "--frame-remove-radius-outlier",
                "--remove-statistical-outlier",
                "--global-remove-radius-outlier",
                "--frame-cluster-mode",
                "largest",
                "--global-cluster-mode",
                "largest",
            ],
            cwd=TSDF_ROOT,
        )
    report["timings_sec"]["fusion"] = time.perf_counter() - stage_start

    refined_report_path = fused_output.with_suffix(".refined_report.json")
    if refined_report_path.exists():
        with refined_report_path.open("r", encoding="utf-8") as handle:
            report["fusion_report"] = json.load(handle)
        report["outputs"]["fusion_report_json"] = str(refined_report_path)

    stage_start = time.perf_counter()
    prediction_run = run_command(
        [
            args.python,
            str(TSDF_ROOT / "detection" / "predict_pointcloud.py"),
            "--input",
            str(fused_output),
            "--checkpoint",
            str(args.checkpoint),
            "--labels",
            str(args.labels),
            "--model",
            args.model,
            "--device",
            args.predict_device,
            "--num-votes",
            str(args.num_votes),
        ],
        cwd=TSDF_ROOT,
    )
    report["timings_sec"]["recognition"] = time.perf_counter() - stage_start

    prediction = parse_prediction_output(prediction_run.stdout)
    prediction["model"] = args.model
    prediction["checkpoint"] = str(args.checkpoint)
    prediction["labels"] = str(args.labels)
    prediction["matches_expected"] = (
        prediction["predicted_label"] == object_spec["expected_label"]
        if prediction["predicted_label"] is not None
        else False
    )
    report["prediction"] = prediction

    report["timings_sec"]["total"] = time.perf_counter() - t0
    report["end_to_end_success"] = bool(prediction["matches_expected"])

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print()
    print("[run_tum_scene_object_flow] Finished.")
    print(f"[run_tum_scene_object_flow] Scene: {scene}")
    print(f"[run_tum_scene_object_flow] Object: {canonical_object}")
    print(
        "[run_tum_scene_object_flow] Selected track: "
        f"{selected_track_id} ({int(selected_track.get('frames_saved', 0))} frames)"
    )
    print(
        "[run_tum_scene_object_flow] Prediction: "
        f"{prediction['predicted_label']} "
        f"(expected {object_spec['expected_label']}, confidence={prediction['confidence']})"
    )
    print(
        "[run_tum_scene_object_flow] End-to-end success: "
        f"{report['end_to_end_success']}"
    )
    print(f"[run_tum_scene_object_flow] Summary report: {summary_path}")


if __name__ == "__main__":
    main()
