# TSDF

## Environment

Create the conda environment:

```bash
conda env create -n tsdf -f environment_py310.yml
conda activate tsdf
```

If the environment already exists, update it:

```bash
conda env update -n tsdf -f environment_py310.yml --prune
conda activate tsdf
```

Run every command from the repository root:

```bash
cd /path/to/TSDF
```

## Project Structure

- `3d_construction/`
  Scene reconstruction and object point cloud fusion.
- `configs/`
  Config files for TUM RGB-D and ScanNet scenes.
- `data/`
  Downloaded datasets. This folder is ignored by git.
- `dataset/`
  Dataset download and loading utilities.
- `detection/`
  PointNet / PointMLP training, validation, and object classification scripts.
- `mask_generation/`
  Scripts that generate per-frame object masks from RGB images.
- `model/`
  Saved checkpoints and labels.
- `scripts/`
  Extra utility scripts.
- `openmvs/`
  TUM RGB-D to OpenMVS workspace export, build scripts, and OpenMVS outputs.

## 1. OpenMVS Reconstruction

Build OpenMVS once:

```bash
bash openmvs/setup_openmvs.sh
```

Run OpenMVS on a TUM scene:

```bash
bash openmvs/run_openmvs_tum.sh \
  --config configs/rgbd/tum/fr3_office.yaml \
  --workspace-name fr3_office_openmvs
```

Default output:

- `openmvs/workspaces/<workspace>/seed_from_depth.pcd`
- `openmvs/workspaces/<workspace>/scene_dense.pcd`
- `openmvs/workspaces/<workspace>/scene_mesh.pcd`
- `openmvs/workspaces/<workspace>/scene_mesh_refine.pcd`
- `openmvs/workspaces/<workspace>/scene_mesh_refine_texture.pcd`

## 2. TSDF Reconstruction

Run TSDF on a TUM scene:

```bash
python3 3d_construction/run_tum_rgbd_tsdf.py \
  --config configs/rgbd/tum/fr3_office.yaml
```

Default output:

- `3d_construction/outputs/<config_stem>.pcd`

## 3. 2D Extraction to 3D Object Point Cloud

Generate 2D masks with YOLO:

```bash
python3 mask_generation/generate_tum_masks_yolo.py \
  --config configs/rgbd/tum/fr3_office.yaml \
  --model yolov8x-seg.pt \
  --target-class chair
```

Default output:

- `mask_generation/outputs/<scene>_<class>_<model>_<timestamp>/`

Fuse masks into a 3D object point cloud:

```bash
python3 3d_construction/fuse_tum_mask_object_pcd.py \
  mask_generation/outputs/<scene>_<class>_<model>_<timestamp>
```

If you want one specific tracked instance:

```bash
python3 3d_construction/fuse_tum_mask_object_pcd.py \
  mask_generation/outputs/<scene>_<class>_<model>_<timestamp> \
  --track-id 0
```

Default output:

- `3d_construction/outputs/fuse_obj_<mask_folder>_<timestamp>/fused_object.pcd`

## 4. Semantic Segmentation

Run PointNet++ semantic segmentation on one scene:

```bash
python3 detection/pointnet2/validate_seg.py \
  --pcd 3d_construction/outputs/fr3_office.pcd \
  --checkpoint model/pointnet2_seg/pointnet2_seg_best.pth \
  --labels model/pointnet2_seg/labels.txt \
  --visualize
```

Default output:

- `model/pointnet2_seg/inference_outputs/single_scene_<timestamp>/`

Run PointNet++ semantic segmentation on a dataset split and save IoU results:

```bash
python3 detection/pointnet2/validate_seg.py \
  --data-root data/S3DIS_seg \
  --split val \
  --checkpoint model/pointnet2_seg/pointnet2_seg_best.pth \
  --labels model/pointnet2_seg/labels.txt
```

Default output:

- `model/pointnet2_seg/inference_outputs/dataset_<split>_<timestamp>/dataset_metrics.json`
- `model/pointnet2_seg/inference_outputs/dataset_<split>_<timestamp>/iou_percent.txt`

Run PointMLP semantic segmentation on one scene:

```bash
python3 detection/validate_pointmlp_seg.py \
  --pcd 3d_construction/outputs/fr3_office.pcd \
  --checkpoint seg_model/pointmlp/pointmlp_seg_best.pth \
  --labels seg_model/pointmlp/labels.txt \
  --visualize
```

Default output:

- `seg_model/pointmlp/inference_outputs/`

## 5. Classification Training

Train PointNet:

```bash
python3 detection/pointnet/train.py \
  --dataset-type modelnet40
```

Default output:

- `model/pointnet/pointnet_best.pth`
- `model/pointnet/labels.txt`

Train PointMLP:

```bash
python3 detection/pointmlp/train.py \
  --dataset-type modelnet40
```

Default output:

- `model/pointmlp/pointmlp_best_weights.pth`
- `model/pointmlp/labels.txt`

Train PointNet++ classification:

```bash
python3 detection/pointnet2/train.py \
  --dataset-type modelnet40
```

Default output:

- `model/pointnet2/pointnet2_best.pth`
- `model/pointnet2/labels.txt`

## 6. Visualize and Classify Your Own Point Cloud

Visualize and classify one custom object with PointNet:

```bash
python3 detection/pointnet/validate_own_object.py \
  3d_construction/outputs/fuse_obj_xxx/fused_object.pcd \
  --use-all-points
```

Visualize and classify one custom object with PointMLP:

```bash
python3 detection/validate/validate_pointmlp_own_object.py \
  3d_construction/outputs/fuse_obj_xxx/fused_object.pcd \
  --use-all-points
```

Visualize and classify one custom object with PointNet++:

```bash
python3 detection/validate/validate_pointnet2_own_object.py \
  3d_construction/outputs/fuse_obj_xxx/fused_object.pcd \
  --use-all-points
```

These commands use the default checkpoints under:

- `model/pointnet/`
- `model/pointmlp/`
- `model/pointnet2/`

## 7. Compare TSDF and OpenMVS

Compare TSDF reconstruction against the default final OpenMVS geometry:

```bash
python3 3d_construction/compare_tsdf_openmvs_no_gt.py \
  --tsdf-pcd 3d_construction/outputs/fr3_office.pcd \
  --openmvs-workspace openmvs/workspaces/fr3_office_openmvs
```

Default output:

- `3d_construction/eval/comparison_report.json`
- `3d_construction/eval/comparison_summary.txt`
- `3d_construction/eval/overview_views.png`
- `3d_construction/eval/consistency_metrics.png`
- `3d_construction/eval/openmvs_slam_view_metrics.png`

## 8. One Shell Script: 2D Detection -> 3D Object -> Classification

Run the full object detection pipeline from the repository root:

```bash
bash scripts/detect_object_from_tsdf.sh \
  --config configs/rgbd/tum/fr3_office.yaml \
  --target-class chair \
  --classifier pointnet2
```

Choose another classifier:

```bash
bash scripts/detect_object_from_tsdf.sh \
  --config configs/rgbd/tum/fr3_office.yaml \
  --target-class chair \
  --classifier pointnet2
```

Pick one tracked instance:

```bash
bash scripts/detect_object_from_tsdf.sh \
  --config configs/rgbd/tum/fr3_office.yaml \
  --target-class chair \
  --classifier pointnet2 \
  --track-id 0
```

This script will:

1. generate 2D masks with YOLO
2. fuse them into a 3D object point cloud
3. classify the fused object with the selected classifier

Default outputs:

- masks: `mask_generation/outputs/<scene>_<class>_<model>_<timestamp>/`
- fused object: `3d_construction/outputs/fuse_obj_<scene>_<class>_<timestamp>/fused_object.pcd`
