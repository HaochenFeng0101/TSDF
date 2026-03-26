# TSDF

This project contains lightweight tools for:

- downloading indoor RGB-D datasets
- reconstructing scene point clouds with TSDF
- generating 2D masks with `Mask R-CNN`
- fusing masked RGB-D observations into object point clouds
- training and validating point cloud classifiers

## Folders

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

## Main Scripts

### `dataset/`

- `download_tum_rgbd_samples.py`
  Download a few TUM RGB-D indoor sample scenes.
- `download_scanobjectnn.py`
  Download `ScanObjectNN`.
- `download_modelnet40_kaggle.py`
  Download `ModelNet40` from Kaggle.
- `download_scannet_scene.py`
  Wrapper around the official ScanNet downloader and SensReader exporter.
- `scanobjectnn_data.py`
  ScanObjectNN dataset loader.
- `modelnet40_data.py`
  ModelNet40 dataset loader.

### `3d_construction/`

- `run_tum_rgbd_tsdf.py`
  Reconstruct a full TUM RGB-D scene into a scene `pcd`.
- `run_scannet_tsdf.py`
  Reconstruct an exported ScanNet scene into a scene `pcd`.
- `fuse_tum_mask_object_pcd.py`
  Fuse multi-frame TUM masks plus depth plus pose into object-level `pcd` files.
  It also supports batch fusion of all `mask_track_*.txt` files into one `time_fuse_*` folder.

### `mask_generation/`

- `generate_tum_masks_maskrcnn.py`
  Run `torchvision Mask R-CNN` on a TUM RGB-D sequence and save masks.
- `generate_scannet_masks_maskrcnn.py`
  Run `torchvision Mask R-CNN` on an exported ScanNet scene and save masks.

### `detection/`

- `train_pointnet_cls.py`
  Train PointNet classification.
- `validate_pointnet_sample.py`
  Inspect one PointNet prediction on ScanObjectNN.
- `predict_pointnet_pointcloud.py`
  Run a trained PointNet checkpoint on one `.off` / point cloud file and optionally visualize it.
- `find_and_classify_object_pcd.py`
  Cluster a scene point cloud and classify each cluster.
- `train_pointmlp_cls.py`
  Wrapper entrypoint for PointMLP training.
- `validate_pointmlp_sample.py`
  Wrapper entrypoint for PointMLP validation.
- `pointnet_model.py`
  PointNet model definition.
- `pointmlp/`
  PointMLP implementation.
  - `model.py`
  - `train.py`
  - `validate.py`

## Environment

Two environments were used during this project:

- `MonoGS`
  Older environment used earlier for reconstruction experiments.
- `tsdf`
  Main environment for current TSDF / mask / classifier scripts.

There is also:

- `environment_py310.yml`
  Python 3.10 environment file for the TSDF project.

## Typical Workflow

### 1. Download TUM RGB-D samples

```bash
cd /TSDF
conda activate tsdf

python dataset/download_tum_rgbd_samples.py
```

This downloads sample scenes under:

- `data/tum/`

### 2. Reconstruct a TUM scene point cloud

```bash
python 3d_construction/run_tum_rgbd_tsdf.py \
  --config configs/rgbd/tum/fr3_office.yaml
```

Default output:

- `3d_construction/outputs/fr3_office.pcd`

### 3. Generate masks for a target class from TUM RGB-D

Example: generate `chair` masks.

```bash
python mask_generation/generate_tum_masks_maskrcnn.py \
  --config configs/rgbd/tum/fr3_office.yaml \
  --target-class chair \
  --max-frames 100 \
  --device cuda
```

If you want to split same-class detections into separate instance tracks:

```bash
python mask_generation/generate_tum_masks_maskrcnn.py \
  --config configs/rgbd/tum/fr3_office.yaml \
  --target-class chair \
  --separate-instances \
  --save-preview \
  --max-frames 100 \
  --device cuda
```

Default output folder:

- `mask_generation/outputs/fr3_office_chair/`

Contents usually include:

- `masks/`
- `mask.txt`
- `detections.jsonl`
- `metadata.json`

If `--separate-instances` is used, it also creates:

- `mask_track_000.txt`
- `mask_track_001.txt`
- ...

### 4. Fuse one target object point cloud from TUM masks

Single merged object:

```bash
python 3d_construction/fuse_tum_mask_object_pcd.py \
  --config configs/rgbd/tum/fr3_office.yaml \
  --mask-dir mask_generation/outputs/fr3_office_chair/masks \
  --mask-list mask_generation/outputs/fr3_office_chair/mask.txt \
  --largest-component \
  --voxel-downsample 0.005 \
  --remove-statistical-outlier
```

Batch fuse all separate tracks into one new timestamped folder:

```bash
python 3d_construction/fuse_tum_mask_object_pcd.py \
  --config configs/rgbd/tum/fr3_office.yaml \
  --mask-dir mask_generation/outputs/fr3_office_chair/masks \
  --fuse-all-tracks \
  --output-dir 3d_construction/outputs/time_fuse_chair_custom \
  --largest-component \
  --voxel-downsample 0.005 \
  --remove-statistical-outlier
```

This creates one `pcd` per track under the target output folder.

### 5. Download ScanObjectNN

```bash
python dataset/download_scanobjectnn.py
```

Default data folder:

- `data/ScanObjectNN`

### OpenMVS Workflow For TUM RGB-D

If you want to try OpenMVS on a TUM RGB-D sequence, use the helper folder:

```bash
bash openmvs/setup_openmvs.sh

python openmvs/export_tum_to_openmvs.py --config configs/rgbd/tum/fr3_office.yaml --workspace-name fr3_office_openmvs --frame-stride 10 --max-frames 120

bash openmvs/run_openmvs_tum.sh --workspace-name fr3_office_openmvs --reexport --frame-stride 10 --max-frames 120
```

The exported workspace is created under:

- `openmvs/workspaces/fr3_office_openmvs/`

Final OpenMVS files are kept in the workspace folder itself.

Run logs are written to:

- `log/oppenmvs/fr3_office_openmvs/`

### 6. Train PointNet

```bash
python detection/train_pointnet_cls.py \
  --dataset-type scanobjectnn
```

Default outputs:

- `model/pointnet/pointnet_best.pth`
- `model/pointnet/pointnet_last.pth`
- `model/pointnet/labels.txt`

### 7. Validate PointNet on one sample

```bash
python detection/validate_pointnet_sample.py \
  --checkpoint model/pointnet/pointnet_best.pth \
  --scanobjectnn-root data/ScanObjectNN \
  --scanobjectnn-variant pb_t50_rs \
  --use-all-points
```

### 7b. Download ModelNet40 from Kaggle

This uses the Kaggle CLI, so the server must already have Kaggle credentials configured.

```bash
python dataset/download_modelnet40_kaggle.py
```

Default data folder:

- `data/ModelNet40`

### 7c. Train PointNet on ModelNet40

```bash
python detection/train_pointnet_cls.py \
  --dataset-type modelnet40 \
  --modelnet40-root data/ModelNet40 \
  --epochs 150 \
  --batch-size 32 \
  --num-points 1024
```

Optional:

- `--modelnet40-sample-method surface`
- `--use-class-weights`
- `--device cuda`

Outputs:

- `model/pointnet/pointnet_best.pth`
- `model/pointnet/pointnet_last.pth`
- `model/pointnet/labels.txt`

### 7d. Predict and visualize one ModelNet40 or point cloud sample with PointNet

```bash
python detection/predict_pointnet_pointcloud.py \
  --input data/ModelNet40/airplane/test/airplane_0627.off \
  --checkpoint model/pointnet/pointnet_best.pth \
  --labels model/pointnet/labels.txt \
  --visualize
```

### 8. Train PointMLP

Recommended lightweight setting on a small GPU:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python detection/pointmlp/train.py \
  --dataset-type scanobjectnn \
  --scanobjectnn-root data/ScanObjectNN \
  --scanobjectnn-variant pb_t50_rs \
  --model-type pointmlpelite \
  --optimizer sgd \
  --lr 0.01 \
  --momentum 0.9 \
  --epochs 200 \
  --batch-size 4 \
  --num-points 512 \
  --use-class-weights \
  --amp
```

Default outputs:

- `model/pointmlp/pointmlp_best.pth`
- `model/pointmlp/pointmlp_last.pth`
- `model/pointmlp/labels.txt`

### 9. Validate PointMLP and visualize one sample

```bash
python detection/validate_pointmlp_sample.py \
  --checkpoint model/pointmlp/pointmlp_best.pth \
  --scanobjectnn-root data/ScanObjectNN \
  --scanobjectnn-variant pb_t50_rs \
  --visualize
```

Optional:

- `--index 12`
- `--use-all-points`
- `--visualize-raw-points`

## ScanNet Notes

`ScanNet` download is not anonymous. You must first get access from:

- http://www.scan-net.org/

Then use the official downloader with:

```bash
python dataset/download_scannet_scene.py \
  --scene-id scene0000_00 \
  --official-downloader /path/to/ScanNet/download-scannet.py \
  --reader-script /path/to/ScanNet/SensReader/python/reader.py \
  --reader-python python2 \
  --downloader-enter-twice
```

After export, you can:

- reconstruct the scene with `3d_construction/run_scannet_tsdf.py`
- generate masks with `mask_generation/generate_scannet_masks_maskrcnn.py`

Example:

```bash
python mask_generation/generate_scannet_masks_maskrcnn.py \
  --config configs/rgbd/scannet/scene0000_00.yaml \
  --target-class chair \
  --separate-instances \
  --save-preview
```

## Notes

- `wandb/` is ignored by git.
- `data/` is ignored by git.
- `mask_generation` scripts only save masks when the target object is detected.
- `fuse_tum_mask_object_pcd.py` can use either:
  - `mask.txt` for one merged object
  - `mask_track_*.txt` for separate instance fusion
