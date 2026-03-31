PointMLP
========

This folder contains the PointMLP branch of the project:

- `common.py`: shared PointMLP building blocks.
- `pointmlp_cls.py`: object classification model for single-object point clouds.
- `pointmlp_seg.py`: scene semantic segmentation model for S3DIS-style inputs.
- `train.py`: ScanObjectNN classification training entrypoint, saving lightweight `*_weights.pth` checkpoints.
- `validate.py`: inspect one classification sample quickly.
- `train_seg.py`: train PointMLP semantic segmentation on S3DIS.
- `validate_seg.py`: run PointMLP semantic segmentation on a scene point cloud.

 
