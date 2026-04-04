# data_preprocess

用于对 `ScanObjectNN` 的 `h5` 数据做较温和的点云退化，模拟真实提取点云中常见的：

- 随机删点
- 局部缺块

这里默认参数偏保守，目标是让训练数据更接近实际提取的椅子点云质量，但不把物体整体形状破坏得太严重。

## 1. 生成处理后的 h5

```bash
python data_preprocess/process_scanobjectnn.py \
  --scanobjectnn-root data/ScanObjectNN \
  --variant pb_t50_rs \
  --split both \
  --output-root data/ScanObjectNN_mild
```

默认会输出到：

- `data/ScanObjectNN_mild/main_split/`

如果你想覆盖已有结果，加：

```bash
python data_preprocess/process_scanobjectnn.py \
  --scanobjectnn-root data/ScanObjectNN \
  --variant pb_t50_rs \
  --split both \
  --output-root data/ScanObjectNN_mild \
  --overwrite
```

## 2. 可视化 h5

只看一个 h5：

```bash
python data_preprocess/visualize_h5.py \
  --h5 data/ScanObjectNN_mild/main_split/training_objectdataset_augmentedrot_scale75.h5 \
  --index 0
```

对比原始和处理后：

```bash
python data_preprocess/visualize_h5.py \
  --h5 data/ScanObjectNN/main_split/training_objectdataset_augmentedrot_scale75.h5 \
  --compare-h5 data/ScanObjectNN_mild/main_split/training_objectdataset_augmentedrot_scale75.h5 \
  --index 0
```

显示时：

- 蓝色：`--h5`
- 红色：`--compare-h5`
- 红色点云会沿 `+X` 方向平移，方便人工对比

## 3. 默认退化强度

- `random-drop-min = 0.03`
- `random-drop-max = 0.08`
- `local-drop-min = 0.05`
- `local-drop-max = 0.12`
- `min-keep-ratio = 0.82`

这组参数属于“轻到中等”的退化。如果后面你想更接近真实重建点云，可以再往上微调，但建议一步一步加，不要一次加太猛。

## 4. 生成处理后的 ModelNet40 h5

如果你的 `ModelNet40` 是原始 `OFF` 目录版：

```bash
python data_preprocess/process_modelnet40.py \
  --modelnet40-root data/ModelNet40 \
  --input-format off \
  --split both \
  --output-root data/ModelNet40_mild
```

如果你的 `ModelNet40` 已经是 `h5` 版 `modelnet40_ply_hdf5_2048`：

```bash
python data_preprocess/process_modelnet40.py \
  --modelnet40-root data/ModelNet40 \
  --input-format h5 \
  --split both \
  --output-root data/ModelNet40_mild
```

默认会输出到：

- `data/ModelNet40_mild/modelnet40_ply_hdf5_2048/ply_data_train0.h5`
- `data/ModelNet40_mild/modelnet40_ply_hdf5_2048/ply_data_test0.h5`

同时会写出：

- `shape_names.txt`
- `train_files.txt`
- `test_files.txt`

如果输入是 `OFF`，脚本会先从网格采样点云，再做和 `ScanObjectNN` 类似的轻度退化处理。

## 5. 可视化 ModelNet40

只看一个 `OFF` 样本：

```bash
python data_preprocess/visualize_modelnet40.py \
  --input data/ModelNet40/chair/train/chair_0001.off
```

只看一个处理后的 `h5` 样本：

```bash
python data_preprocess/visualize_modelnet40.py \
  --input data/ModelNet40_mild/modelnet40_ply_hdf5_2048/ply_data_train0.h5 \
  --index 0
```

对比原始 `OFF` 和处理后的 `h5`：

```bash
python data_preprocess/visualize_modelnet40.py \
  --input data/ModelNet40/chair/train/chair_0001.off \
  --compare data/ModelNet40_mild/modelnet40_ply_hdf5_2048/ply_data_train0.h5
```

如果处理后的 `h5` 由 `process_modelnet40.py` 生成，脚本会优先根据写入的 `sample_path` 元数据自动匹配同一个样本。
