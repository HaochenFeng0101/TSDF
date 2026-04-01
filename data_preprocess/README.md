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
