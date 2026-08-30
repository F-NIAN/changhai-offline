# yolo_image_train

完整交接时间线、逐版本命令和结果分析见 `experiments/04_real_yolo_rgb_roi/README.md`（路径相对仓库根目录）。

本文件夹用于复刻 `image_train` v1-v4，但检测输入不再使用 ActionMixed 数据集自带的 `frames/{split}/*.txt` 手工/导出 YOLO 框，而是：

```text
ActionMixed images/{split}/*.jpg
  -> yolo/clean-large-v0.3.zip 内的 large YOLO
  -> yolo/clean-small-v0.3.zip 内的 small YOLO
  -> 合并成新的 frames/{split}/*.txt
  -> 复用 ActionMixed labels/{split}/*.txt 时序动作标签
  -> 调用 image_train v1-v4 训练/测试
```

## 1. YOLO 模型类别

`yolo/clean-large-v0.3.zip`：

| local id | 类别 | ActionMixed global id |
|---:|---|---:|
| 0 | `hand` | 0 |
| 1 | `scope_control_body` | 1 |
| 2 | `scope_mid_section` | 2 |

`yolo/clean-small-v0.3.zip`：

| local id | 类别 | ActionMixed global id |
|---:|---|---:|
| 0 | `syringe` | 4 |
| 1 | `air_gun` | 5 |
| 2 | `scope_distal_end` | 3 |
| 3 | `short_brush` | 6 |
| 4 | `brush_tip_out` | 7 |

两个模型合并后覆盖现有 ActionMixed 检测 8 类：

```text
0 hand
1 scope_control_body
2 scope_mid_section
3 scope_distal_end
4 syringe
5 air_gun
6 short_brush
7 brush_tip_out
```

## 2. 生成的数据集结构

默认输出：

```text
yolo_image_train/
  generated/
    actionmixed_yolo_v03/
      images/
        train/
        val/
        test/
      labels/
        train/
        val/
        test/
      frames/
        data.yaml
        train/
        val/
        test/
      yolo_generation_summary.json
```

其中：

- `images/` 默认使用硬链接指向原始 ActionMixed 图片；失败时复制。
- `labels/` 默认使用硬链接/复制原始动作标签。
- `frames/` 是两个 YOLO 模型对图片推理后生成的新检测结果。
- 每个 frame txt 行格式仍为 `class_id cx cy w h conf`，坐标为 0-1 归一化。

## 3. 依赖

当前训练代码只依赖已有 PyTorch；但 YOLO 推理需要额外安装：

```powershell
python -m pip install ultralytics
```

如果只想先检查目录结构和解压模型，不跑 YOLO：

```powershell
python .\yolo_image_train\common\prepare_yolo_actionmixed.py --skip-inference
```

## 4. 只生成 YOLO 推理版数据集

```powershell
python .\yolo_image_train\common\prepare_yolo_actionmixed.py `
  --conf 0.25 `
  --iou 0.7 `
  --imgsz 640
```

调试时可以限制帧数：

```powershell
python .\yolo_image_train\common\prepare_yolo_actionmixed.py `
  --limit-frames 20 `
  --conf 0.25
```

## 5. 复刻 image_train v1-v4

完整复刻：

```powershell
python .\yolo_image_train\run_yolo_image_train.py
```

只跑准备，不训练：

```powershell
python .\yolo_image_train\run_yolo_image_train.py --prepare-only
```

只跑某个版本和某个模型：

```powershell
python .\yolo_image_train\run_yolo_image_train.py `
  --versions v4 `
  --models bigru
```

如果已经生成好了 `generated/actionmixed_yolo_v03/frames`，可以跳过 YOLO 推理准备：

```powershell
python .\yolo_image_train\run_yolo_image_train.py `
  --skip-prepare `
  --versions v1 v2 v3 v4
```

## 6. 输出

训练输出默认写在：

```text
yolo_image_train/
  output_v1/
  output_v2/
  output_v3/
  output_v4/
  yolo_image_train_v1.md
  yolo_image_train_v2.md
  yolo_image_train_v3.md
  yolo_image_train_v4.md
  yolo_image_train_run_summary.json
```

这条实验线不会覆盖 `image_train/image_train_v1.md` 到 `image_train/image_train_v4.md`。

## 7. 当前注意点

- 这条线训练时仍复用 ActionMixed 的动作时序标注，只替换检测框来源。
- 当前数据仍是切分片段，不是完整视频级训练。
- large/small 两个 YOLO 模型类别互补，脚本会把 local class id 映射到 ActionMixed global class id。
- 如果 YOLO 推理阈值变化，必须重新生成 `frames/`，否则不同实验不可直接比较。
- 第一次 YOLO 推理会处理约 9500 张图，可能比较耗时。
