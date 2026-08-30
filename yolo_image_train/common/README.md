# yolo_image_train 数据准备

`prepare_yolo_actionmixed.py` 解压 large/small 两组 YOLO 权重，对 ActionMixed 图片推理，映射并合并成统一 8 类
`frames/{split}/*.txt`，同时复用图片和动作标签，供 `run_yolo_image_train.py` 调用。

完整命令、参数和数据结构见 `yolo_image_train/README.md` 及仓库根相对路径
`experiments/04_real_yolo_rgb_roi/README.md`。
