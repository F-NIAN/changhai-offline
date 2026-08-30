# image_train v4 代码

入口：`image_train/v4/train_image_v4.py`；集中配置：`image_train/v4/configs/image_train_v4.json`。
本版本加入 proxy ROI、训练调度/early stopping，并分开报告 raw 与 postprocessed 指标。

完整命令、指标和分析见仓库根相对路径 `experiments/03_rgb_roi/04_v4_proxy_roi/README.md`；
原始结果见 `image_train/image_train_v4.md` 和 `image_train/output_v4/image_train_v4.json`。
