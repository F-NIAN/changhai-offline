# 2026-07-10：ActionMixed 数据集接入

## 简单总结

当时基础链路只支持早期示例数据。本次接入 ModelScope `lhh010/cleansight-ActionMixed`，把逐帧 YOLO 框、
动作标签和 train/val/test 声明转成时序模型输入，并将标签固定为 `idle` 加五类动作。初训确认三模型都能跑完，
但简单聚合特征不足以处理双手、遮挡和目标跳变，后续需要升级特征版本。

## 数据输入

```text
input/modelscope/lhh010__cleansight-ActionMixed/
├── frames/{train,val,test}/{video}.mp4-{frame_id:06d}.txt
└── labels/{train,val,test}/{video}.mp4.txt
```

检测类别由数据 YAML 映射为 hand、scope 三部位、syringe、air_gun、short_brush、brush_tip_out；动作映射为
`idle`、`air_injection`、`flush`、`long_brush_insert`、`long_brush_withdraw`、`short_brush_cleaning`。

## 代码与命令

- 修改：`dataset.py`、`data_transfer.py`、`run_pipeline.py`。
- 文档：`docs/update_0710.md`、`output/training_summary_report.md`。

```bash
python run_pipeline.py \
  --input-source actionmixed \
  --actionmixed-root input/modelscope/lhh010__cleansight-ActionMixed \
  --models ms_tcn asformer bigru \
  --epochs 1 \
  --out-dir output
```

`--actionmixed-fps` 可覆盖默认近似采样 FPS；`--actionmixed-refresh-lfs` 可刷新 LFS 文件；数据已完整时不要使用
`--actionmixed-force-clone`，该选项会重建本地数据目录。

## 结果与后续

当时共生成 16 条序列、4501 个采样帧。结果用于证明数据接入和五类动作闭环，不作为最终指标。后续先完成
68 维完整序列对比，再通过 hand top-2、非手 top-1、短遮挡插值和关系变化构造 113 维 v2。
