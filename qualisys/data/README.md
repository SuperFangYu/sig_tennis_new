# Qualisys 本地实验数据

- `input/video/`：按 `英文名_英文动作_编号` 命名的同次采集视频。
- `input/qtm/`：与视频同名的 Qualisys 3D TSV，不放测力台文件。
- `intermediate/`：两套独立 8 角 CSV 与人工事件时间表。
- `output/`：对齐明细、角度指标、事件误差和对齐质控表。

这三个目录中的真实视频、TSV、CSV 和结果均被 `.gitignore` 排除；仅目录说明与示例配置
进入版本库。测力台文件不属于本模块输入。
