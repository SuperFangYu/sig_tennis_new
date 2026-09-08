# Qualisys 验证输入

推荐将同一次采集的三个文件分别放进 `video/`、`rtmpose/`、`qtm/`。程序不再扫描目录、
不需要 manifest，也不强制文件同名；实际使用哪个文件，由五个动作入口顶部的三条路径明确指定。

为减少选错，仍建议统一命名，例如 `fy_zs_1.avi`、`fy_zs_1.csv`、`fy_zs_1.tsv`。
QTM 目录只放 `DATA_INCLUDED=3D` 的 marker TSV，不放测力台 analog 文件。
