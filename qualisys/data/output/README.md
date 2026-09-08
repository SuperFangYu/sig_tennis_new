# 验证输出

每个试次的 `{trial_id}/` 目录包含：

- `aligned_angles.csv`：以视频帧为观察时刻的两套角度、Qualisys 映射时间和逐帧误差。
- `angle_metrics.csv`：每次动作及合并数据的 MAE、RMSE、偏倚、LoA、相关和 CCC 等初步指标。
- `event_metrics.csv`：开始、击球、结束三个事件处的 8 角误差。
- `alignment_qc.csv`：时间映射斜率、截距、锚点残差和有效帧数，供检查是否对齐合理。

当前输出是试验性描述统计；正式论文的 ICC 置信区间、重复测量 Bland–Altman、受试者/试次
聚类 bootstrap 和图表将在试验设计及样本量确定后增加。
