# 验证输出

每次运行都会新建且不会覆盖旧结果：

```text
output/{动作}/{视频主文件名}/{YYYYMMDD_HHMMSS_mmm}/
```

主要文件：

- `run_status.json`：输入路径、参数、完成状态；失败时记录原因。
- `alignment_qc.csv` / `alignment_qc.png`：自动峰值对齐的质量检查。
- `alignment_anchors.csv`：五个视频/QTM 拍头速度锚点及时间残差。
- `repetition_windows.csv`：自动识别的五次动作起止时间。
- `qualisys_8_angles.csv`：Qualisys ZY 平面同定义八角。
- `aligned_angles.csv`：逐视频帧的两套角度与误差。
- `angle_metrics.csv`：每次动作及合并结果的 MAE、RMSE、bias、LoA、相关和 CCC。
- `anchor_angle_errors.csv`：拍头速度峰时刻的八角误差；它不是触球角。
- `video_racket_alignment.csv` / `qualisys_racket_alignment.csv`：对齐信号与峰值标记。

这些结果是描述性效度统计。正式总体推断仍需按受试者/试次处理重复测量结构。
