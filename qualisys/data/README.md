# Qualisys 本地验证数据

- `input/video/`：同次采集的原始侧面视频。
- `input/rtmpose/`：已经由独立 RTMPose 脚本生成的 8 角 CSV。
- `input/qtm/`：Qualisys 3D marker TSV；不使用测力台文件。
- `output/`：程序按动作、样本和运行时间逐次新建的验证结果。

真实视频、TSV、CSV 和运行结果均被 `.gitignore` 排除，不会提交到 Git。
