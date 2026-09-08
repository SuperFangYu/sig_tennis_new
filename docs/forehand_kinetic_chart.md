# 正手动力链图与持拍手

## 后端

- 切分与作图入口：[`algorithm/forehand/segmentation.py`](../algorithm/forehand/segmentation.py)，绘图封装在 [`algorithm/forehand/plot_kinetic_chain.py`](../algorithm/forehand/plot_kinetic_chain.py)。
- 每个通过筛选的正手区间导出一张 PNG：`{视频名}_kinetic_chain_{序号}.png`，与切分视频片段序号一致。
- **时间窗与 MP4 一致**：角度曲线与阶段着色使用与 `forehand_intervals` 相同的 `start_t`/`end_t`（含动态缓冲与相邻峰裁剪），经 `searchsorted` 得到帧索引后再切片，避免仅用 `left_bases`/`right_bases` 导致 X 轴远长于切分视频。
- 对切片后的膝/肩髋/肘角度做 **Savitzky-Golay** 平滑（`fps≥26` 时 `window_length=15`，否则 `11`，`polyorder=3`；段长不足则跳过滤波）。
- 击球时刻仍参与底色绿区计算；**击球瞬间竖虚线当前已注释、暂不绘制**。作图函数 **`ax.set_xlim(segment_time[0], segment_time[-1])`** 锁死横轴。
- **三阶段底色**：不再用掉拍头最低点 `p` 分界；以击球时刻为中心，`hit_window = clip(clip_duration×0.2, 0.2, 0.4)` 秒，绿区为 `[contact−0.65·hit_window, contact+0.35·hit_window]` 与切片求交；金区 `[time_start, t_green_start]`，蓝区 `[t_green_end, time_end]`。
- 击球帧（索引）：在波峰索引 `p` 与 `end_idx` 闭区间内，对拍头速度 `v = sqrt((∂x)^2 + (∂y)^2)` 取最大值；其中 `∂x = np.gradient(x_clean)`，`∂y = np.gradient(y_clean)`。
- 每张图保存后调用 `plt.close(fig)`，避免内存泄漏与画布叠加。
- `POST /api/forehand/analyze` 支持表单字段 `handedness`：`right`（默认）或 `left`。左手时使用镜像后的 X 位移判据，并读取 `left_knee_angle`、`left_elbow_angle`（需使用更新后的 [`algorithm/forehand/angles_csv.py`](../algorithm/forehand/angles_csv.py) 导出的身体 CSV）。

## 前端（`Vue/forehand.html`）

- 上传前可选择持拍手；分析结果中 `artifacts` 含 `kind === "kinetic_chart"` 的条目。
- 动力链区块默认收起（单行占位）；展开后 **片段 1** 默认展示大图及下载；**片段 2 及以后** 为折叠行，点击「查看大图」后再显示该 PNG 与下载链接。

## 测试

```text
python -m unittest tests.test_forehand_kinetic_chart -v
```

在项目根目录执行，且需已安装 `matplotlib`、`numpy`。
