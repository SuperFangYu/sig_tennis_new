# RTMPose–Qualisys 验证模块

本目录独立完成同一次采集的单目视频与 Qualisys 3D 轨迹对比，不依赖前端、后端或测力台。
每个 `trial_id` 必须对应真正同步采集的一个视频和一个 3D marker TSV；当前目录里名称相近但
并非同次采集的文件不能放在同一试次中比较。

## 文件职责

| 文件 | 一句话说明 |
|---|---|
| `config.py` | 保存数据目录、8 个角度列、ZY 平面与默认 marker 映射。 |
| `trials.py` | 自动配对 video/qtm 同名文件，或读取可选 `manifest.csv`。 |
| `run_rtmpose.py` | 对试次视频运行项目已有 RTMPose，独立生成 8 角 CSV。 |
| `run_qualisys.py` | 读取 3D TSV、投影到 ZY 平面并按相同三点定义生成 8 角 CSV。 |
| `compare_angles.py` | 用人工事件锚点逐动作对齐、重采样并输出初步一致性指标。 |
| `run_validation.py` | 一次运行一个或多个阶段的总入口。 |
| `data/` | 保存本地输入、中间表和最终验证结果，实验数据不会提交到 Git。 |

## 输入组织

不放测力台的 `_a_1.tsv`、`_a_2.tsv`。默认输入结构是：

```text
qualisys/data/input/
├── video/
│   ├── 张三_正手.mp4
│   ├── 张三_正手截击.mp4
│   └── 张三_发球.mp4
├── qtm/
│   ├── 张三_正手.tsv
│   ├── 张三_正手截击.tsv
│   └── 张三_发球.tsv
├── manifest.example.csv     # 可选清单格式示例
└── marker_map.example.json  # 可选点位映射格式示例
```

视频与 TSV 主文件名完全相同时自动配对，不需要创建 `manifest.csv`。文件名必须是
`人名_动作`，动作后缀支持：正手、反手、正手截击、反手截击、发球。自动模式默认右手持拍。

`manifest.example.csv` 只是可选的试次索引模板：需要设置左手、使用不同文件名、自定义 marker
映射或记录备注时，才复制为 `manifest.csv`。它不会参与角度计算，也不是程序输出。

## 推荐运行顺序

```powershell
conda activate fytennis

# 1. 分别生成视频与 Qualisys 八角 CSV
python -m qualisys.run_rtmpose --trial 张三_正手
python -m qualisys.run_qualisys --trial 张三_正手

# 2. 创建事件模板，人工填写每个动作的 start/contact/end 秒数
python -m qualisys.compare_angles --trial 张三_正手 --prepare-events

# 3. 对齐和比较
python -m qualisys.compare_angles --trial 张三_正手
```

也可用总入口选择阶段：

```powershell
python -m qualisys.run_validation --trial 张三_正手 --stage rtmpose --stage qualisys
python -m qualisys.run_validation --trial 张三_正手 --prepare-events
python -m qualisys.run_validation --trial 张三_正手 --stage compare
```

不传 `--trial` 会处理自动发现或清单中的全部试次。RTMPose 阶段依赖 CUDA、模型权重和真实视频；
Qualisys 转换与 CSV 对比阶段只依赖 NumPy/Pandas。

五类实验动作中的正手截击和反手截击会分别保存和统计，但 RTMPose 的 8 角计算都复用 Web
已有的 `volley` 姿态入口，因为三点角度公式完全相同。Web 页面仍保持原来的四类动作，不拆分。

## 计算边界

- 两套时间都以各自文件首帧归零，帧率不同不需要逐帧编号相同。
- 每个重复动作使用开始、击球、结束事件拟合 `t_qualisys = a × t_video + b`。
- 同一重复动作的 8 个角度共用同一时间映射，不能逐关节移动曲线来降低误差。
- Qualisys 100 Hz 曲线插值到视频帧时刻；这不会把视频虚构成 100 Hz 数据。
- 主比较为 ZY 平面无符号内角，不拿 Qualisys 三维角直接比较 RTMPose 二维角。
- 当前髋点默认使用同侧 ASIS/PSIS 中点，只是临时近似；正式金标准应优先导出 QTM
  Skeleton 的髋关节中心，并通过 `marker_map_file` 或后续骨架读取器替换。
- QTM 中 `TRAJECTORY_TYPES=Mixed` 可能包含补点；当前转换不再次插值，后续应补充质量标记统计。

完整实验组织、输出解释与后续信效度统计见
[`docs/qualisys_validation.md`](../docs/qualisys_validation.md)。
