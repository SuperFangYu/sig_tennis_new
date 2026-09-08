# RTMPose–Qualisys 八角验证：使用说明

本目录比较同一次采集的侧面视频与 Qualisys 3D marker 数据，只计算左右肩、肘、髋、膝
8 个无符号二维角度，不使用测力台，也不需要 `manifest.csv`。

## 你实际需要认识的文件

| 文件 | 用途 |
|---|---|
| `run_validation.py` | 你在 PyCharm 中直接运行的总入口，只需修改顶部 3 个参数。 |
| `trials.py` | 自动扫描并配对 `video/` 与 `qtm/` 中的同名文件。 |
| `run_rtmpose.py` | 阶段 1A：从视频生成 RTMPose 8 角 CSV。 |
| `run_qualisys.py` | 阶段 1B：从 3D TSV 的 ZY 平面生成 Qualisys 8 角 CSV。 |
| `compare_angles.py` | 阶段 2/3：生成事件表，并按事件对齐后计算误差。 |
| `config.py` | 保存五动作名称、事件定义、8 个角度列及 Qualisys 点位映射。 |

通常只运行 `run_validation.py`，其他文件不用单独打开。

## 第一步：给输入文件命名

视频和 TSV 主文件名必须完全相同，格式是：

```text
英文名_英文动作_编号
```

你的动作缩写已经支持：

| 缩写 | 动作 | 示例配对 |
|---|---|---|
| `zs` | 正手 | `video/fy_zs_1.avi` ↔ `qtm/fy_zs_1.tsv` |
| `fs` | 反手 | `video/fy_fs_1.avi` ↔ `qtm/fy_fs_1.tsv` |
| `jjzs` | 正手截击 | `video/fy_jjzs_1.avi` ↔ `qtm/fy_jjzs_1.tsv` |
| `jjfs` | 反手截击 | `video/fy_jjfs_1.avi` ↔ `qtm/fy_jjfs_1.tsv` |
| `fq` | 发球 | `video/fy_fq_1.avi` ↔ `qtm/fy_fq_1.tsv` |

也支持完整英文动作名，例如 `tom_forehand_1`、`tom_backhand_volley_2`。末尾编号必须是
1、2、3……这样的正整数。

文件放置位置：

```text
qualisys/data/input/
├── video/                 # 只放视频
│   ├── fy_zs_1.avi
│   └── fy_fq_1.avi
└── qtm/                   # 只放 3D TSV，不放测力台文件
    ├── fy_zs_1.tsv
    └── fy_fq_1.tsv
```

左手球员不需要额外设置：本实验输出左右两侧全部 8 角，水平翻转侧面机位不会改变三点夹角
大小。比较时仍然保持 RTMPose 左角对 Qualisys 左角、右角对右角。

## 第二步：在 PyCharm 运行角度提取

打开 `qualisys/run_validation.py`，在顶部找到：

```python
RUN_MODE = "extract"
TRIAL_ID = ""  # 留空处理全部配对；也可填写 "fy_zs_1"
DEVICE = "cuda:0"
```

设置 `RUN_MODE = "extract"` 后右键运行。每个配对会生成：

```text
qualisys/data/intermediate/fy_zs_1/
├── rtmpose_8_angles.csv
└── qualisys_8_angles.csv
```

这一步不做时间对齐，只分别计算两套角度。

## 第三步：生成并填写动作时间表

把 `RUN_MODE` 改成：

```python
RUN_MODE = "prepare_events"
```

再次运行，程序会在每个试次的 `intermediate` 目录生成：

- `video_events.csv`：填写视频中的事件秒数。
- `qualisys_events.csv`：填写 QTM 中相同事件的相对秒数。

每张表都有四列：

| 列 | 怎么填 |
|---|---|
| `repetition` | 文件中第几次完整动作，从 1 开始。 |
| `event` | 固定为 `start`、`contact`、`end`，不要修改。 |
| `time` | 你需要填写的秒数。 |
| `definition` | 程序根据动作写入的判断标准，不要修改。 |

如果一个文件里有 5 次击球，就保留 15 行：每个 `repetition` 都要有三行事件。两张表的
`repetition=1` 必须是同一次击球，不能只按出现顺序猜测。

视频时间以第一帧为 0 秒。Qualisys 时间使用 `qualisys_8_angles.csv` 的 `time` 列，该列已经
自动执行“原始 QTM Time − 第一行 Time”。

## 五种动作的时间划分

角度公式不需要分五套，五种动作始终使用相同 8 个三点夹角。需要分别处理的是动作标签和
`start/contact/end` 的物理定义。相似动作共用规则，因此实际是三组：

| 动作组 | `start` | `contact` | `end` |
|---|---|---|---|
| 正手 `zs`、反手 `fs` | 后摆结束，持拍手或拍头开始持续向击球方向加速 | 球拍触球；看不清时取拍头最接近来球的位置 | 随挥结束后，持拍手或拍头速度第一次明显降到低谷 |
| 正手截击 `jjzs`、反手截击 `jjfs` | 准备姿势后，球拍开始持续向来球方向移动 | 球拍触球；看不清时取拍面最接近来球的位置 | 短促挡击结束后，球拍速度第一次明显降到低谷 |
| 发球 `fq` | 持拍手或拍头离开稳定准备位置，发球动作正式启动 | 球拍触球；看不清时取拍头进入最高击球区域的时刻 | 落地随挥完成后，持拍手或拍头速度第一次明显降到低谷 |

正手和反手可以用同一判断逻辑，正手截击和反手截击也可以用同一逻辑；但五种动作仍分别
保存和统计，不能把正手截击与普通正手合并。

## 第四步：运行对齐和指标计算

两张事件表填写完成后，把顶部参数改为：

```python
RUN_MODE = "compare"
```

再次运行，结果保存在：

```text
qualisys/data/output/fy_zs_1/
├── aligned_angles.csv     # 视频每帧对应的两套角度和逐帧误差
├── angle_metrics.csv      # MAE、RMSE、bias、LoA、相关、CCC 等
├── event_metrics.csv      # start/contact/end 三个事件处的角度误差
└── alignment_qc.csv       # 时间映射斜率、截距、锚点残差和有效帧数
```

程序对每次动作拟合 `t_qualisys = a × t_video + b`，Qualisys 100 Hz 曲线被插值到视频帧
时刻。同一次动作的 8 个角共用一个映射，不能逐关节移动曲线来人为降低误差。

## 可选命令行方式

不使用 PyCharm 时，也可以运行：

```powershell
python -m qualisys.run_validation --mode extract --trial fy_zs_1
python -m qualisys.run_validation --mode prepare_events --trial fy_zs_1
python -m qualisys.run_validation --mode compare --trial fy_zs_1
```

不传 `--trial` 会处理全部同名配对。

## 当前计算注意事项

- Qualisys 只使用 Z/Y 两轴，计算与 RTMPose 相同的 0–180° 无符号内角。
- 正手截击和反手截击分别统计，但 RTMPose 都复用已有 `volley` 姿态入口，因为角度公式相同。
- 当前髋点使用同侧 ASIS/PSIS 中点作为临时近似；正式实验最好换成 QTM Skeleton 髋关节中心。
- QTM 的 `TRAJECTORY_TYPES=Mixed` 可能已经包含补点；当前 Qualisys 转换不会再次插值。
- Web 页面仍保持正手、反手、发球、截击四类，本模块的五动作拆分不会影响前端。

信效度指标和实验质量控制的详细说明见
[`docs/qualisys_validation.md`](../docs/qualisys_validation.md)。
