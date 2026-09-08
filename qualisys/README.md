# Qualisys–RTMPose 五动作角度验证

## 这一版做什么

本模块只负责验证，不重新生成 RTMPose 人体角度，也不读取测力台。你为一个动作提供：

1. 与 QTM 同次采集的原始侧面视频；
2. 已经运行好的 RTMPose 8 角 CSV；
3. Qualisys 导出的 3D marker TSV。

程序用视频 YOLO 拍头速度和 QTM `Racket_top` 的 ZY 平面速度自动寻找五次动作，用五个
速度峰拟合统一时间关系，再比较左右肩、肘、髋、膝 8 个角度。没有球、看不到触球都可以；
这里的自动锚点是“动作中拍头速度峰”，不是触球帧。

## 目录结构

```text
qualisys/
├── actions/
│   ├── forehand_validation.py          # 正手手动入口
│   ├── backhand_validation.py          # 反手手动入口
│   ├── forehand_volley_validation.py   # 正手截击手动入口
│   ├── backhand_volley_validation.py   # 反手截击手动入口
│   └── serve_validation.py             # 发球手动入口
├── core/
│   ├── action_profiles.py              # 五动作的峰值与窗口参数
│   ├── qtm.py                          # QTM TSV、ZY 八角、Racket_top 速度
│   ├── alignment.py                    # 自动识别五次动作并拟合时间映射
│   ├── metrics.py                      # 插值与一致性指标
│   ├── plotting.py                     # 对齐质控图
│   ├── io.py                           # 输入检查与新建输出目录
│   └── runner.py                       # 单次任务编排
└── data/
    ├── input/video/
    ├── input/rtmpose/
    ├── input/qtm/
    └── output/
```

五个动作只有时间节奏参数不同，八角公式和统计核心只有一份，因此后续修改不会出现五份代码
计算标准不一致。Web 仍然只展示正手、反手、发球、截击四类，不受这里的五动作入口影响。

## 运行前准备

建议把文件按下面方式放置，但程序并不依赖同名扫描：

```text
qualisys/data/input/video/fy_zs_1.avi
qualisys/data/input/rtmpose/fy_zs_1.csv
qualisys/data/input/qtm/fy_zs_1.tsv
```

RTMPose CSV 必须只需具备这些字段（多余字段会被忽略）：

```text
frame,time,
left_shoulder_angle,right_shoulder_angle,
left_elbow_angle,right_elbow_angle,
left_hip_angle,right_hip_angle,
left_knee_angle,right_knee_angle
```

QTM TSV 必须是 `DATA_INCLUDED=3D` 的 marker 导出，并至少包含当前人体 marker 映射以及
`Racket_top X/Y/Z`。不用放 `_a_1.tsv`、`_a_2.tsv` 等测力台文件。球拍 YOLO 权重默认读取
项目根目录 `weights/bestnew.pt`。

## 在 PyCharm 中怎么用

以正手为例，打开 `qualisys/actions/forehand_validation.py`，只修改顶部三条路径：

```python
VIDEO_PATH = REPO_ROOT / "qualisys/data/input/video/fy_zs_1.avi"
RTMPOSE_CSV_PATH = REPO_ROOT / "qualisys/data/input/rtmpose/fy_zs_1.csv"
QUALISYS_TSV_PATH = REPO_ROOT / "qualisys/data/input/qtm/fy_zs_1.tsv"
```

保持：

```python
EXPECTED_REPETITIONS = 5
```

然后右键运行该文件即可。反手、正手截击、反手截击、发球分别打开对应的另外四个入口，
操作完全一样。路径可以写成 `Path(r"D:\...")` 绝对路径；三个文件不要求同名，但必须来自
同一次采集、动作出现顺序一致。

你不需要填写五次动作的起止时间。程序会：

1. 从视频提取拍头速度；
2. 从 QTM `Racket_top` 的 Z/Y 坐标计算拍头速度；
3. 两边各找候选速度峰，并从中匹配五个顺序一致的动作锚点；
4. 拟合一次 `t_qualisys = slope × t_video + intercept`；
5. 以视频速度峰周围的速度回落点自动划分五个动作窗口；
6. 把 Qualisys 八角插值到视频帧时刻并输出指标。

## 输出在哪里

不需要也不能手填 `OUTPUT_DIR`。每次运行自动在固定根目录下建立新文件夹：

```text
D:\FY\sig_tennis_new\qualisys\data\output\
└── forehand/
    └── fy_zs_1/
        ├── 20260908_153012_123/
        └── 20260908_154455_807/
```

即使同一个文件反复运行，也不会覆盖上一次结果。优先看：

- `alignment_qc.png`：红色叉号应在两边各显示五个正确动作峰；
- `alignment_qc.csv`：`status=pass`、时间斜率、锚点 RMSE、球拍有效率；
- `repetition_windows.csv`：五次动作的自动起止时间是否合理；
- `angle_metrics.csv`：最终每次动作和 `all` 的角度误差与一致性结果。

完整文件含义见 `data/output/README.md`。

## 自动对齐失败怎么办

程序不会为了凑够五次动作而输出看似漂亮的结果。以下情况会停止，并在本次新目录的
`run_status.json` 写明失败原因：

- 视频拍头原始有效率低于 20%；
- QTM `Racket_top` 有效率低于 80%；
- 任一侧找不到五个可靠速度峰；
- 五个峰拟合出的时间比例不在 0.85–1.15；
- 锚点 RMSE 超过 0.20 秒或单个残差超过 0.35 秒；
- 映射后的动作窗口超出 QTM 数据范围。

失败时先检查三个输入是否为同一次采集，再看球拍识别和 `Racket_top` 缺失情况。动作确实不是
五次时才修改 `EXPECTED_REPETITIONS`；正式实验若预设为五次，不建议为了让某个坏试次通过而
临时改阈值。第一版不使用自由 DTW，也不允许每个关节各自左右平移。

## 当前角度定义与限制

- RTMPose 使用图像 x/y；Qualisys 使用全局 Z/Y，两边都是 0–180° 无符号三点内角。
- 肩：肘–肩–髋；肘：肩–肘–腕；髋：肩–髋–膝；膝：髋–膝–踝。
- QTM 肘、腕、膝、踝取内外侧 marker 中点。
- 当前髋取同侧 ASIS/PSIS 中点，只是临时近似，不等同于 Skeleton 解剖学髋中心。
- 自动对齐解决的是“同一物理动作对应哪个时刻”，不是提高或修饰角度准确率。
- `all` 只是逐帧描述统计；正式论文推断要处理受试者/试次内的重复测量和帧间自相关。

研究设计与各指标解释见 [`docs/qualisys_validation.md`](../docs/qualisys_validation.md)。
