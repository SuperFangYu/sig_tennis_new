# 五动作离线实验入口

这套入口不需要启动 FastAPI 或前端，适用于 RTMPose 与 Qualisys 的对比实验。

## 在 PyCharm 中运行

根据实验动作打开以下一个文件。这里把截击拆成正手截击与反手截击，但不修改 Web 的四动作页面：

- `standalone/forehand_analysis.py`
- `standalone/backhand_analysis.py`
- `standalone/forehand_volley_analysis.py`
- `standalone/backhand_volley_analysis.py`
- `standalone/serve_analysis.py`

修改文件顶部的参数：

```python
INPUT_VIDEO = REPO_ROOT / "data" / "manual" / "input" / "serve.mp4"
OUTPUT_ROOT = REPO_ROOT / "data" / "manual" / "output"
DEVICE = "cuda:0"
HANDEDNESS = "right"
```

`INPUT_VIDEO` 也可以改成 Windows 绝对路径，例如：

```python
INPUT_VIDEO = Path(r"D:\experiment\serve_01.mp4")
```

然后右键运行该 Python 文件。默认使用项目已有的三个权重：

- `weights/yolov8m.pt`：人体检测
- `weights/rtmpose_m_halpe26.pth`：人体关键点
- `weights/bestnew.pt`：球拍五点

正手截击和反手截击分别输出到 `forehand_volley/` 与 `backhand_volley/`，底层均复用项目已有
`volley` 姿态与球拍处理。两者的 8 角公式相同，拆分的目的是让后续实验分组和统计不混在一起。

## 每次运行的结果

以发球视频 `serve_01.mp4` 为例，结果保存在：

```text
data/manual/output/serve/serve_01/
├── serve_01_body_angles.csv
└── serve_01_pose_racket_overlay.mp4
```

- `body_angles.csv`：除 `frame`、`time` 外，仅保存左右肩/肘/髋/膝共 8 个二维夹角。
- `pose_racket_overlay.mp4`：基于 Halpe26 的人体骨架、黄色人体点、洋红色球拍连线和绿色球拍点。头部不绘制鼻、双眼和头顶突出点，只保留左右耳，并在两耳中间同高度生成一个简洁的面部中心点。

球拍五点数据只作为视频绘制的临时缓存，绘制完成后自动清理，不会在输出目录中增加额外 CSV。项目原有 Web 球拍 CSV 与分析流水线不受影响。
如果同一视频目录中残留上一版离线入口生成的 `*_racket_keypoints.csv` 或
`*_combined.csv`，新脚本成功运行后会清理这两个已废弃文件。

拍面按 `t-l-b-r-t` 闭合，拍柄按 `b-h` 连接，拍面和拍柄颜色一致。球拍可视化会用宽松的滚动尺寸与时间连续性检查去除明显跳点，并只补最多 2 帧的小缺口。侧视拍面允许被压缩得很窄，不使用拍面面积、凸性或最小宽度作为限制。上述修正只影响绘制，不改写球拍 CSV。输出视频由 OpenCV 编码，保留原始尺寸与帧率，当前不复制原视频音轨。

## 二维角度标准

所有角度都在视频图像的 x-y 平面计算，三点夹角的中间点为角度顶点：

- 肩角：肘—肩—髋
- 肘角：肩—肘—腕
- 髋角：肩—髋—膝
- 膝角：髋—膝—踝

这些角度可以与 Qualisys 选定的 ZY 平面投影角比较，但需要后续完成时间对齐、左右侧和角度定义核对。

正式成对验证请不要直接把本目录任意视频与 Qualisys TSV 拼接。应将同次采集文件按
`英文名_英文动作_编号` 同名放入 `qualisys/data/input/video/` 和 `qtm/`，再按
[`qualisys/README.md`](../qualisys/README.md) 的三步流程处理；该流程不需要球拍叠加视频，
也不读取测力台数据。
