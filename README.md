# sig_tennis_new

基于计算机视觉的网球动作分析系统。上传击球视频后，自动完成球拍轨迹追踪、人体姿态估计、动作切分与动力链分析，并通过 Web 界面展示结果。

## 功能概览

| 动作类型 | 说明 |
|---------|------|
| 正手击球 | 球拍追踪 + RTMPose 姿态 + 无监督切分 + 动力链图 / 速度重心图 |
| 反手击球 | 同上，针对反手动作优化 |
| 发球 | 轨迹分析 + 发球动力链图 |
| 截击 | 多片段切分 + 轨迹图 / 动力链图 |

支持选择持拍手（左手 / 右手），分析结果包含 CSV 数据、可视化图表与切分视频片段。

## 技术栈

- **后端**：Python、FastAPI、Uvicorn
- **前端**：Vue 3（静态页面，由 FastAPI 托管）
- **视觉算法**：Ultralytics YOLO、MMPose RTMPose、OpenCV
- **数据处理**：NumPy、Pandas、SciPy、Matplotlib、MoviePy

## 项目结构

```
sig_tennis_new/
├── main.py                 # 项目入口，启动服务并打开浏览器
├── requirements.txt        # Python 依赖
├── backend/                # FastAPI 后端
│   ├── app.py              # 应用配置与路由挂载
│   ├── routers/            # 各动作类型 API（forehand / backhand / serve / volley）
│   ├── services/           # 分析流水线编排
│   └── schemas.py          # 请求 / 响应模型
├── algorithm/              # 算法模块（按动作类型分目录）
│   ├── forehand/           # 正手：追踪、姿态、切分、作图
│   ├── backhand/           # 反手
│   ├── serve/              # 发球
│   └── volley/             # 截击
├── Vue/                    # 前端静态资源
│   ├── index.html          # 首页
│   ├── forehand.html       # 正手分析页
│   ├── backhand.html       # 反手分析页
│   ├── serve.html          # 发球分析页
│   └── volley.html         # 截击分析页
├── configs/                # RTMPose 模型配置
├── weights/                # 模型权重（需自行准备，见下文）
├── data/
│   ├── inputs/             # 上传视频存放目录
│   └── outputs/            # 分析产物输出目录
├── docs/                   # 补充文档
└── tests/                  # 单元测试
```

## 环境要求

- Python 3.9+
- 建议使用 NVIDIA GPU（姿态估计与 YOLO 推理依赖 PyTorch）
- Windows / Linux / macOS 均可运行

## 安装

```bash
# 克隆或进入项目目录
cd sig_tennis_new

# 创建虚拟环境（推荐）
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 安装依赖
pip install -r requirements.txt
```

### 模型权重

分析前需在项目根目录下准备 `weights/` 文件夹，并放入以下文件：

| 文件 | 用途 |
|------|------|
| `bestnew.pt` | 球拍检测与轨迹追踪（YOLO） |
| `yolov8m.pt` | 人体检测（YOLOv8） |
| `rtmpose_m_halpe26.pth` | 人体姿态估计（RTMPose，Halpe26 关键点） |

权重文件体积较大，未纳入版本库，请按实际部署环境自行获取或训练。

## 启动

```bash
python main.py
```

服务默认监听 `http://127.0.0.1:8000`，启动约 1.5 秒后会自动打开浏览器。

也可手动访问：

- 首页：`http://127.0.0.1:8000/`
- API 文档：`http://127.0.0.1:8000/docs`

## 使用流程

1. 在首页选择动作类型（正手 / 反手 / 发球 / 截击）
2. 上传击球视频，选择持拍手
3. 等待后端完成分析（追踪 → 姿态估计 → 切分 → 作图）
4. 在页面查看切分片段、动力链图、轨迹图等产物，支持下载

## 分析流水线

每种动作的分析均遵循统一的三阶段流水线：

```
上传视频
  → 球拍轨迹追踪（point_track）   → 拍头 CSV + 轨迹图
  → 人体姿态估计（rtmpose_csv）   → 关节角度 CSV
  → 无监督动作切分（segmentation）→ 片段 MP4 + 分析图表
```

产物按 `data/outputs/{run_id}/{视频名}/` 目录组织，`run_id` 为分析时的时间戳。

## API 接口

各动作模块路由前缀一致，以正手为例：

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/forehand/health` | 健康检查 |
| `POST` | `/api/forehand/analyze` | 上传视频并分析（表单字段：`file`、`handedness`） |
| `GET` | `/api/forehand/artifacts/{run_id}` | 查询某次运行的产物列表 |
| `GET` | `/api/forehand/file?path=...` | 下载指定产物文件 |

反手、发球、截击分别对应 `/api/backhand`、`/api/serve`、`/api/volley`，接口结构相同。

### 分析响应示例

```json
{
  "run_id": "20260517_153532",
  "video_name": "dark1",
  "message": "ok",
  "artifacts": [
    { "kind": "racket_csv", "filename": "...", "relative_path": "..." },
    { "kind": "body_csv", "filename": "...", "relative_path": "..." },
    { "kind": "kinetic_chart", "filename": "...", "relative_path": "..." },
    { "kind": "clip", "filename": "...", "relative_path": "..." }
  ],
  "intervals": [[1.2, 2.8]]
}
```

## 产物类型

| kind | 说明 |
|------|------|
| `racket_csv` | 球拍关键点轨迹数据 |
| `body_csv` | 人体关节角度数据 |
| `racket_chart` | 球拍轨迹可视化图 |
| `final_chart` | 综合分析图（正手 / 反手） |
| `clip` | 切分后的动作视频片段 |
| `kinetic_chart` | 动力链角度曲线图 |
| `speed_cog_chart` | 速度重心图 |
| `serve_trace_chart` | 发球轨迹图 |
| `serve_kinetic_chart` | 发球动力链图 |
| `volley_trace_chart` | 截击轨迹图 |
| `volley_kinetic_chart` | 截击动力链图 |

## 补充文档

- [正手动力链图说明](docs/forehand_kinetic_chart.md)

## 测试

```bash
python -m unittest tests.test_forehand_kinetic_chart -v
```

## 许可证

本项目仅供学习与研究使用。模型权重与训练数据请遵循各自来源的许可协议。
