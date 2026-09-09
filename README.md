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

```text
sig_tennis_new/
├── main.py                         # 启动 FastAPI 服务并按配置打开浏览器。
├── requirements.txt                # 列出项目 Python 直接依赖。
├── backend/                        # Web 后端与分析任务编排。
├── algorithm/                      # 人体、球拍、切分和运动学算法。
├── standalone/                     # 不启动 Web 的五动作本地视频分析入口。
├── qualisys/                       # RTMPose–Qualisys 成对验证流水线，不使用测力台。
├── Vue/                            # 由 FastAPI 托管的 Vue 静态前端。
├── configs/                        # RTMPose 模型配置。
├── weights/                        # 本地模型权重，不提交 Git。
├── data/                           # Web 与手动分析的本地输入输出。
├── docs/                           # 计算标准、架构、环境和实验说明。
└── tests/                          # 不依赖真实视频的自动化测试。
```

### 后端文件

| 文件 | 一句话说明 |
|---|---|
| `backend/app.py` | 创建 FastAPI 应用、挂载四动作路由并托管前端静态文件。 |
| `backend/schemas.py` | 定义分析响应和产物信息的数据模型。 |
| `backend/routers/action_router.py` | 统一实现上传、分析、历史产物查询和安全下载。 |
| `backend/routers/forehand.py` | 注册正手 API。 |
| `backend/routers/backhand.py` | 注册反手 API。 |
| `backend/routers/serve.py` | 注册发球 API。 |
| `backend/routers/volley.py` | 注册截击 API。 |
| `backend/services/pipeline.py` | 按动作编排球拍追踪、姿态估计、融合、切分和作图。 |
| `backend/**/__init__.py` | 声明 Python 包，不承载业务逻辑。 |

### 公共算法文件

| 文件 | 一句话说明 |
|---|---|
| `algorithm/common/action_angles.py` | 给四种动作提供统一 RTMPose CSV 调用接口。 |
| `algorithm/common/pose_csv_core.py` | 逐帧检测人体、估计 Halpe26 并导出运动学特征。 |
| `algorithm/common/pose_features.py` | 定义 Halpe26 点位、骨架连接和二维关节角公式。 |
| `algorithm/common/point_track_core.py` | 运行 YOLO 球拍五点追踪并保存轨迹数据。 |
| `algorithm/common/racket_features.py` | 计算球拍中心、拍轴、速度等派生特征。 |
| `algorithm/common/kinematic_fusion.py` | 按帧融合人体与球拍运动学数据。 |
| `algorithm/common/segmentation_helpers.py` | 提供动作切分、平滑、评分和统计的共享工具。 |
| `algorithm/common/kinematic_plots.py` | 绘制共用动力链和角度分析图。 |
| `algorithm/common/analysis_overlay.py` | 把简化 Halpe26 骨架与稳定后的球拍五点画回视频。 |
| `algorithm/common/manual_analysis.py` | 编排手动视频的八角 CSV 与人体/球拍叠加视频。 |
| `algorithm/common/thresholds.py` | 集中保存人体关键点置信度等阈值。 |
| `algorithm/common/inference/detector.py` | 封装 Ultralytics 人体检测器。 |
| `algorithm/common/inference/pose_estimator.py` | 封装 MMPose RTMPose Halpe26 推理器。 |
| `algorithm/**/__init__.py` | 声明算法 Python 包。 |

### 四动作算法文件

每个 `algorithm/{forehand,backhand,serve,volley}/angles_csv.py` 都是对应动作的人体角度入口，
每个 `point_track.py` 都是对应动作的球拍追踪兼容入口，每个 `segmentation.py` 都保存对应动作
的切分规则；`forehand/plot_kinetic_chain.py` 与 `backhand/plot_kinetic_chain.py` 分别生成正手和
反手专项动力链图。Web 和底层算法仍保持四类；五个 `standalone/*_analysis.py` 将截击拆成
正手截击和反手截击，用于在 PyCharm 中独立运行及分组实验。

| 五动作离线入口 | 一句话说明 |
|---|---|
| `standalone/forehand_analysis.py` | 手动运行正手视频分析。 |
| `standalone/backhand_analysis.py` | 手动运行反手视频分析。 |
| `standalone/forehand_volley_analysis.py` | 手动运行正手截击视频分析并单独保存结果。 |
| `standalone/backhand_volley_analysis.py` | 手动运行反手截击视频分析并单独保存结果。 |
| `standalone/serve_analysis.py` | 手动运行发球视频分析。 |

### Qualisys 验证文件

| 文件 | 一句话说明 |
|---|---|
| `qualisys/actions/forehand_validation.py` | 手填输入路径和视频有效范围后运行正手五次动作验证。 |
| `qualisys/actions/backhand_validation.py` | 手填输入路径和视频有效范围后运行反手五次动作验证。 |
| `qualisys/actions/forehand_volley_validation.py` | 手填输入路径和视频有效范围后运行正手截击五次动作验证。 |
| `qualisys/actions/backhand_volley_validation.py` | 手填输入路径和视频有效范围后运行反手截击五次动作验证。 |
| `qualisys/actions/serve_validation.py` | 手填输入路径和视频有效范围后运行发球五次动作验证。 |
| `qualisys/config.py` | 定义固定输入/输出目录、8 角列和 Qualisys marker 映射。 |
| `qualisys/core/qtm.py` | 把 3D marker 投影到 ZY，生成八角与 QTM 拍头速度。 |
| `qualisys/core/alignment.py` | 用视频/QTM 拍头速度峰自动识别并匹配五次动作。 |
| `qualisys/core/metrics.py` | 对齐两套八角并计算 MAE、RMSE、bias、LoA、相关与 CCC。 |
| `qualisys/core/runner.py` | 编排单个动作验证并为每次运行建立独立输出目录。 |

### 前端、配置与测试文件

| 文件 | 一句话说明 |
|---|---|
| `Vue/index.html` | 提供动作类型选择首页。 |
| `Vue/analysis.html` | 提供四动作共用的上传、分析和结果页面。 |
| `Vue/forehand.html`、`backhand.html`、`serve.html`、`volley.html` | 保留旧动作地址并跳转到共用分析页。 |
| `Vue/assets/app.js` | 保存前端动作配置、API 请求和结果渲染逻辑。 |
| `Vue/assets/app.css` | 保存全站样式。 |
| `Vue/assets/vue.global.prod.js` | 提供本地 Vue 3 运行时。 |
| `configs/rtmpose_m_halpe26.py` | 配置 RTMPose-M Halpe26 网络与数据集元信息。 |
| `tests/test_refactor_contracts.py` | 检查统一入口、8 角 CSV、叠加骨架和球拍稳定契约。 |
| `tests/test_forehand_kinetic_chart.py` | 检查正手动力链图输出。 |
| `tests/test_kinematic_summary.py` | 检查分段运动学汇总统计。 |
| `tests/test_relaxed_segmentation.py` | 检查宽松动作切分对短序列与缺失数据的行为。 |
| `tests/test_qualisys_framework.py` | 检查 ZY 八角、五峰自动对齐、统一时间映射和新建输出目录。 |

### 文档与数据目录

| 文件或目录 | 一句话说明 |
|---|---|
| `docs/architecture.md` | 说明单仓库依赖方向、稳定接口与各流水线边界。 |
| `docs/angle_csv.md` | 说明 RTMPose 关节角、字段及插值行为。 |
| `docs/manual_analysis.md` | 说明五动作 PyCharm 离线入口的参数、产物与可视化。 |
| `docs/qualisys_validation.md` | 说明成对输入、ZY 角度、逐动作对齐、指标和实验局限。 |
| `docs/forehand_kinetic_chart.md` | 说明正手动力链图各阶段和曲线含义。 |
| `docs/environment.md` | 记录 5090 上 `fytennis` 环境和关键依赖版本。 |
| `docs/research/` | 保存早期特征研究材料，不作为当前运行入口。 |
| `data/inputs/` | 保存 Web 上传视频。 |
| `data/outputs/` | 保存按运行时间与视频名组织的 Web 分析产物。 |
| `data/manual/input/` | 保存五动作手动分析视频。 |
| `data/manual/output/` | 保存手动分析的 8 角 CSV 与叠加视频。 |
| `qualisys/data/input/video/` | 保存与 QTM 同次采集的验证视频。 |
| `qualisys/data/input/rtmpose/` | 保存已单独运行完成的 RTMPose 8 角 CSV。 |
| `qualisys/data/input/qtm/` | 保存 Qualisys 3D marker TSV，不放测力台文件。 |
| `qualisys/data/output/` | 按动作/样本/运行时间逐次新建对齐、指标与质控结果。 |
| `.gitignore` | 排除权重、视频、实验数据、运行结果和本机配置。 |
| `AGENTS.md` | 规定团队沟通、Git 分支和 5090 验证流程。 |

更细的依赖方向和不得破坏的稳定接口见[项目结构与开发边界](docs/architecture.md)。

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
conda activate fytennis
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
  → 人体姿态估计（angles_csv）    → 关键点与关节角度 CSV
  → 无监督动作切分（segmentation）→ 片段 MP4 + 分析图表
```

产物按 `data/outputs/{run_id}/{视频名}/` 目录组织，`run_id` 为分析时的时间戳。

## 独立导出人体关键点与角度 CSV

四类动作各有一个可独立调用的入口。它们只运行人体检测和 RTMPose，输出与 Web
流水线完全相同的关键点、置信度和 2D 关节角 CSV，不进行球拍追踪、动作切分或画图。

```bash
python -m algorithm.forehand.angles_csv --video input.mp4 --output-dir result_analysis
python -m algorithm.backhand.angles_csv --video input.mp4 --output-dir result_analysis
python -m algorithm.serve.angles_csv --video input.mp4 --output-dir result_analysis
python -m algorithm.volley.angles_csv --video input.mp4 --output-dir result_analysis
```

可通过 `--device`、`--yolo-conf`、`--kpt-thr`、`--config`、`--checkpoint` 和
`--yolo-model` 覆盖默认推理参数。Python代码也可以直接导入各模块的
`run_angles_csv(video_path, output_dir, ...)`。

## 不启动 Web 的完整离线分析

`standalone/` 下提供正手、反手、正手截击、反手截击、发球五个可在 PyCharm 中直接运行的脚本。
只需修改脚本顶部的输入视频路径，即可一次生成：仅含左右肩/肘/髋/膝 8 个二维角度
的人体 CSV，以及绘制基于 Halpe26 的简化面部骨架和球拍轮廓 MP4。球拍五点只作为视频绘制
的临时数据，不额外保存 CSV；原有 Web 流水线不受影响。

输入默认位于 `data/manual/input/`，结果位于
`data/manual/output/{动作}/{视频名}/`。详细说明见
[五动作离线实验入口](docs/manual_analysis.md)。

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
- [项目结构与开发边界](docs/architecture.md)
- [独立角度 CSV 计算标准](docs/angle_csv.md)
- [五动作离线实验入口](docs/manual_analysis.md)
- [RTMPose–Qualisys 信效度验证方案](docs/qualisys_validation.md)
- [Qualisys 验证模块操作说明](qualisys/README.md)
- [fytennis 环境记录](docs/environment.md)
- 历史研究材料位于 `docs/research/`

## 测试

```bash
$env:MPLBACKEND = "Agg"
python -m unittest discover -s tests -v
```

## 许可证

本项目仅供学习与研究使用。模型权重与训练数据请遵循各自来源的许可协议。
