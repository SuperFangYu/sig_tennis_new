# 项目结构与开发边界

## 当前架构

项目采用单仓库结构，前端、后端和算法一起维护：

```text
Vue -> FastAPI routers -> pipeline -> action algorithms -> common algorithms
                                      -> data/outputs
```

- `Vue/analysis.html` 是四类动作共用的分析页面，动作差异集中在
  `Vue/assets/app.js` 的 `ACTION_ANALYSIS_CONFIGS`。
- `backend/routers/action_router.py` 统一上传、历史产物查询和文件下载。
- `backend/services/pipeline.py` 统一四类动作的追踪、角度导出、融合和切分顺序。
- `algorithm/common/` 只保存动作无关的通用算法。
- `algorithm/{action}/` 保存动作入口和动作特有的切分规则。

## 稳定接口

重构时应保持下列接口稳定：

- `/api/{action}/health`
- `/api/{action}/analyze`
- `/api/{action}/artifacts/{run_id}`
- `/api/{action}/file`
- `data/outputs/{run_id}/{video_name}/` 输出层级
- API artifact 的 `kind`、`filename`、`relative_path`
- 四个 `algorithm/{action}/angles_csv.py` 的 `run_angles_csv` 函数

## 后续视频可视化边界

人体与球拍关键点共同画回视频时，不应修改 `angles_csv.py`。建议新增：

```text
algorithm/common/visualization/
├── skeleton_style.py
├── body_overlay.py
├── racket_overlay.py
└── combined_video.py
```

人体骨架与球拍五点应使用独立的颜色、线宽和连线拓扑；可视化读取已有CSV，避免再次
运行模型，也避免影响现有Web分析结果。
