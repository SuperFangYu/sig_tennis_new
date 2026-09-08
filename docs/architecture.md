# 项目结构与开发边界

## 当前架构

项目采用单仓库结构，前端、后端和算法一起维护：

```text
Vue -> FastAPI routers -> pipeline -> action algorithms -> common algorithms
                                      -> data/outputs

standalone scripts -> common manual pipeline -> data/manual

paired video + Qualisys 3D TSV -> qualisys validation -> qualisys/data/output
```

- `Vue/analysis.html` 是四类动作共用的分析页面，动作差异集中在
  `Vue/assets/app.js` 的 `ACTION_ANALYSIS_CONFIGS`。
- `backend/routers/action_router.py` 统一上传、历史产物查询和文件下载。
- `backend/services/pipeline.py` 统一四类动作的追踪、角度导出、融合和切分顺序。
- `algorithm/common/` 只保存动作无关的通用算法。
- `algorithm/{action}/` 保存动作入口和动作特有的切分规则。
- `standalone/` 调用公共算法生成 8 角 CSV 和人体/球拍叠加视频，不经过 Web；实验层把截击
  拆为正手截击和反手截击，因此有五个入口，但二者都复用底层 `volley` 算法。
- `qualisys/` 是独立实验验证层，只复用 RTMPose 角度定义，不接入后端 API，也不读取测力台。

## 稳定接口

重构时应保持下列接口稳定：

- `/api/{action}/health`
- `/api/{action}/analyze`
- `/api/{action}/artifacts/{run_id}`
- `/api/{action}/file`
- `data/outputs/{run_id}/{video_name}/` 输出层级
- API artifact 的 `kind`、`filename`、`relative_path`
- 四个 `algorithm/{action}/angles_csv.py` 的 `run_angles_csv` 函数

## 离线视频可视化边界

`algorithm/common/analysis_overlay.py` 负责把内存中的 Halpe26 人体点和临时球拍五点共同
画回视频；`algorithm/common/manual_analysis.py` 负责三阶段编排。可视化稳定规则只作用于
叠加视频，不改写 Web 球拍数据。正式离线输出只保留 8 角 CSV 和叠加视频。

## 实验验证边界

五个 `qualisys/actions/*_validation.py` 分别由使用者填写视频、已生成的 RTMPose 8 角 CSV
和 QTM 3D TSV 路径；公共核心使用视频/QTM 拍头速度峰自动匹配五次动作并建立统一时间映射。
测力台文件、肩髋分离角和 Web 产物都不进入当前 8 角效度比较；结果固定写入
`qualisys/data/output/动作/样本/运行时间/`，每次运行新建目录。详细约束见
[`qualisys_validation.md`](qualisys_validation.md)。
