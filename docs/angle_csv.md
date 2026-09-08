# 独立角度 CSV 计算标准

四个 `algorithm/{action}/angles_csv.py` 使用同一套 Halpe-26 二维图像平面定义，保证
独立脚本和 Web 流水线输出一致。

## 核心关节角

所有角度单位均为度，使用三点夹角，第二个点是角度顶点：

| CSV列 | 三点定义 |
|---|---|
| `left_shoulder_angle` | 左肘—左肩—左髋 |
| `right_shoulder_angle` | 右肘—右肩—右髋 |
| `left_elbow_angle` | 左肩—左肘—左腕 |
| `right_elbow_angle` | 右肩—右肘—右腕 |
| `left_hip_angle` | 左肩—左髋—左膝 |
| `right_hip_angle` | 右肩—右髋—右膝 |
| `left_knee_angle` | 左髋—左膝—左踝 |
| `right_knee_angle` | 右髋—右膝—右踝 |
| `shoulder_hip_angle` | 左右肩连线与左右髋连线的无符号夹角 |

这些是图像 `x/y` 平面上的二维角度，不是三维解剖学关节角。CSV同时保留12个关节的
`x`、`y`、`conf`，以及肩髋中心、身体中心、肩髋倾斜和躯干角等辅助字段。

## 动作差异

- 正手、发球、截击使用统一基础字段。
- 反手额外保留 `shoulder_turn_x_diff`，供现有反手切分规则使用。
- 动作类型只影响动作特有字段和输出文件名，不改变上述核心关节角定义。

低于关键点置信度阈值的点先记为缺失值。当前导出流程为保持Web历史行为，会对坐标和
角度列执行线性插值并在首尾前后填充；置信度列不插值。

## 离线与验证 CSV 的区别

- Web 与 `algorithm/{action}/angles_csv.py` 默认保留关键点坐标、置信度和更多派生字段。
- `standalone/*_analysis.py` 的正式 CSV 只保留 `frame`、`time` 与左右肩/肘/髋/膝 8 角。
- Qualisys 验证直接读取已经生成的 RTMPose 8 角 CSV，不会再次运行人体姿态模型。
- `qualisys/core/qtm.py` 将 3D marker 投影到 ZY 后用相同定义计算 8 角，不计算肩髋分离角。

由于现有 RTMPose 核心会对缺失角做插值，正式效度研究还需要增加原始有效帧率和插值比例
输出；现阶段结果应作为流程试跑和探索性统计，不应省略这一局限。
