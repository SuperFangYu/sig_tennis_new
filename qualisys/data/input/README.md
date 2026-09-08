# 成对试次输入

复制 `manifest.example.csv` 为 `manifest.csv`，再为每个试次建立同名子目录。一个试次只能
登记同一次采集的视频和 Qualisys `DATA_INCLUDED=3D` TSV；不要放测力台 analog TSV。

清单字段：

| 字段 | 含义 |
|---|---|
| `trial_id` | 唯一试次编号，也是中间结果和输出目录名。 |
| `participant_id` | 受试者匿名编号。 |
| `action` | `forehand`、`backhand`、`serve` 或 `volley`。 |
| `handedness` | `right` 或 `left`。 |
| `view_plane` | 当前固定填写 `ZY`。 |
| `video_file` | 视频路径，相对本目录或使用绝对路径。 |
| `qualisys_3d_file` | 3D marker TSV 路径。 |
| `marker_map_file` | 可选 JSON；点名与默认映射不同时填写。 |
| `notes` | 自由备注，如机位、异常帧或试次剔除原因。 |

`marker_map.example.json` 展示当前默认点位映射。若复制后修改，请在 manifest 中填写该文件；
真实清单和试次文件不会提交到 Git。
