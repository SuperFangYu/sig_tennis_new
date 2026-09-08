# 成对试次输入

最简单的方式是不创建清单，直接将文件分别放入：

```text
video/人名_动作.mp4
qtm/人名_动作.tsv
```

程序按主文件名完全相同自动配对。动作只能是正手、反手、正手截击、反手截击、发球。例如
`video/张三_反手截击.mp4` 配对 `qtm/张三_反手截击.tsv`。自动模式默认右手持拍。

`manifest.example.csv` 是可选的“试次索引模板”，不是必须处理的数据。如果受试者为左手、
文件无法同名、需要自定义 marker 映射或记录备注，再复制为 `manifest.csv` 并填写。

清单字段：

| 字段 | 含义 |
|---|---|
| `trial_id` | 唯一试次编号，也是中间结果和输出目录名。 |
| `participant_id` | 受试者匿名编号。 |
| `action` | 五类动作代码之一，截击分为 `forehand_volley`、`backhand_volley`。 |
| `handedness` | `right` 或 `left`。 |
| `view_plane` | 当前固定填写 `ZY`。 |
| `video_file` | 视频路径，相对本目录或使用绝对路径。 |
| `qualisys_3d_file` | 3D marker TSV 路径。 |
| `marker_map_file` | 可选 JSON；点名与默认映射不同时填写。 |
| `notes` | 自由备注，如机位、异常帧或试次剔除原因。 |

`marker_map.example.json` 展示当前默认点位映射。若复制后修改，请在 manifest 中填写该文件；
真实清单、视频和 TSV 不会提交到 Git。
