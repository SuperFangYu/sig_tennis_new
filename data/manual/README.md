# 离线实验数据目录

- `input/`：放入要手动分析的视频。五个独立脚本默认寻找
  `forehand.mp4`、`backhand.mp4`、`forehand_volley.mp4`、`backhand_volley.mp4`、`serve.mp4`，
  也可以直接修改脚本顶部的
  `INPUT_VIDEO` 指向其他文件。
- `output/`：运行时自动创建，结果按 `{动作}/{视频名}/` 分类保存。

人体角度 CSV 仅保存 frame、time 和左右肩/肘/髋/膝 8 个角度；完整 Halpe26
关键点只在内存中用于可视化视频，不额外写入 CSV。

输入视频和运行结果属于本地实验数据，已由 `.gitignore` 排除，不会提交到 Git。
