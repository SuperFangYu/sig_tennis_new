# 离线实验数据目录

- `input/`：放入要手动分析的视频。四个独立脚本默认寻找
  `forehand.mp4`、`backhand.mp4`、`serve.mp4`、`volley.mp4`，也可以直接修改脚本顶部的
  `INPUT_VIDEO` 指向其他文件。
- `output/`：运行时自动创建，结果按 `{动作}/{视频名}/` 分类保存。

输入视频和运行结果属于本地实验数据，已由 `.gitignore` 排除，不会提交到 Git。
