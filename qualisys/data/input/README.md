# Qualisys 验证输入

这里不需要 manifest。程序只按 `video/` 与 `qtm/` 的同名文件自动配对。

文件名固定为 `英文名_英文动作_编号`，例如：

```text
video/fy_zs_1.avi
qtm/fy_zs_1.tsv
```

动作缩写：`zs=正手`、`fs=反手`、`jjzs=正手截击`、`jjfs=反手截击`、`fq=发球`。
也支持完整英文 `forehand`、`backhand`、`forehand_volley`、`backhand_volley`、`serve`。

视频与 TSV 主文件名必须完全相同，末尾编号必须是正整数。`qtm/` 只放 `DATA_INCLUDED=3D`
的 marker TSV，不放测力台 analog 文件。真实输入不会提交到 Git。
