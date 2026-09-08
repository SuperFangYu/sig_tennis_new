# 中间结果

每个试次运行后建立 `{trial_id}/`，其中：

- `rtmpose_8_angles.csv`：视频首帧归零后的 RTMPose 左右肩、肘、髋、膝 8 角。
- `qualisys_8_angles.csv`：Qualisys 首帧归零、3D marker 投影到 ZY 后的同定义 8 角。
- `video_events.csv`：人工填写每个视频动作的 `start/contact/end` 秒数，`definition` 列提供动作提示。
- `qualisys_events.csv`：人工填写相同动作在 Qualisys 中的 `start/contact/end` 相对秒数。

事件表的 `repetition` 必须一一对应；例如两边的 `repetition=3` 都表示同一次第三拍动作。
若一个原始文件含多次击球，每次动作各填写三行事件，不能把整段文件只当成一次动作。
这些文件是可追溯对齐依据，但属于本地实验数据，不提交到 Git。
