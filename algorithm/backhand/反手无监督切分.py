import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from moviepy import VideoFileClip

# 解决图表中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 核心参数配置 =====================
# ⚠️ 注意：请确保这些路径指向你当前的实验视频和对应的 CSV 数据
TARGET_VIDEO = r"D:\FY\mediapipe_demo\视频\个人\750025740.mp4"
RACKET_CSV = r"result_analysis\750025740_kpt_t_backend.csv"
BODY_CSV = r"D:\FY\rtm\result_analysis\750025740_body_backhand_rtmpose.csv"
OUTPUT_DIR = "cropped_actions_backhand"  # 【修改】输出文件夹改为 backhand

video_name = os.path.splitext(os.path.basename(TARGET_VIDEO))[0]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 【基础动作检测设置】
PROMINENCE_THRESHOLD = 45.0
MIN_DISTANCE_SEC = 1.6  # 稍微拉大动作间距，防止准备阶段过长导致交叉

# 🚀 【核心修改：解决准备阶段缺失 - 实施非对称自适应扩张配置】
# 我们不再使用统一的 EXTEND_RATIO，而是分开控制

# A. 针对【准备阶段】(前面) 的进取型配置
# 目的：强行向前多切一点，捕捉侧身 Unit Turn
PREP_EXTEND_RATIO = 1.0  # 动态补偿量：额外扩张该动作原始物理时长的 120% (原为 80%)
PREP_MIN_BUFFER = 1.0  # 🚨 强行保底：无论动作多快，击球前最少留够 1.0 秒准备时间 (原为 0.5)
PREP_MAX_BUFFER = 1.5  # 强制封顶：防止前面留白太多

# B. 针对【随挥阶段】(后面) 的保守型配置 (保持原样即可)
FOLLOW_EXTEND_RATIO = 0.3  # 随挥通常很快，不需要扩张太多
FOLLOW_MIN_BUFFER = 0.2  # 随挥保底 0.4 秒
FOLLOW_MAX_BUFFER = 0.4  # 随挥封顶 0.7 秒

# 【分类特征阈值】(反手专项)
MIN_SWING_WIDTH = 350
BODY_TOLERANCE = 150

# ===================== 2. 多模态数据融合 =====================
print(f"🚀 正在加载并强行清洗多模态数据...")


def robust_read_csv(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        clean_text = f.read()
    clean_text = clean_text.replace("'", "").replace("‘", "").replace("’", "")
    try:
        df_clean = pd.read_csv(io.StringIO(clean_text), engine='python', on_bad_lines='skip')
    except TypeError:
        df_clean = pd.read_csv(io.StringIO(clean_text), engine='python', error_bad_lines=False)

    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    return df_clean.dropna(subset=['frame'])


df_racket = robust_read_csv(RACKET_CSV)
df_body = robust_read_csv(BODY_CSV)

df_racket['frame'] = df_racket['frame'].astype(int)
df_body['frame'] = df_body['frame'].astype(int)
df = pd.merge(df_racket, df_body, on='frame', how='inner', suffixes=('', '_drop'))

if len(df) == 0:
    print("\n❌ 严重错误：数据对齐后为空！请检查 CSV 文件是否匹配。")
    exit()

T = df['y_clean'].values
X = df['x_clean'].values
Body_X = df['body_center_x'].values
time_axis = df['time'].values
fps = int(round(len(df) / time_axis[-1])) if time_axis[-1] > 0 else 30

# ===================== 3. 提取击球锚点 (波峰检测) =====================
peaks, properties = find_peaks(T, prominence=PROMINENCE_THRESHOLD, distance=int(MIN_DISTANCE_SEC * fps))
left_bases = properties["left_bases"]
right_bases = properties["right_bases"]
print(f"✅ 基础识别完成：共发现 {len(peaks)} 个潜在动作波动。")

# ===================== 4. 核心逻辑：特征过滤与【非对称自适应切分】 =====================
backhand_intervals = []
backhand_peaks = []
other_peaks = []

print("🔍 正在应用自适应运动学特征进行反手多维度筛选及【准备阶段修正】...")

for i, p in enumerate(peaks):
    start_idx = left_bases[i]
    end_idx = right_bases[i]
    hit_time = time_axis[p]

    # --- 【核心修改点】步骤 A: 计算【非对称】自适应扩张时间 ---
    original_duration = time_axis[end_idx] - time_axis[start_idx]

    # 1. 计算前面的缓冲 (进取型)
    prep_buffer = np.clip(original_duration * PREP_EXTEND_RATIO, PREP_MIN_BUFFER, PREP_MAX_BUFFER)

    # 2. 计算后面的缓冲 (保守型)
    follow_buffer = np.clip(original_duration * FOLLOW_EXTEND_RATIO, FOLLOW_MIN_BUFFER, FOLLOW_MAX_BUFFER)

    # --- 步骤 B: 提取特征进行【反手】判定 (逻辑保持不变) ---
    avg_body_x = np.nanmean(Body_X[start_idx:end_idx + 1])
    min_x_prep = np.min(X[start_idx:p + 1])  # 引拍阶段的最左点
    max_x_follow = np.max(X[p:end_idx + 1])  # 随挥阶段的最右点

    is_prep_left = min_x_prep < (avg_body_x + BODY_TOLERANCE)
    swing_width = max_x_follow - min_x_prep
    is_valid_width = swing_width > MIN_SWING_WIDTH

    # --- 步骤 C: 决策树与日志打印 ---
    if is_prep_left and is_valid_width:
        backhand_peaks.append(p)

        # 【核心修改点】应用非对称动态边界
        # 前面减去 prep_buffer，后面加上 follow_buffer
        start_t = time_axis[start_idx] - prep_buffer
        end_t = time_axis[end_idx] + follow_buffer

        # 防交叉保护逻辑 (保持，但需要确保 MIN_DISTANCE_SEC 足够大)
        if i > 0:
            prev_p = peaks[i - 1]
            start_t = max(start_t, (time_axis[prev_p] + time_axis[p]) / 2.0)
        else:
            start_t = max(start_t, 0)
        if i < len(peaks) - 1:
            next_p = peaks[i + 1]
            end_t = min(end_t, (time_axis[p] + time_axis[next_p]) / 2.0)

        # 边界修正：确保 start_t 小于 contact 时间，end_t 大于 contact 时间
        start_t = min(start_t, hit_time - 0.2)  # 至少留 0.2s 给引拍

        print(
            f"  [保留] {hit_time:5.2f}s | 位移: {swing_width:4.0f}px | 前缓冲(准备): {prep_buffer:4.2f}s | 状态: 标准反手")

        backhand_intervals.append((start_t, end_t))
    else:
        other_peaks.append(p)
        reason = "未向左侧身引拍" if not is_prep_left else f"横移不足({swing_width:.0f}px)"
        print(f"  [过滤] {hit_time:5.2f}s | 状态: 排除 ({reason})")

print(f"🎯 最终结果：确认 {len(backhand_intervals)} 次标准反手 (已补偿准备阶段)。")

# ===================== 5. 可视化图表生成 (双子图架构) =====================
print("📊 正在绘制多模态轨迹验证图并保存...")
# 稍微拉大画布高度
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)

# 子图 1: Y轴运动与自适应区间
ax1.plot(time_axis, T, color='#1f77b4', alpha=0.8, linewidth=1.5, label='球拍Y轴轨迹')
ax1.plot(time_axis[backhand_peaks], T[backhand_peaks], "r*", markersize=18, markeredgecolor='black', label='确认为反手')
ax1.plot(time_axis[other_peaks], T[other_peaks], "kX", markersize=12, alpha=0.4, label='已排除波动')

for i, (st, et) in enumerate(backhand_intervals):
    ax1.axvspan(st, et, alpha=0.2, color='mediumpurple', label='非对称扩张切分区间' if i == 0 else "")
    # 在图上标出 Contact Point 方便对齐
    cp_time = time_axis[backhand_peaks[i]]
    ax1.axvline(cp_time, color='red', linestyle=':', alpha=0.5)

ax1.set_title(f"多模态轨迹特征与【准备阶段补偿】自适应分割 - 反手 [{video_name}]", fontsize=16)
ax1.set_ylabel("Y轴坐标 (像素)"), ax1.invert_yaxis(), ax1.legend(loc='lower right'), ax1.grid(True, alpha=0.3)

# 子图 2: X轴空间特征对比 (保持不变)
ax2.plot(time_axis, X, color='blue', linewidth=1.2, label='球拍 X 轴')
ax2.plot(time_axis, Body_X, color='darkorange', linestyle='--', linewidth=2, label='身体动态中心线')

for i, (st, et) in enumerate(backhand_intervals):
    ax2.axvspan(st, et, alpha=0.15, color='magenta', label='反手有效挥动区' if i == 0 else "")

ax2.set_xlabel("视频运行时间 (秒)"), ax2.set_ylabel("X轴坐标 (像素)"), ax2.legend(loc='upper right'), ax2.grid(True,
                                                                                                               alpha=0.3)

plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, f"{video_name}_backhand_prep_compensated.png")
plt.savefig(chart_path, dpi=300)
print(f"📈 验证图表已导出至: {chart_path}")
plt.show(block=False)

# ===================== 6. 视频自动切分 =====================
if len(backhand_intervals) > 0:
    print(f"✂️ 正在使用 MoviePy 2.0 进行反手切分...")
    video = VideoFileClip(TARGET_VIDEO)
    for i, (start_cut, end_cut) in enumerate(backhand_intervals):
        start_cut = max(0, start_cut)
        end_cut = min(video.duration, end_cut)

        output_filename = os.path.join(OUTPUT_DIR, f"{video_name}_Backhand_Prep_v2_{i + 1}.mp4")

        try:
            clip = video.subclipped(start_cut, end_cut)
        except AttributeError:
            clip = video.subclip(start_cut, end_cut)

        clip.write_videofile(output_filename, codec="libx264", audio=False, logger=None)
        print(f"  -> 已导出第 {i + 1} 段视频: [{start_cut:.2f}s - {end_cut:.2f}s]")

    video.close()
    print("\n✨ 反手准备阶段补偿工作流处理完成！")
else:
    print("\n⚠️ 未检测到符合条件的反手动作，未生成视频。")