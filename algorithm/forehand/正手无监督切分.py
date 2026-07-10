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
# 与后端一致的切分、动力链图、持拍手逻辑见 algorithm.forehand.segmentation.run_segmentation。
# ⚠️ 注意：请确保这些路径指向你当前的实验视频和对应的 CSV 数据
TARGET_VIDEO = r"D:\FY\mediapipe_demo\视频\个人\dark1.mp4"
RACKET_CSV = r"result_analysis\dark1_kpt_t_backend.csv"
BODY_CSV = r"D:\FY\rtm\result_analysis\dark1_body_forehand_rtmpose.csv"
OUTPUT_DIR = "cropped_actions_forehand"

video_name = os.path.splitext(os.path.basename(TARGET_VIDEO))[0]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 【基础动作检测设置】
PROMINENCE_THRESHOLD = 45.0
MIN_DISTANCE_SEC = 1.5

# 🚀 【核心创新：自适应比例扩张配置】
# 用于解决不同动作节奏下引拍和随挥完整度不一致的问题
EXTEND_RATIO = 0.8  # 动态补偿量：额外扩张该动作原始物理时长的 40%
MIN_BUFFER = 0.5  # 强制保底：哪怕动作再快，前后最少也留 0.3 秒，防止切太死
MAX_BUFFER = 0.8  # 强制封顶：哪怕动作再慢，前后最多留 0.7 秒，防止视频过长拖沓

# 【分类特征阈值】
MIN_DROP_TIME = 0.15  # 最短掉拍头时间
MIN_SWING_WIDTH = 140  # 横向必须跨越身体
BODY_TOLERANCE = 150  # 身体中心线容错

# ===================== 2. 多模态数据融合 (防弹清洗版) =====================
print(f"🚀 正在加载并强行清洗多模态数据...")


def robust_read_csv(file_path):
    """终极读取函数：处理单引号、乱码并强转数字"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        clean_text = f.read()
    # 消除 WPS/Excel 可能产生的单引号标记
    clean_text = clean_text.replace("'", "").replace("‘", "").replace("’", "")
    try:
        df_clean = pd.read_csv(io.StringIO(clean_text), engine='python', on_bad_lines='skip')
    except TypeError:
        df_clean = pd.read_csv(io.StringIO(clean_text), engine='python', error_bad_lines=False)

    # 强制将所有列转为数字类型，无法转换的变为 NaN
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    return df_clean.dropna(subset=['frame'])


df_racket = robust_read_csv(RACKET_CSV)
df_body = robust_read_csv(BODY_CSV)

# 统一帧号格式并对齐
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

# ===================== 4. 核心逻辑：特征过滤与自适应切分 =====================
forehand_intervals = []
forehand_peaks = []
other_peaks = []

print("🔍 正在应用自适应运动学特征进行多维度筛选...")

for i, p in enumerate(peaks):
    start_idx = left_bases[i]
    end_idx = right_bases[i]
    hit_time = time_axis[p]

    # --- 步骤 A: 计算自适应扩张时间 ---
    original_duration = time_axis[end_idx] - time_axis[start_idx]
    # 根据动作本身的快慢，动态计算缓冲时间，并限制在 [0.3, 0.7] 之间
    dynamic_buffer = np.clip(original_duration * EXTEND_RATIO, MIN_BUFFER, MAX_BUFFER)

    # --- 步骤 B: 提取特征进行正手判定 ---
    avg_body_x = np.nanmean(Body_X[start_idx:end_idx + 1])
    max_x_prep = np.max(X[start_idx:p + 1])
    min_x_follow = np.min(X[p:end_idx + 1])

    # 空间约束判定
    is_prep_right = max_x_prep > (avg_body_x - BODY_TOLERANCE)
    swing_width = max_x_prep - min_x_follow
    is_valid_width = swing_width > MIN_SWING_WIDTH

    # --- 步骤 C: 决策树与日志打印 ---
    if is_prep_right and is_valid_width:
        forehand_peaks.append(p)
        print(f"  [保留] {hit_time:5.2f}s | 位移: {swing_width:4.0f}px | 缓冲: {dynamic_buffer:4.2f}s | 状态: 标准正手")

        # 应用动态边界
        start_t = time_axis[start_idx] - dynamic_buffer
        end_t = time_axis[end_idx] + dynamic_buffer

        # 防交叉保护逻辑
        if i > 0:
            prev_p = peaks[i - 1]
            start_t = max(start_t, (time_axis[prev_p] + time_axis[p]) / 2.0)
        else:
            start_t = max(start_t, 0)
        if i < len(peaks) - 1:
            next_p = peaks[i + 1]
            end_t = min(end_t, (time_axis[p] + time_axis[next_p]) / 2.0)

        forehand_intervals.append((start_t, end_t))
    else:
        other_peaks.append(p)
        reason = "位置偏移" if not is_prep_right else f"横移不足({swing_width:.0f}px)"
        print(f"  [过滤] {hit_time:5.2f}s | 状态: 排除 ({reason})")

print(f"🎯 最终结果：确认 {len(forehand_intervals)} 次标准正手。")

# ===================== 5. 可视化图表生成 (双子图架构) =====================
print("📊 正在绘制多模态轨迹验证图并保存...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)

# 子图 1: Y轴运动与自适应区间
ax1.plot(time_axis, T, color='#1f77b4', alpha=0.8, linewidth=1.5, label='球拍Y轴轨迹')
ax1.plot(time_axis[forehand_peaks], T[forehand_peaks], "r*", markersize=18, markeredgecolor='black', label='确认为正手')
ax1.plot(time_axis[other_peaks], T[other_peaks], "kX", markersize=12, alpha=0.4, label='已排除波动')

for i, (st, et) in enumerate(forehand_intervals):
    ax1.axvspan(st, et, alpha=0.2, color='limegreen', label='自适应扩张切分区间' if i == 0 else "")

ax1.set_title(f"多模态时空轨迹特征与自适应缓冲分割分析 - [{video_name}]", fontsize=16)
ax1.set_ylabel("Y轴坐标 (像素)"), ax1.invert_yaxis(), ax1.legend(loc='lower right'), ax1.grid(True, alpha=0.3)

# 子图 2: X轴空间特征对比
ax2.plot(time_axis, X, color='blue', linewidth=1.2, label='球拍 X 轴')
ax2.plot(time_axis, Body_X, color='darkorange', linestyle='--', linewidth=2, label='身体动态中心线')

for i, (st, et) in enumerate(forehand_intervals):
    ax2.axvspan(st, et, alpha=0.15, color='gold', label='正手有效挥动区' if i == 0 else "")

ax2.set_xlabel("视频运行时间 (秒)"), ax2.set_ylabel("X轴坐标 (像素)"), ax2.legend(loc='upper right'), ax2.grid(True,
                                                                                                               alpha=0.3)

plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, f"{video_name}_final_analysis.png")
plt.savefig(chart_path, dpi=300)
print(f"📈 验证图表已导出至: {chart_path}")
plt.show(block=False)

# ===================== 6. 视频自动切分 =====================
if len(forehand_intervals) > 0:
    print(f"✂️ 正在使用 MoviePy 2.0 进行自适应切分...")
    video = VideoFileClip(TARGET_VIDEO)
    for i, (start_cut, end_cut) in enumerate(forehand_intervals):
        # 边界检查
        start_cut = max(0, start_cut)
        end_cut = min(video.duration, end_cut)

        output_filename = os.path.join(OUTPUT_DIR, f"{video_name}_Forehand_v2_{i + 1}.mp4")

        # MoviePy 2.0 兼容性调用
        try:
            clip = video.subclipped(start_cut, end_cut)
        except AttributeError:
            clip = video.subclip(start_cut, end_cut)

        clip.write_videofile(output_filename, codec="libx264", audio=False, logger=None)
        print(f"  -> 已导出第 {i + 1} 段视频: [{start_cut:.2f}s - {end_cut:.2f}s]")

    video.close()
    print("\n✨ 全部工作流处理完成！")
else:
    print("\n⚠️ 未检测到符合条件的动作，未生成视频。")