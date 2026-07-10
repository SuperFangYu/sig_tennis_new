"""采集层置信度阈值（研究二统一配置）。"""

HUMAN_KPT_CONF_THRESH = 0.30
RACKET_DETECT_CONF_THRESH = 0.10
RACKET_KPT_VALID_THRESH = 0.25

# 球拍框过滤（与检测 conf 分离，保留较低值以兼容原逻辑）
RACKET_BOX_CONF_THRESH = 0.005
