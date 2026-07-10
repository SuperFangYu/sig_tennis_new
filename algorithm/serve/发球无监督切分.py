"""
CLI：发球切分与双图导出。推荐使用仓库根目录下：
  python -m algorithm.serve.发球无监督切分
若模块名不便导入，可直接：
  python algorithm/serve/segmentation.py （不可用）
请从项目根设置 PYTHONPATH=. 后调用本脚本内 main。
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="发球无监督切分（球拍 CSV + 身体 CSV）")
    parser.add_argument("--video", type=str, required=True, help="输入视频")
    parser.add_argument("--racket-csv", type=str, required=True, help="球拍追踪 CSV（含 y_clean）")
    parser.add_argument("--body-csv", type=str, required=True, help="发球身体特征 CSV")
    parser.add_argument("--output-dir", type=str, default="cropped_actions_serve", help="输出目录")
    parser.add_argument("--handedness", type=str, default="right", help="持拍手：right / left")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from algorithm.serve.segmentation import run_segmentation

    run_segmentation(args.video, args.racket_csv, args.body_csv, out, handedness=args.handedness)


if __name__ == "__main__":
    main()
