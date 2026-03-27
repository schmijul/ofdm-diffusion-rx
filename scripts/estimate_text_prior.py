#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.text_utils import estimate_qam16_bit_priors_from_text_files


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--texts",
        required=True,
        help="Comma-separated list of train .txt paths used to estimate bit priors",
    )
    p.add_argument(
        "--max-bytes-per-file",
        type=int,
        default=0,
        help="Optional cap per file (0 means full file)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    text_paths = [p.strip() for p in args.texts.split(",") if p.strip()]
    p_global, p_pos, n_bytes = estimate_qam16_bit_priors_from_text_files(
        text_paths, max_bytes_per_file=args.max_bytes_per_file
    )

    print("Estimated 16-QAM bit priors from text corpus")
    print(f"files={len(text_paths)} total_bytes={n_bytes}")
    print(f"bit_one_prob={p_global:.6f}")
    print("bit_one_prob_per_position=[" + ", ".join(f"{v:.6f}" for v in p_pos) + "]")
    print()
    print("YAML snippet:")
    print("modulation:")
    print("  order: 16")
    print(f"  bit_one_prob: {p_global:.6f}")
    print("  bit_one_prob_per_position: [" + ", ".join(f"{v:.6f}" for v in p_pos) + "]")


if __name__ == "__main__":
    main()
