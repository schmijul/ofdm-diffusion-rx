#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.study_utils import load_csv_rows, summarize_delta_curve


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--uniform-csv", required=True)
    p.add_argument("--non-iid-csv", required=True)
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    uniform_summary = summarize_delta_curve(load_csv_rows(args.uniform_csv))
    non_iid_summary = summarize_delta_curve(load_csv_rows(args.non_iid_csv))
    supports_hypothesis = (non_iid_summary["avg_delta"] < 0.0) and (uniform_summary["avg_delta"] >= 0.0)

    summary_md = outdir / "regime_summary.md"
    summary_csv = outdir / "regime_summary.csv"

    summary_md.write_text(
        "\n".join(
            [
                "# Regime Study Summary",
                "",
                f"- Uniform avg delta: {uniform_summary['avg_delta']:.4e}",
                f"- Uniform best delta: {uniform_summary['best_delta']:.4e}",
                f"- Uniform worst delta: {uniform_summary['worst_delta']:.4e}",
                f"- Non-IID avg delta: {non_iid_summary['avg_delta']:.4e}",
                f"- Non-IID best delta: {non_iid_summary['best_delta']:.4e}",
                f"- Non-IID worst delta: {non_iid_summary['worst_delta']:.4e}",
                f"- Supports non-IID gain hypothesis: {'yes' if supports_hypothesis else 'no'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary_csv.write_text(
        "\n".join(
            [
                "regime,avg_delta,best_delta,worst_delta,snr_min_db,snr_max_db,n_snrs",
                (
                    f"uniform,{uniform_summary['avg_delta']},{uniform_summary['best_delta']},"
                    f"{uniform_summary['worst_delta']},{uniform_summary['snr_min_db']},"
                    f"{uniform_summary['snr_max_db']},{uniform_summary['n_snrs']}"
                ),
                (
                    f"non_iid,{non_iid_summary['avg_delta']},{non_iid_summary['best_delta']},"
                    f"{non_iid_summary['worst_delta']},{non_iid_summary['snr_min_db']},"
                    f"{non_iid_summary['snr_max_db']},{non_iid_summary['n_snrs']}"
                ),
                f"hypothesis_support,{int(supports_hypothesis)},,,,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
