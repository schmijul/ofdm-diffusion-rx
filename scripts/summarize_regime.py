#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.study_utils import load_csv_rows, summarize_delta_curve


def format_optional(value: float | int | None) -> str:
    return "" if value is None else str(value)


def summary_row(regime: str, summary: dict) -> str:
    return ",".join(
        [
            regime,
            str(summary["avg_delta"]),
            str(summary["best_delta"]),
            str(summary["worst_delta"]),
            str(summary["snr_min_db"]),
            str(summary["snr_max_db"]),
            str(summary["n_snrs"]),
            str(summary["n_diffusion_wins"]),
            format_optional(summary.get("avg_mmse")),
            format_optional(summary.get("avg_diffusion")),
            format_optional(summary.get("avg_relative_ber_reduction_pct")),
        ]
    )


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
                f"- Uniform diffusion wins: {uniform_summary['n_diffusion_wins']}/{uniform_summary['n_snrs']} SNR points",
                f"- Non-IID avg delta: {non_iid_summary['avg_delta']:.4e}",
                f"- Non-IID best delta: {non_iid_summary['best_delta']:.4e}",
                f"- Non-IID worst delta: {non_iid_summary['worst_delta']:.4e}",
                f"- Non-IID diffusion wins: {non_iid_summary['n_diffusion_wins']}/{non_iid_summary['n_snrs']} SNR points",
                f"- Non-IID avg LS+MMSE BER: {non_iid_summary.get('avg_mmse', 0.0):.4e}",
                f"- Non-IID avg Diffusion+MMSE BER: {non_iid_summary.get('avg_diffusion', 0.0):.4e}",
                f"- Non-IID relative BER reduction: {non_iid_summary.get('avg_relative_ber_reduction_pct', 0.0):.2f}%",
                f"- Supports non-IID gain hypothesis: {'yes' if supports_hypothesis else 'no'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary_csv.write_text(
        "\n".join(
            [
                (
                    "regime,avg_delta,best_delta,worst_delta,snr_min_db,snr_max_db,n_snrs,"
                    "n_diffusion_wins,avg_mmse,avg_diffusion,avg_relative_ber_reduction_pct"
                ),
                summary_row("uniform", uniform_summary),
                summary_row("non_iid", non_iid_summary),
                f"hypothesis_support,{int(supports_hypothesis)},,,,,,,,,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
