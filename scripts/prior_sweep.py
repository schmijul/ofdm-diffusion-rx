#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.study_utils import (
    linear_slope,
    load_csv_rows,
    normalize_unique_bit_priors,
    parse_float_list,
    parse_int_list,
    prior_slug,
    summarize_delta_curve,
)
from src.utils import load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", default="config/exp_uniform_fast.yaml")
    p.add_argument("--priors", default="0.1,0.2,0.3,0.4,0.5")
    p.add_argument("--outdir", default="results/prior_sweep")
    p.add_argument("--n-frames", type=int, default=20)
    p.add_argument("--seeds", default="1,2")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-val", type=int, default=None)
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--skip-plots", action="store_true")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def write_config(path: Path, cfg: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def train_if_needed(config_path: Path, outdir: Path, args) -> Path:
    checkpoint = outdir / "best_model.pt"
    if checkpoint.exists() and not args.force_train:
        print(f"Reusing checkpoint: {checkpoint}")
        return checkpoint

    cmd = [sys.executable, "scripts/train.py", "--config", str(config_path), "--outdir", str(outdir)]
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.n_train is not None:
        cmd.extend(["--n-train", str(args.n_train)])
    if args.n_val is not None:
        cmd.extend(["--n-val", str(args.n_val)])
    run(cmd)
    return checkpoint


def main():
    args = parse_args()
    if args.n_frames <= 0:
        raise ValueError("--n-frames must be positive")
    parse_int_list(args.seeds)
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("--epochs must be positive when provided")
    if args.n_train is not None and args.n_train <= 0:
        raise ValueError("--n-train must be positive when provided")
    if args.n_val is not None and args.n_val <= 0:
        raise ValueError("--n-val must be positive when provided")

    bit_priors = normalize_unique_bit_priors(parse_float_list(args.priors))

    outdir = Path(args.outdir)
    config_dir = outdir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for bit_one_prob in bit_priors:
        cfg = load_config(args.base_config)
        cfg["modulation"]["bit_one_prob"] = float(bit_one_prob)

        slug = prior_slug(bit_one_prob)
        cfg_path = config_dir / f"{slug}.yaml"
        study_dir = outdir / slug
        study_dir.mkdir(parents=True, exist_ok=True)
        write_config(cfg_path, cfg)

        checkpoint = train_if_needed(cfg_path, study_dir, args)
        run(
            [
                sys.executable,
                "scripts/benchmark.py",
                "--config",
                str(cfg_path),
                "--checkpoint",
                str(checkpoint),
                "--outdir",
                str(study_dir),
                "--n-frames",
                str(args.n_frames),
                "--seeds",
                args.seeds,
            ]
        )

        delta_summary = summarize_delta_curve(load_csv_rows(study_dir / "benchmark_summary.csv"))
        summary_rows.append(
            {
                "bit_one_prob": float(bit_one_prob),
                "prior_skew": abs(float(bit_one_prob) - 0.5),
                "avg_delta": delta_summary["avg_delta"],
                "best_delta": delta_summary["best_delta"],
                "worst_delta": delta_summary["worst_delta"],
                "snr_min_db": delta_summary["snr_min_db"],
                "snr_max_db": delta_summary["snr_max_db"],
                "n_snrs": delta_summary["n_snrs"],
            }
        )

    summary_rows.sort(key=lambda row: row["bit_one_prob"])

    csv_path = outdir / "prior_sweep_summary.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("bit_one_prob,prior_skew,avg_delta,best_delta,worst_delta,snr_min_db,snr_max_db,n_snrs\n")
        for row in summary_rows:
            f.write(
                f"{row['bit_one_prob']},{row['prior_skew']},{row['avg_delta']},{row['best_delta']},"
                f"{row['worst_delta']},{row['snr_min_db']},{row['snr_max_db']},{row['n_snrs']}\n"
            )

    bit_one_prob_values = [row["bit_one_prob"] for row in summary_rows]
    prior_skew_values = [row["prior_skew"] for row in summary_rows]
    average_delta_values = [row["avg_delta"] for row in summary_rows]
    best_delta_values = [row["best_delta"] for row in summary_rows]
    slope_avg_delta_vs_skew = linear_slope(prior_skew_values, average_delta_values) if len(summary_rows) >= 2 else float("nan")

    if not args.skip_plots:
        plt.figure(figsize=(7.2, 4.8))
        plt.axhline(0.0, color="black", linewidth=1.0)
        plt.plot(bit_one_prob_values, average_delta_values, marker="o", label="Average delta over SNR")
        plt.plot(bit_one_prob_values, best_delta_values, marker="d", label="Best delta over SNR")
        plt.xlabel("Bit-one prior probability")
        plt.ylabel("Delta BER (Diffusion - MMSE)")
        plt.title("Diffusion Gain vs Bit Prior")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "prior_sweep_delta_vs_prior.png", dpi=160)
        plt.savefig(outdir / "prior_sweep_delta_vs_prior.pdf")
        plt.close()

    summary_md = outdir / "prior_sweep_summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# Prior Sweep Summary",
                "",
                f"- Number of priors: {len(summary_rows)}",
                f"- Slope avg_delta vs prior_skew: {slope_avg_delta_vs_skew:.4e}",
                "- Interpretation: more negative slope means diffusion gains increase as priors become more skewed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
