#!/usr/bin/env python3
import argparse
import csv
import itertools
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to benchmark_seed_summary.csv")
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bootstrap_ci(values: torch.Tensor, n_bootstrap: int = 4000, alpha: float = 0.05) -> tuple[float, float]:
    if values.numel() == 1:
        value = float(values.item())
        return value, value
    n = values.numel()
    idx = torch.randint(0, n, (n_bootstrap, n))
    samples = values[idx].mean(dim=1)
    lo = float(torch.quantile(samples, alpha / 2.0).item())
    hi = float(torch.quantile(samples, 1.0 - alpha / 2.0).item())
    return lo, hi


def exact_sign_flip_pvalue(values: torch.Tensor) -> float:
    # Two-sided randomization test around zero mean using all sign flips.
    n = values.numel()
    obs = abs(float(values.mean().item()))
    if n == 0:
        return float("nan")
    all_means = []
    for bits in itertools.product([-1.0, 1.0], repeat=n):
        signs = torch.tensor(bits, dtype=torch.float32)
        all_means.append(abs(float((values * signs).mean().item())))
    count = sum(1 for x in all_means if x >= obs - 1e-12)
    return float(count) / float(len(all_means))


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(Path(args.csv))
    if not rows:
        raise ValueError("Seed summary CSV is empty")

    snrs = sorted({float(r["snr_db"]) for r in rows})
    report_rows = []

    for snr_db in snrs:
        snr_rows = [r for r in rows if float(r["snr_db"]) == snr_db]
        deltas = torch.tensor([float(r["delta_diff_minus_mmse_seed_mean"]) for r in snr_rows], dtype=torch.float32)
        mean_delta = float(deltas.mean().item())
        ci_lo, ci_hi = bootstrap_ci(deltas)
        p_value = exact_sign_flip_pvalue(deltas)
        report_rows.append(
            {
                "snr_db": snr_db,
                "n_seeds": int(deltas.numel()),
                "mean_delta": mean_delta,
                "ci95_lo": ci_lo,
                "ci95_hi": ci_hi,
                "p_value_signflip": p_value,
            }
        )

    # Aggregate over SNR by averaging each seed's delta across SNRs.
    seed_ids = sorted({int(r["seed"]) for r in rows})
    seed_means = []
    for seed in seed_ids:
        seed_rows = [r for r in rows if int(r["seed"]) == seed]
        vals = [float(r["delta_diff_minus_mmse_seed_mean"]) for r in seed_rows]
        seed_means.append(sum(vals) / len(vals))
    seed_tensor = torch.tensor(seed_means, dtype=torch.float32)
    agg_mean = float(seed_tensor.mean().item())
    agg_lo, agg_hi = bootstrap_ci(seed_tensor)
    agg_p = exact_sign_flip_pvalue(seed_tensor)

    csv_path = outdir / "significance_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["snr_db", "n_seeds", "mean_delta", "ci95_lo", "ci95_hi", "p_value_signflip"])
        for row in report_rows:
            writer.writerow(
                [
                    row["snr_db"],
                    row["n_seeds"],
                    row["mean_delta"],
                    row["ci95_lo"],
                    row["ci95_hi"],
                    row["p_value_signflip"],
                ]
            )
        writer.writerow(["aggregate", len(seed_ids), agg_mean, agg_lo, agg_hi, agg_p])

    md_path = outdir / "significance_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Significance Summary\n\n")
        for row in report_rows:
            f.write(
                f"- SNR {row['snr_db']:.1f} dB: "
                f"delta={row['mean_delta']:.4e}, "
                f"95% CI [{row['ci95_lo']:.4e}, {row['ci95_hi']:.4e}], "
                f"p={row['p_value_signflip']:.4e}\n"
            )
        f.write(
            "\n"
            f"- Aggregate across SNRs: delta={agg_mean:.4e}, "
            f"95% CI [{agg_lo:.4e}, {agg_hi:.4e}], p={agg_p:.4e}\n"
        )


if __name__ == "__main__":
    main()
