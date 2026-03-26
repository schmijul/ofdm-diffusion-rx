#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--uniform-csv", required=True)
    p.add_argument("--non-iid-csv", required=True)
    p.add_argument("--outdir", default="results/regime_compare")
    return p.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows_uniform = load_rows(Path(args.uniform_csv))
    rows_non_iid = load_rows(Path(args.non_iid_csv))

    snr_u = [float(r["snr_db"]) for r in rows_uniform]
    snr_n = [float(r["snr_db"]) for r in rows_non_iid]
    if snr_u != snr_n:
        raise ValueError("SNR grids must match between uniform and non-IID CSVs")
    snr = snr_u

    delta_u = [float(r["delta_diff_minus_mmse_mean"]) for r in rows_uniform]
    delta_u_std = [float(r["delta_diff_minus_mmse_std"]) for r in rows_uniform]
    delta_n = [float(r["delta_diff_minus_mmse_mean"]) for r in rows_non_iid]
    delta_n_std = [float(r["delta_diff_minus_mmse_std"]) for r in rows_non_iid]

    plt.figure(figsize=(7.2, 4.8))
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.errorbar(snr, delta_u, yerr=delta_u_std, marker="o", capsize=3, label="Uniform prior (p=0.5)")
    plt.errorbar(snr, delta_n, yerr=delta_n_std, marker="d", capsize=3, label="Non-IID prior (p=0.2)")
    plt.xlabel("SNR [dB]")
    plt.ylabel("Delta BER (Diffusion - MMSE)")
    plt.title("Regime Comparison: Diffusion Gain vs Symbol Prior")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "regime_delta_comparison.png", dpi=160)
    plt.savefig(outdir / "regime_delta_comparison.pdf")
    plt.close()


if __name__ == "__main__":
    main()
