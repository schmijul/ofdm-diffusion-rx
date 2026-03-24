#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--outdir", default="results/benchmark")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with Path(args.csv).open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    snr = [float(r["snr_db"]) for r in rows]
    mmse = [float(r["ls_mmse_mean"]) for r in rows]
    mmse_std = [float(r["ls_mmse_std"]) for r in rows]
    diff = [float(r["diffusion_mmse_mean"]) for r in rows]
    diff_std = [float(r["diffusion_mmse_std"]) for r in rows]
    delta = [float(r["delta_diff_minus_mmse_mean"]) for r in rows]
    delta_std = [float(r["delta_diff_minus_mmse_std"]) for r in rows]

    plt.figure(figsize=(7, 4.6))
    plt.errorbar(snr, mmse, yerr=mmse_std, marker="s", capsize=3, label="LS+MMSE")
    plt.errorbar(snr, diff, yerr=diff_std, marker="d", capsize=3, label="Diffusion+MMSE")
    plt.yscale("log")
    plt.xlabel("SNR [dB]")
    plt.ylabel("BER mean +/- std")
    plt.title("Benchmark BER with uncertainty")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "benchmark_ber_errorbars.png", dpi=160)
    plt.savefig(outdir / "benchmark_ber_errorbars.pdf")
    plt.close()

    plt.figure(figsize=(7, 4.6))
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.errorbar(snr, delta, yerr=delta_std, marker="o", capsize=3)
    plt.xlabel("SNR [dB]")
    plt.ylabel("Delta BER (Diffusion - MMSE)")
    plt.title("Diffusion gain (negative is better)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "benchmark_delta_diff_minus_mmse.png", dpi=160)
    plt.savefig(outdir / "benchmark_delta_diff_minus_mmse.pdf")
    plt.close()


if __name__ == "__main__":
    main()
