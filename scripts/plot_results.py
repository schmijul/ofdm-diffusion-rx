#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import torch

from src.classical_receiver import run_classical_frame
from src.utils import load_config, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--csv", default="results/ber_results.csv")
    p.add_argument("--outdir", default="results")
    return p.parse_args()


def load_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def plot_ber(rows, outdir: Path):
    snr = [float(r["snr_db"]) for r in rows]
    ls_zf = [float(r["ls_zf"]) for r in rows]
    ls_mmse = [float(r["ls_mmse"]) for r in rows]
    genie = [float(r["perfect_mmse"]) for r in rows]
    diff = [float(r["diffusion_mmse"]) for r in rows]

    plt.figure(figsize=(7, 4.6))
    plt.semilogy(snr, ls_zf, marker="o", label="LS+ZF")
    plt.semilogy(snr, ls_mmse, marker="s", label="LS+MMSE")
    plt.semilogy(snr, genie, marker="^", label="PerfectCSI+MMSE")
    if not all(torch.isnan(torch.tensor(diff))):
        plt.semilogy(snr, diff, marker="d", label="Diffusion+MMSE")

    plt.xlabel("SNR [dB]")
    plt.ylabel("BER")
    plt.title("BER vs SNR")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "ber_vs_snr.png", dpi=160)
    plt.savefig(outdir / "ber_vs_snr.pdf")
    plt.close()


def plot_constellations(cfg: dict, outdir: Path):
    snr_values = cfg["evaluation"]["constellation_snrs_db"]
    fig, axes = plt.subplots(len(snr_values), 3, figsize=(10, 3.2 * len(snr_values)))
    if len(snr_values) == 1:
        axes = [axes]

    for row_idx, snr_db in enumerate(snr_values):
        out = run_classical_frame(cfg, snr_db=float(snr_db), method="ls_mmse", perfect_csi=False)
        tx = out["tx_symbols"]
        rx = out["rx_symbols"]
        eq = out["equalized_symbols"]

        for ax, data, title in zip(
            axes[row_idx],
            [tx, rx, eq],
            [f"TX (SNR={snr_db} dB)", "After Channel", "After MMSE"],
        ):
            ax.scatter(data.real[:1000], data.imag[:1000], s=6, alpha=0.45)
            ax.set_xlabel("I")
            ax.set_ylabel("Q")
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
            ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(outdir / "constellation_grid.png", dpi=160)
    fig.savefig(outdir / "constellation_grid.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(Path(args.csv))
    plot_ber(rows, outdir)
    plot_constellations(cfg, outdir)


if __name__ == "__main__":
    main()
