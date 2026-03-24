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
from src.diffusion.ddpm import DDPM
from src.diffusion.model import ResidualMLPDenoiser
from src.diffusion.noise_schedule import NoiseSchedule
from src.utils import real_to_complex
from src.utils import load_config, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--csv", default="results/ber_results.csv")
    p.add_argument("--outdir", default="results")
    p.add_argument("--checkpoint", default="results/best_model.pt")
    return p.parse_args()


def load_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def plot_ber(rows, outdir: Path):
    snr = [float(r["snr_db"]) for r in rows]
    ls_zf = [float(r["ls_zf_ber"]) for r in rows]
    ls_mmse = [float(r["ls_mmse_ber"]) for r in rows]
    genie = [float(r["perfect_mmse_ber"]) for r in rows]
    diff = [float(r["diffusion_mmse_ber"]) for r in rows]

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


def load_diffusion(cfg: dict, checkpoint: Path):
    if not checkpoint.exists():
        return None

    device = torch.device("cpu")
    model_cfg = cfg["diffusion"]["model"]
    model = ResidualMLPDenoiser(
        input_dim=2,
        hidden_dim=int(model_cfg["hidden_dim"]),
        n_res_blocks=int(model_cfg["n_res_blocks"]),
        time_dim=int(model_cfg["time_embedding_dim"]),
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    schedule = NoiseSchedule(
        n_timesteps=int(cfg["diffusion"]["n_timesteps"]),
        beta_start=float(cfg["diffusion"]["beta_start"]),
        beta_end=float(cfg["diffusion"]["beta_end"]),
        schedule=str(cfg["diffusion"]["schedule"]),
    )
    return DDPM(model, schedule, device, inference_steps=int(cfg["diffusion"]["inference_steps"]))


def plot_constellations(cfg: dict, outdir: Path, ddpm: DDPM | None):
    snr_values = cfg["evaluation"]["constellation_snrs_db"]
    n_cols = 4 if ddpm is not None else 3
    fig, axes = plt.subplots(len(snr_values), n_cols, figsize=(3.4 * n_cols, 3.2 * len(snr_values)))
    if len(snr_values) == 1:
        axes = [axes]  # Normalize indexing.

    for row_idx, snr_db in enumerate(snr_values):
        out = run_classical_frame(cfg, snr_db=float(snr_db), method="ls_mmse", perfect_csi=False)
        tx = out["tx_symbols"]
        rx = out["rx_symbols"]
        eq = out["equalized_symbols"]

        data_list = [tx, rx, eq]
        title_list = [f"TX (SNR={snr_db} dB)", "After Channel", "After MMSE"]
        if ddpm is not None:
            x_eq_real = torch.stack([eq.real, eq.imag], dim=1).float()
            snr_tensor = torch.full((x_eq_real.shape[0], 1), float(snr_db))
            with torch.no_grad():
                x_dn = ddpm.denoise_from_equalized(x_eq_real, snr_tensor)
            dn = real_to_complex(x_dn)
            data_list.append(dn)
            title_list.append("After Diffusion")

        for ax, data, title in zip(axes[row_idx], data_list, title_list):
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
    ddpm = load_diffusion(cfg, Path(args.checkpoint))
    plot_ber(rows, outdir)
    plot_constellations(cfg, outdir, ddpm)


if __name__ == "__main__":
    main()
