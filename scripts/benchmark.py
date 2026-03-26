#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.classical_receiver import run_receiver_on_frame, simulate_received_frame
from src.demapper import qam16_to_bits
from src.diffusion.ddpm import DDPM
from src.diffusion.model import ResidualMLPDenoiser
from src.diffusion.noise_schedule import NoiseSchedule
from src.metrics import bit_error_rate
from src.study_utils import parse_int_list
from src.utils import get_device, load_config, real_to_complex, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--checkpoint", default="results/full/best_model.pt")
    p.add_argument("--outdir", default="results/benchmark")
    p.add_argument("--n-frames", type=int, default=200)
    p.add_argument("--seeds", default="11,22,33")
    return p.parse_args()


def snr_grid(cfg: dict) -> list[float]:
    start, end = cfg["snr_range_db"]
    step = float(cfg["snr_step_db"])
    n = int(round((end - start) / step)) + 1
    return [float(start + i * step) for i in range(n)]


def load_diffusion(cfg: dict, checkpoint: Path, device: torch.device):
    if not checkpoint.exists():
        return None

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


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    t = torch.tensor(xs, dtype=torch.float64)
    return float(t.mean().item()), float(t.std(unbiased=True).item())


def main():
    args = parse_args()
    if args.n_frames <= 0:
        raise ValueError("--n-frames must be positive")
    cfg = load_config(args.config)
    device = get_device(cfg)
    seeds = parse_int_list(args.seeds)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ddpm = load_diffusion(cfg, Path(args.checkpoint), device)

    out_csv = outdir / "benchmark_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "snr_db",
                "ls_zf_mean",
                "ls_zf_std",
                "ls_mmse_mean",
                "ls_mmse_std",
                "perfect_mmse_mean",
                "perfect_mmse_std",
                "diffusion_mmse_mean",
                "diffusion_mmse_std",
                "delta_diff_minus_mmse_mean",
                "delta_diff_minus_mmse_std",
                "diffusion_gain_vs_mmse_mean",
                "diffusion_gain_vs_mmse_std",
                "n_seeds",
                "n_frames_per_seed",
            ]
        )

        for snr_db in snr_grid(cfg):
            zf_seed_means = []
            mmse_seed_means = []
            genie_seed_means = []
            diff_seed_means = []
            delta_seed_means = []

            for seed in seeds:
                set_seed(seed)
                zf_runs = []
                mmse_runs = []
                genie_runs = []
                diff_runs = []

                for _ in range(args.n_frames):
                    frame = simulate_received_frame(cfg, snr_db=snr_db)
                    out_zf = run_receiver_on_frame(frame, method="ls_zf", perfect_csi=False)
                    out_mmse = run_receiver_on_frame(frame, method="ls_mmse", perfect_csi=False)
                    out_genie = run_receiver_on_frame(frame, method="perfect_mmse", perfect_csi=True)

                    zf_runs.append(out_zf["ber"])
                    mmse_runs.append(out_mmse["ber"])
                    genie_runs.append(out_genie["ber"])

                    if ddpm is not None:
                        x_eq = out_mmse["equalized_symbols"].to(device)
                        x_eq_real = torch.stack([x_eq.real, x_eq.imag], dim=1).float()
                        snr_tensor = torch.full((x_eq_real.shape[0], 1), snr_db, device=device)
                        with torch.no_grad():
                            x_dn = ddpm.denoise_from_equalized(x_eq_real, snr_tensor)
                        x_dn_complex = real_to_complex(x_dn.cpu())
                        bits_dn = qam16_to_bits(x_dn_complex)
                        diff_runs.append(bit_error_rate(out_mmse["bits_tx"].long(), bits_dn.long()))

                zf_seed = float(torch.tensor(zf_runs).mean().item())
                mmse_seed = float(torch.tensor(mmse_runs).mean().item())
                genie_seed = float(torch.tensor(genie_runs).mean().item())
                zf_seed_means.append(zf_seed)
                mmse_seed_means.append(mmse_seed)
                genie_seed_means.append(genie_seed)

                if diff_runs:
                    diff_seed = float(torch.tensor(diff_runs).mean().item())
                    diff_seed_means.append(diff_seed)
                    delta_seed_means.append(diff_seed - mmse_seed)

            zf_mean, zf_std = mean_std(zf_seed_means)
            mmse_mean, mmse_std = mean_std(mmse_seed_means)
            genie_mean, genie_std = mean_std(genie_seed_means)
            diff_mean, diff_std = mean_std(diff_seed_means)
            delta_mean, delta_std = mean_std(delta_seed_means)

            writer.writerow(
                [
                    snr_db,
                    zf_mean,
                    zf_std,
                    mmse_mean,
                    mmse_std,
                    genie_mean,
                    genie_std,
                    diff_mean,
                    diff_std,
                    delta_mean,
                    delta_std,
                    delta_mean,
                    delta_std,
                    len(seeds),
                    args.n_frames,
                ]
            )

            print(
                f"snr={snr_db:.1f} "
                f"mmse={mmse_mean:.4e}±{mmse_std:.2e} "
                f"diff={diff_mean:.4e}±{diff_std:.2e} "
                f"delta(diff-mmse)={delta_mean:.4e}±{delta_std:.2e}"
            )


if __name__ == "__main__":
    main()
