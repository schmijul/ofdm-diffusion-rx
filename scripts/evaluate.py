#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.classical_receiver import run_classical_frame
from src.demapper import qam16_to_bits
from src.diffusion.ddpm import DDPM
from src.diffusion.model import ResidualMLPDenoiser
from src.diffusion.noise_schedule import NoiseSchedule
from src.metrics import bit_error_rate
from src.utils import get_device, load_config, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--checkpoint", default="results/best_model.pt")
    p.add_argument("--outdir", default="results")
    p.add_argument("--n-frames", type=int, default=None)
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
    return DDPM(model, schedule, device)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = get_device(cfg)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n_frames = args.n_frames or int(cfg["evaluation"]["n_test_frames"])
    ddpm = load_diffusion(cfg, Path(args.checkpoint), device)

    rows = ["snr_db,ls_zf,ls_mmse,perfect_mmse,diffusion_mmse"]

    for snr_db in snr_grid(cfg):
        zf_ber = []
        mmse_ber = []
        genie_ber = []
        diff_ber = []

        for _ in range(n_frames):
            out_zf = run_classical_frame(cfg, snr_db=snr_db, method="ls_zf", perfect_csi=False)
            out_mmse = run_classical_frame(cfg, snr_db=snr_db, method="ls_mmse", perfect_csi=False)
            out_genie = run_classical_frame(cfg, snr_db=snr_db, method="perfect_mmse", perfect_csi=True)

            zf_ber.append(out_zf["ber"])
            mmse_ber.append(out_mmse["ber"])
            genie_ber.append(out_genie["ber"])

            if ddpm is not None:
                x_eq = out_mmse["equalized_symbols"].to(device)
                x_eq_real = torch.stack([x_eq.real, x_eq.imag], dim=1).float()
                snr_tensor = torch.full((x_eq_real.shape[0], 1), snr_db, device=device)
                with torch.no_grad():
                    x_dn = ddpm.denoise_from_equalized(x_eq_real, snr_tensor)
                x_dn_complex = x_dn[:, 0].cpu() + 1j * x_dn[:, 1].cpu()
                bits_dn = qam16_to_bits(x_dn_complex)
                diff_ber.append(bit_error_rate(out_mmse["bits_tx"].long(), bits_dn.long()))

        m_zf = sum(zf_ber) / len(zf_ber)
        m_mmse = sum(mmse_ber) / len(mmse_ber)
        m_genie = sum(genie_ber) / len(genie_ber)
        m_diff = (sum(diff_ber) / len(diff_ber)) if diff_ber else float("nan")

        rows.append(f"{snr_db},{m_zf},{m_mmse},{m_genie},{m_diff}")
        print(f"snr={snr_db:.1f} zf={m_zf:.4e} mmse={m_mmse:.4e} genie={m_genie:.4e} diff={m_diff:.4e}")

    (outdir / "ber_results.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
