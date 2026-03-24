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

from src.classical_receiver import run_receiver_on_frame, simulate_received_frame
from src.demapper import qam16_to_bits
from src.diffusion.ddpm import DDPM
from src.diffusion.model import ResidualMLPDenoiser
from src.diffusion.noise_schedule import NoiseSchedule
from src.metrics import bit_error_rate
from src.text_utils import (
    assert_disjoint_test_file,
    bits_to_bytes,
    bytes_to_bits,
    byte_error_rate,
    char_mismatch_rate,
)
from src.utils import get_device, load_config, real_to_complex, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True, help="Path to test .txt file")
    p.add_argument("--train-texts", default="", help="Comma-separated train text paths for leak check")
    p.add_argument("--config", default="config/compare.yaml")
    p.add_argument("--checkpoint", default="results/compare_run/best_model.pt")
    p.add_argument("--outdir", default="results/text_benchmark")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def snr_grid(cfg: dict) -> list[float]:
    start, end = cfg["snr_range_db"]
    step = float(cfg["snr_step_db"])
    n = int(round((end - start) / step)) + 1
    return [float(start + i * step) for i in range(n)]


def bits_per_frame(cfg: dict) -> int:
    n_sc = int(cfg["ofdm"]["n_subcarriers"])
    n_p = int(cfg["ofdm"]["n_pilot_subcarriers"])
    n_sym = int(cfg["ofdm"]["n_ofdm_symbols"])
    n_data = (n_sc - n_p) * n_sym
    return n_data * 4


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


def plot_metric(rows: list[dict], key_mmse: str, key_diff: str, ylabel: str, title: str, out_png: Path, out_pdf: Path):
    snr = [float(r["snr_db"]) for r in rows]
    mmse = [float(r[key_mmse]) for r in rows]
    diff = [float(r[key_diff]) for r in rows]

    plt.figure(figsize=(7, 4.6))
    plt.plot(snr, mmse, marker="s", label="LS+MMSE")
    plt.plot(snr, diff, marker="d", label="Diffusion+MMSE")
    plt.xlabel("SNR [dB]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.savefig(out_pdf)
    plt.close()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)

    test_path = Path(args.text)
    if not test_path.exists():
        raise FileNotFoundError(f"Test text not found: {test_path}")

    train_paths = [p.strip() for p in args.train_texts.split(",") if p.strip()]
    if train_paths:
        assert_disjoint_test_file(test_path, train_paths)

    device = get_device(cfg)
    ddpm = load_diffusion(cfg, Path(args.checkpoint), device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    recon_dir = outdir / "reconstructed"
    recon_dir.mkdir(parents=True, exist_ok=True)

    raw = test_path.read_bytes()
    ref_text = raw.decode("utf-8", errors="replace")
    bits = bytes_to_bits(raw)

    n_bits_frame = bits_per_frame(cfg)
    n_frames = (bits.numel() + n_bits_frame - 1) // n_bits_frame
    padded = torch.zeros(n_frames * n_bits_frame, dtype=torch.long)
    padded[: bits.numel()] = bits

    rows = []

    for snr_db in snr_grid(cfg):
        tx_all = []
        mmse_all = []
        diff_all = []

        for frame_idx in range(n_frames):
            b0 = frame_idx * n_bits_frame
            b1 = b0 + n_bits_frame
            bits_frame = padded[b0:b1]

            frame = simulate_received_frame(cfg, snr_db=snr_db, bits_tx=bits_frame)
            out_mmse = run_receiver_on_frame(frame, method="ls_mmse", perfect_csi=False)

            tx_all.append(out_mmse["bits_tx"].long())
            mmse_all.append(out_mmse["bits_rx"].long())

            if ddpm is not None:
                x_eq = out_mmse["equalized_symbols"].to(device)
                x_eq_real = torch.stack([x_eq.real, x_eq.imag], dim=1).float()
                snr_tensor = torch.full((x_eq_real.shape[0], 1), snr_db, device=device)
                with torch.no_grad():
                    x_dn = ddpm.denoise_from_equalized(x_eq_real, snr_tensor)
                x_dn_complex = real_to_complex(x_dn.cpu())
                bits_dn = qam16_to_bits(x_dn_complex).long()
                diff_all.append(bits_dn)
            else:
                diff_all.append(out_mmse["bits_rx"].long())

        tx_bits = torch.cat(tx_all)[: bits.numel()]
        mmse_bits = torch.cat(mmse_all)[: bits.numel()]
        diff_bits = torch.cat(diff_all)[: bits.numel()]

        mmse_ber = bit_error_rate(tx_bits, mmse_bits)
        diff_ber = bit_error_rate(tx_bits, diff_bits)

        ref_bytes = bits_to_bytes(tx_bits)
        mmse_bytes = bits_to_bytes(mmse_bits)
        diff_bytes = bits_to_bytes(diff_bits)

        mmse_ber_byte = byte_error_rate(ref_bytes, mmse_bytes)
        diff_ber_byte = byte_error_rate(ref_bytes, diff_bytes)

        mmse_text = mmse_bytes.decode("utf-8", errors="replace")
        diff_text = diff_bytes.decode("utf-8", errors="replace")

        mmse_char = char_mismatch_rate(ref_text, mmse_text)
        diff_char = char_mismatch_rate(ref_text, diff_text)

        rows.append(
            {
                "snr_db": snr_db,
                "mmse_ber": mmse_ber,
                "diff_ber": diff_ber,
                "mmse_byte_error": mmse_ber_byte,
                "diff_byte_error": diff_ber_byte,
                "mmse_char_mismatch": mmse_char,
                "diff_char_mismatch": diff_char,
            }
        )

        (recon_dir / f"mmse_snr_{snr_db:.1f}.txt").write_text(mmse_text, encoding="utf-8", errors="replace")
        (recon_dir / f"diffusion_snr_{snr_db:.1f}.txt").write_text(diff_text, encoding="utf-8", errors="replace")

        print(
            f"snr={snr_db:.1f} "
            f"ber(mmse={mmse_ber:.4e}, diff={diff_ber:.4e}) "
            f"byte(mmse={mmse_ber_byte:.4e}, diff={diff_ber_byte:.4e})"
        )

    csv_path = outdir / "text_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "snr_db",
                "mmse_ber",
                "diff_ber",
                "mmse_byte_error",
                "diff_byte_error",
                "mmse_char_mismatch",
                "diff_char_mismatch",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    plot_metric(
        rows,
        key_mmse="mmse_ber",
        key_diff="diff_ber",
        ylabel="BER",
        title="Text Transmission BER vs SNR",
        out_png=outdir / "text_ber_vs_snr.png",
        out_pdf=outdir / "text_ber_vs_snr.pdf",
    )
    plot_metric(
        rows,
        key_mmse="mmse_byte_error",
        key_diff="diff_byte_error",
        ylabel="Byte Error Rate",
        title="Text Transmission Byte Error vs SNR",
        out_png=outdir / "text_byte_error_vs_snr.png",
        out_pdf=outdir / "text_byte_error_vs_snr.pdf",
    )


if __name__ == "__main__":
    main()
