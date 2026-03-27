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
    p.add_argument(
        "--diff-prior-weight",
        type=float,
        default=0.35,
        help="Prior strength for optional prior-aware demap in diffusion branch (0 disables)",
    )
    p.add_argument(
        "--mmse-prior-weight",
        type=float,
        default=0.0,
        help="Prior strength for optional prior-aware demap in MMSE branch (0 disables)",
    )
    p.add_argument("--max-bytes", type=int, default=0, help="Optional cap on processed bytes (0 means full file)")
    p.add_argument("--start-byte", type=int, default=0, help="Optional byte offset into the file before slicing")
    return p.parse_args()


def _qam16_map_bits_with_priors(symbols: torch.Tensor, bit_one_probs: list[float], prior_weight: float) -> torch.Tensor:
    if prior_weight <= 0.0:
        return qam16_to_bits(symbols).long()
    if len(bit_one_probs) != 4:
        return qam16_to_bits(symbols).long()

    device = symbols.device
    dtype = symbols.real.dtype
    levels = torch.tensor([-3.0, -1.0, 1.0, 3.0], device=device, dtype=dtype) / (10.0**0.5)
    grid_i, grid_q = torch.meshgrid(levels, levels, indexing="ij")
    const = grid_i.reshape(-1) + 1j * grid_q.reshape(-1)

    bits_lut = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        device=device,
        dtype=torch.long,
    )

    probs = torch.tensor(bit_one_probs, device=device, dtype=dtype).clamp(1e-4, 1.0 - 1e-4)
    logp1 = torch.log(probs).unsqueeze(0)
    logp0 = torch.log(1.0 - probs).unsqueeze(0)
    bits_lut_f = bits_lut.to(dtype=dtype)
    log_prior = torch.sum(bits_lut_f * logp1 + (1.0 - bits_lut_f) * logp0, dim=1)

    dist2 = torch.abs(symbols.unsqueeze(1) - const.unsqueeze(0)).pow(2)
    score = dist2 - prior_weight * log_prior.unsqueeze(0)
    best = torch.argmin(score, dim=1)
    return bits_lut[best].reshape(-1).long()


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
    if args.max_bytes < 0:
        raise ValueError("--max-bytes must be non-negative")
    if args.start_byte < 0:
        raise ValueError("--start-byte must be non-negative")
    if args.diff_prior_weight < 0.0:
        raise ValueError("--diff-prior-weight must be non-negative")
    if args.mmse_prior_weight < 0.0:
        raise ValueError("--mmse-prior-weight must be non-negative")

    train_paths = [p.strip() for p in args.train_texts.split(",") if p.strip()]
    if train_paths:
        assert_disjoint_test_file(test_path, train_paths)

    device = get_device(cfg)
    ddpm = load_diffusion(cfg, Path(args.checkpoint), device)
    bit_pos_priors = cfg.get("modulation", {}).get("bit_one_prob_per_position", None)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    recon_dir = outdir / "reconstructed"
    recon_dir.mkdir(parents=True, exist_ok=True)

    full_raw = test_path.read_bytes()
    if args.start_byte > len(full_raw):
        raise ValueError("--start-byte is beyond end of file")
    raw = full_raw[args.start_byte :]
    if args.max_bytes > 0:
        raw = raw[: args.max_bytes]
    if not raw:
        raise ValueError("Selected byte range is empty")

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
        mmse_prior_all = []
        diff_all = []

        for frame_idx in range(n_frames):
            b0 = frame_idx * n_bits_frame
            b1 = b0 + n_bits_frame
            bits_frame = padded[b0:b1]

            frame = simulate_received_frame(cfg, snr_db=snr_db, bits_tx=bits_frame)
            out_mmse = run_receiver_on_frame(frame, method="ls_mmse", perfect_csi=False)

            tx_all.append(out_mmse["bits_tx"].long())
            mmse_all.append(out_mmse["bits_rx"].long())
            mmse_prior_all.append(
                _qam16_map_bits_with_priors(
                    out_mmse["equalized_symbols"],
                    bit_one_probs=bit_pos_priors if isinstance(bit_pos_priors, list) else [],
                    prior_weight=float(args.mmse_prior_weight),
                )
            )

            if ddpm is not None:
                x_eq = out_mmse["equalized_symbols"].to(device)
                x_eq_real = torch.stack([x_eq.real, x_eq.imag], dim=1).float()
                snr_tensor = torch.full((x_eq_real.shape[0], 1), snr_db, device=device)
                with torch.no_grad():
                    x_dn = ddpm.denoise_from_equalized(x_eq_real, snr_tensor)
                x_dn_complex = real_to_complex(x_dn.cpu())
                bits_dn = _qam16_map_bits_with_priors(
                    x_dn_complex,
                    bit_one_probs=bit_pos_priors if isinstance(bit_pos_priors, list) else [],
                    prior_weight=float(args.diff_prior_weight),
                )
                diff_all.append(bits_dn)
            else:
                diff_all.append(out_mmse["bits_rx"].long())

        tx_bits = torch.cat(tx_all)[: bits.numel()]
        mmse_bits = torch.cat(mmse_all)[: bits.numel()]
        mmse_prior_bits = torch.cat(mmse_prior_all)[: bits.numel()]
        diff_bits = torch.cat(diff_all)[: bits.numel()]

        mmse_ber = bit_error_rate(tx_bits, mmse_bits)
        mmse_prior_ber = bit_error_rate(tx_bits, mmse_prior_bits)
        diff_ber = bit_error_rate(tx_bits, diff_bits)

        ref_bytes = bits_to_bytes(tx_bits)
        mmse_bytes = bits_to_bytes(mmse_bits)
        mmse_prior_bytes = bits_to_bytes(mmse_prior_bits)
        diff_bytes = bits_to_bytes(diff_bits)

        mmse_ber_byte = byte_error_rate(ref_bytes, mmse_bytes)
        mmse_prior_ber_byte = byte_error_rate(ref_bytes, mmse_prior_bytes)
        diff_ber_byte = byte_error_rate(ref_bytes, diff_bytes)

        mmse_text = mmse_bytes.decode("utf-8", errors="replace")
        mmse_prior_text = mmse_prior_bytes.decode("utf-8", errors="replace")
        diff_text = diff_bytes.decode("utf-8", errors="replace")

        mmse_char = char_mismatch_rate(ref_text, mmse_text)
        mmse_prior_char = char_mismatch_rate(ref_text, mmse_prior_text)
        diff_char = char_mismatch_rate(ref_text, diff_text)

        rows.append(
            {
                "snr_db": snr_db,
                "mmse_ber": mmse_ber,
                "mmse_prior_ber": mmse_prior_ber,
                "diff_ber": diff_ber,
                "mmse_byte_error": mmse_ber_byte,
                "mmse_prior_byte_error": mmse_prior_ber_byte,
                "diff_byte_error": diff_ber_byte,
                "mmse_char_mismatch": mmse_char,
                "mmse_prior_char_mismatch": mmse_prior_char,
                "diff_char_mismatch": diff_char,
            }
        )

        (recon_dir / f"mmse_snr_{snr_db:.1f}.txt").write_text(mmse_text, encoding="utf-8", errors="replace")
        (recon_dir / f"mmse_prior_snr_{snr_db:.1f}.txt").write_text(
            mmse_prior_text, encoding="utf-8", errors="replace"
        )
        (recon_dir / f"diffusion_snr_{snr_db:.1f}.txt").write_text(diff_text, encoding="utf-8", errors="replace")

        print(
            f"snr={snr_db:.1f} "
            f"ber(mmse={mmse_ber:.4e}, mmse_prior={mmse_prior_ber:.4e}, diff={diff_ber:.4e}) "
            f"byte(mmse={mmse_ber_byte:.4e}, mmse_prior={mmse_prior_ber_byte:.4e}, diff={diff_ber_byte:.4e})"
        )

    csv_path = outdir / "text_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "snr_db",
                "mmse_ber",
                "mmse_prior_ber",
                "diff_ber",
                "mmse_byte_error",
                "mmse_prior_byte_error",
                "diff_byte_error",
                "mmse_char_mismatch",
                "mmse_prior_char_mismatch",
                "diff_char_mismatch",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    metadata_path = outdir / "run_metadata.txt"
    metadata_path.write_text(
        "\n".join(
            [
                f"test_file={test_path}",
                f"file_size_bytes={len(full_raw)}",
                f"start_byte={args.start_byte}",
                f"max_bytes={args.max_bytes}",
                f"processed_bytes={len(raw)}",
                f"n_bits={bits.numel()}",
                f"n_frames={n_frames}",
                f"config={args.config}",
                f"checkpoint={args.checkpoint}",
                f"seed={args.seed}",
                f"diff_prior_weight={args.diff_prior_weight}",
                f"mmse_prior_weight={args.mmse_prior_weight}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

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
