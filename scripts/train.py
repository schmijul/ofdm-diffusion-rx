#!/usr/bin/env python3
import argparse
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import EqualizedSymbolDataset, generate_symbol_dataset
from src.diffusion.ddpm import DDPM
from src.diffusion.model import build_denoiser_from_config, build_prior_context_from_config
from src.diffusion.noise_schedule import NoiseSchedule
from src.text_utils import estimate_qam16_bit_priors_from_text_files
from src.utils import get_device, load_config, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-val", type=int, default=None)
    p.add_argument("--train-texts", default="", help="Comma-separated train .txt paths for prior estimation")
    p.add_argument("--max-bytes-per-text", type=int, default=0, help="Optional cap per text for prior estimation")
    p.add_argument("--outdir", default="results")
    return p.parse_args()


def evaluate_loss(ddpm: DDPM, loader: DataLoader, device: torch.device) -> float:
    ddpm.model.eval()
    losses = []
    with torch.no_grad():
        for _, x_clean, snr_db in loader:
            x_clean = x_clean.to(device)
            snr_db = snr_db.to(device)
            losses.append(ddpm.p_losses(x_clean, snr_db).item())
    return float(sum(losses) / max(len(losses), 1))
def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.train_texts.strip():
        text_paths = [p.strip() for p in args.train_texts.split(",") if p.strip()]
        p_global, p_pos, n_bytes = estimate_qam16_bit_priors_from_text_files(
            text_paths, max_bytes_per_file=args.max_bytes_per_text
        )
        cfg = copy.deepcopy(cfg)
        cfg.setdefault("modulation", {})
        cfg["modulation"]["bit_one_prob"] = p_global
        cfg["modulation"]["bit_one_prob_per_position"] = p_pos
        print(
            "using text-estimated priors: "
            f"files={len(text_paths)} bytes={n_bytes} "
            f"bit_one_prob={p_global:.6f} "
            f"bit_one_prob_per_position={[round(v, 6) for v in p_pos]}"
        )
    set_seed(int(cfg["seed"]))
    device = get_device(cfg)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n_train = args.n_train or int(cfg["training"]["n_train_samples"])
    n_val = args.n_val or int(cfg["training"]["n_val_samples"])
    epochs = args.epochs or int(cfg["training"]["epochs"])
    batch_size = int(cfg["training"]["batch_size"])

    snr_min, snr_max = cfg["training"]["snr_train_range_db"]

    train_ds = EqualizedSymbolDataset(generate_symbol_dataset(cfg, n_train, snr_min, snr_max))
    val_ds = EqualizedSymbolDataset(generate_symbol_dataset(cfg, n_val, snr_min, snr_max))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model_cfg = cfg["diffusion"]["model"]
    model = build_denoiser_from_config(model_cfg).to(device)

    schedule = NoiseSchedule(
        n_timesteps=int(cfg["diffusion"]["n_timesteps"]),
        beta_start=float(cfg["diffusion"]["beta_start"]),
        beta_end=float(cfg["diffusion"]["beta_end"]),
        schedule=str(cfg["diffusion"]["schedule"]),
    )
    prior_context = build_prior_context_from_config(cfg, device)
    ddpm = DDPM(
        model,
        schedule,
        device,
        prior_context=prior_context,
        bit_loss_weight=float(cfg.get("training", {}).get("bit_loss_weight", 0.0)),
        bit_logit_temperature=float(cfg.get("training", {}).get("bit_logit_temperature", 24.0)),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["lr"]))

    train_losses = []
    val_losses = []
    best_val = float("inf")
    best_path = outdir / "best_model.pt"

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for _, x_clean, snr_db in tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}"):
            x_clean = x_clean.to(device)
            snr_db = snr_db.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = ddpm.p_losses(x_clean, snr_db)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
        val_loss = evaluate_loss(ddpm, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                },
                best_path,
            )

        print(f"epoch={epoch + 1} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    # Save loss curves and CSV-friendly log.
    log_path = outdir / "train_log.csv"
    with log_path.open("w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss\n")
        for i, (tl, vl) in enumerate(zip(train_losses, val_losses), start=1):
            f.write(f"{i},{tl},{vl}\n")

    plt.figure(figsize=(7, 4.5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("DDPM Training Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "training_curves.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
