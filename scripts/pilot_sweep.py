#!/usr/bin/env python3
import argparse
import copy
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import torch

from src.classical_receiver import run_classical_frame
from src.study_utils import parse_int_list
from src.utils import load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/exp_uniform_large.yaml")
    parser.add_argument("--pilot-counts", default="4,8,16")
    parser.add_argument("--snrs", default="0,4,8,12")
    parser.add_argument("--n-frames", type=int, default=60)
    parser.add_argument("--seeds", default="1,2")
    parser.add_argument("--outdir", default="results/pilot_sweep")
    return parser.parse_args()


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(tensor.mean().item()), float(tensor.std(unbiased=True).item())


def main():
    args = parse_args()
    if args.n_frames <= 0:
        raise ValueError("--n-frames must be positive")

    pilot_counts = parse_int_list(args.pilot_counts)
    snrs = [float(x) for x in args.snrs.split(",") if x.strip()]
    seeds = parse_int_list(args.seeds)
    if not snrs:
        raise ValueError("--snrs must contain at least one value")

    base_cfg = load_config(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_csv = outdir / "pilot_sweep_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pilot_count",
                "snr_db",
                "ls_mmse_mean",
                "ls_mmse_std",
                "perfect_mmse_mean",
                "perfect_mmse_std",
                "estimation_gap_mean",
                "estimation_gap_std",
                "n_seeds",
                "n_frames_per_seed",
            ]
        )

        for pilot_count in pilot_counts:
            if pilot_count <= 0:
                raise ValueError("pilot counts must be positive")
            if pilot_count >= int(base_cfg["ofdm"]["n_subcarriers"]):
                raise ValueError("pilot counts must be smaller than the number of subcarriers")

            cfg = copy.deepcopy(base_cfg)
            cfg["ofdm"]["n_pilot_subcarriers"] = int(pilot_count)

            for snr_db in snrs:
                ls_mmse_seed_means = []
                perfect_seed_means = []
                gap_seed_means = []

                for seed in seeds:
                    set_seed(seed)
                    ls_mmse_runs = []
                    perfect_runs = []

                    for _ in range(args.n_frames):
                        out_ls = run_classical_frame(cfg, snr_db=snr_db, method="ls_mmse", perfect_csi=False)
                        out_perfect = run_classical_frame(cfg, snr_db=snr_db, method="perfect_mmse", perfect_csi=True)
                        ls_mmse_runs.append(out_ls["ber"])
                        perfect_runs.append(out_perfect["ber"])

                    ls_seed = float(torch.tensor(ls_mmse_runs).mean().item())
                    perfect_seed = float(torch.tensor(perfect_runs).mean().item())
                    ls_mmse_seed_means.append(ls_seed)
                    perfect_seed_means.append(perfect_seed)
                    gap_seed_means.append(ls_seed - perfect_seed)

                ls_mean, ls_std = mean_std(ls_mmse_seed_means)
                perfect_mean, perfect_std = mean_std(perfect_seed_means)
                gap_mean, gap_std = mean_std(gap_seed_means)

                writer.writerow(
                    [
                        pilot_count,
                        snr_db,
                        ls_mean,
                        ls_std,
                        perfect_mean,
                        perfect_std,
                        gap_mean,
                        gap_std,
                        len(seeds),
                        args.n_frames,
                    ]
                )
                handle.flush()

                print(
                    f"pilots={pilot_count} snr={snr_db:.1f} "
                    f"ls_mmse={ls_mean:.4e} perfect={perfect_mean:.4e} gap={gap_mean:.4e}"
                )

    rows = list(csv.DictReader(out_csv.open("r", encoding="utf-8")))
    for metric, filename, ylabel in [
        ("ls_mmse_mean", "pilot_sweep_ls_mmse.png", "LS+MMSE BER"),
        ("estimation_gap_mean", "pilot_sweep_estimation_gap.png", "BER gap (LS+MMSE - Perfect-CSI MMSE)"),
    ]:
        plt.figure(figsize=(7.2, 4.8))
        for pilot_count in pilot_counts:
            pilot_rows = [r for r in rows if int(r["pilot_count"]) == pilot_count]
            snr_values = [float(r["snr_db"]) for r in pilot_rows]
            metric_values = [float(r[metric]) for r in pilot_rows]
            plt.plot(snr_values, metric_values, marker="o", label=f"{pilot_count} pilots")
        plt.xlabel("SNR [dB]")
        plt.ylabel(ylabel)
        plt.title("Pilot-count diagnostic sweep")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
