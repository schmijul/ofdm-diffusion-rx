#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--regime",
        action="append",
        required=True,
        help="Label and path pair in the form label:path_to_regime_dir",
    )
    p.add_argument("--outdir", default="results/frontend_compare")
    return p.parse_args()


def parse_label_and_path(items: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"Invalid --regime value: {item}. Expected label:path")
        label, path = item.split(":", 1)
        label = label.strip()
        path = Path(path.strip())
        if not label:
            raise ValueError(f"Invalid --regime label in: {item}")
        parsed.append((label, path))
    return parsed


def load_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def avg(values: list[float]) -> float:
    return float(sum(values) / len(values))


def main():
    args = parse_args()
    regimes = parse_label_and_path(args.regime)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, regime_path in regimes:
        uniform_csv = regime_path / "uniform" / "benchmark_summary.csv"
        non_iid_csv = regime_path / "non_iid" / "benchmark_summary.csv"
        if not uniform_csv.exists() or not non_iid_csv.exists():
            raise FileNotFoundError(f"Missing benchmark CSVs under: {regime_path}")

        uniform_rows = load_csv_rows(uniform_csv)
        non_iid_rows = load_csv_rows(non_iid_csv)
        rows.append(
            {
                "label": label,
                "uniform_avg_delta": avg([float(r["delta_diff_minus_mmse_mean"]) for r in uniform_rows]),
                "non_iid_avg_delta": avg([float(r["delta_diff_minus_mmse_mean"]) for r in non_iid_rows]),
                "uniform_avg_mmse": avg([float(r["ls_mmse_mean"]) for r in uniform_rows]),
                "non_iid_avg_mmse": avg([float(r["ls_mmse_mean"]) for r in non_iid_rows]),
            }
        )

    csv_path = outdir / "frontend_regime_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label",
                "uniform_avg_delta",
                "non_iid_avg_delta",
                "uniform_avg_mmse",
                "non_iid_avg_mmse",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["label"],
                    row["uniform_avg_delta"],
                    row["non_iid_avg_delta"],
                    row["uniform_avg_mmse"],
                    row["non_iid_avg_mmse"],
                ]
            )

    labels = [row["label"] for row in rows]
    x = list(range(len(labels)))
    width = 0.38

    plt.figure(figsize=(7.6, 4.8))
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.bar([v - width / 2 for v in x], [row["uniform_avg_delta"] for row in rows], width=width, label="Uniform avg delta")
    plt.bar([v + width / 2 for v in x], [row["non_iid_avg_delta"] for row in rows], width=width, label="Non-IID avg delta")
    plt.xticks(x, labels)
    plt.ylabel("Average Delta BER (Diffusion - MMSE)")
    plt.title("Front-End Comparison: Diffusion Gain Stability")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "frontend_regime_delta_comparison.png", dpi=160)
    plt.savefig(outdir / "frontend_regime_delta_comparison.pdf")
    plt.close()

    plt.figure(figsize=(7.6, 4.8))
    plt.bar([v - width / 2 for v in x], [row["uniform_avg_mmse"] for row in rows], width=width, label="Uniform LS+MMSE BER")
    plt.bar([v + width / 2 for v in x], [row["non_iid_avg_mmse"] for row in rows], width=width, label="Non-IID LS+MMSE BER")
    plt.xticks(x, labels)
    plt.ylabel("Average LS+MMSE BER")
    plt.title("Front-End Comparison: Classical BER")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "frontend_regime_mmse_comparison.png", dpi=160)
    plt.savefig(outdir / "frontend_regime_mmse_comparison.pdf")
    plt.close()

    md_path = outdir / "frontend_regime_comparison.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Front-End Regime Comparison\n\n")
        for row in rows:
            f.write(
                f"- {row['label']}: "
                f"uniform avg delta={row['uniform_avg_delta']:.4e}, "
                f"non-IID avg delta={row['non_iid_avg_delta']:.4e}, "
                f"uniform avg mmse={row['uniform_avg_mmse']:.4e}, "
                f"non-IID avg mmse={row['non_iid_avg_mmse']:.4e}\n"
            )


if __name__ == "__main__":
    main()
