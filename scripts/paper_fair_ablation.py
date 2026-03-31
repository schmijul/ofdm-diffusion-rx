#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/compare_text_real_long.yaml")
    p.add_argument("--outdir", default="results/paper_long_run")
    p.add_argument("--train-texts", default="data/grundgesetz.txt,data/text8.txt")
    p.add_argument("--max-bytes-per-text", type=int, default=2000000)
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--seeds", default="1,2,3,4,5")
    p.add_argument("--diff-prior-weights", default="0.25,0.35,0.45")
    p.add_argument("--max-bytes", type=int, default=20000)
    p.add_argument("--grundgesetz-start-byte", type=int, default=0)
    p.add_argument("--text8-start-byte", type=int, default=1000000)
    return p.parse_args()


def parse_int_list(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected non-empty integer list")
    return vals


def parse_float_list(text: str) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected non-empty float list")
    return vals


def run(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_mean_ber(csv_path: Path) -> tuple[float, float, float]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    mmse = sum(float(r["mmse_ber"]) for r in rows) / max(len(rows), 1)
    mmse_prior = sum(float(r["mmse_prior_ber"]) for r in rows) / max(len(rows), 1)
    diff = sum(float(r["diff_ber"]) for r in rows) / max(len(rows), 1)
    return mmse, mmse_prior, diff


def maybe_train(args, outdir: Path) -> Path:
    ckpt = outdir / "train" / "best_model.pt"
    train_log = outdir / "train" / "train_log.csv"
    curves = outdir / "train" / "training_curves.png"
    train_complete = ckpt.exists() and train_log.exists() and curves.exists()
    if args.skip_train:
        if not ckpt.exists():
            raise FileNotFoundError(f"--skip-train set but checkpoint missing: {ckpt}")
        return ckpt
    if train_complete and not args.force_train:
        return ckpt
    (outdir / "train").mkdir(parents=True, exist_ok=True)
    run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "scripts/train.py"),
            "--config",
            args.config,
            "--train-texts",
            args.train_texts,
            "--max-bytes-per-text",
            str(args.max_bytes_per_text),
            "--outdir",
            str(outdir / "train"),
        ]
    )
    return ckpt


def benchmark_one(
    *,
    cfg_path: str,
    checkpoint: Path,
    text_path: str,
    start_byte: int,
    max_bytes: int,
    seed: int,
    prior_weight: float,
    outdir: Path,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_csv = outdir / "text_metrics.csv"
    if metrics_csv.exists():
        return metrics_csv
    run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "scripts/text_benchmark.py"),
            "--text",
            text_path,
            "--config",
            cfg_path,
            "--checkpoint",
            str(checkpoint),
            "--outdir",
            str(outdir),
            "--seed",
            str(seed),
            "--max-bytes",
            str(max_bytes),
            "--start-byte",
            str(start_byte),
            "--diff-prior-weight",
            str(prior_weight),
            "--mmse-prior-weight",
            str(prior_weight),
        ]
    )
    return metrics_csv


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint = maybe_train(args, outdir)
    seeds = parse_int_list(args.seeds)
    weights = parse_float_list(args.diff_prior_weights)

    corpora = [
        ("grundgesetz", "data/grundgesetz.txt", int(args.grundgesetz_start_byte)),
        ("text8", "data/text8.txt", int(args.text8_start_byte)),
    ]

    rows: list[dict] = []
    for corpus_name, text_path, start_byte in corpora:
        for prior_weight in weights:
            for seed in seeds:
                run_dir = (
                    outdir
                    / "bench"
                    / corpus_name
                    / f"w_{prior_weight:.2f}"
                    / f"seed_{seed}"
                )
                metrics_csv = benchmark_one(
                    cfg_path=args.config,
                    checkpoint=checkpoint,
                    text_path=text_path,
                    start_byte=start_byte,
                    max_bytes=int(args.max_bytes),
                    seed=seed,
                    prior_weight=prior_weight,
                    outdir=run_dir,
                )
                mmse, mmse_prior, diff = read_mean_ber(metrics_csv)
                rows.append(
                    {
                        "corpus": corpus_name,
                        "seed": seed,
                        "prior_weight": prior_weight,
                        "mean_mmse_ber": mmse,
                        "mean_mmse_prior_ber": mmse_prior,
                        "mean_diff_ber": diff,
                        "delta_diff_minus_mmse_prior": diff - mmse_prior,
                        "delta_diff_minus_mmse": diff - mmse,
                        "metrics_csv": str(metrics_csv),
                    }
                )

    summary_csv = outdir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "corpus",
                "seed",
                "prior_weight",
                "mean_mmse_ber",
                "mean_mmse_prior_ber",
                "mean_diff_ber",
                "delta_diff_minus_mmse_prior",
                "delta_diff_minus_mmse",
                "metrics_csv",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        key = (str(row["corpus"]), float(row["prior_weight"]))
        grouped.setdefault(key, []).append(row)
    agg = []
    for (corpus, weight), g in sorted(grouped.items()):
        n = len(g)
        agg.append(
            {
                "corpus": corpus,
                "prior_weight": weight,
                "n_runs": n,
                "mean_mmse_ber": sum(float(x["mean_mmse_ber"]) for x in g) / n,
                "mean_mmse_prior_ber": sum(float(x["mean_mmse_prior_ber"]) for x in g) / n,
                "mean_diff_ber": sum(float(x["mean_diff_ber"]) for x in g) / n,
                "mean_delta_diff_minus_mmse_prior": sum(float(x["delta_diff_minus_mmse_prior"]) for x in g) / n,
                "mean_delta_diff_minus_mmse": sum(float(x["delta_diff_minus_mmse"]) for x in g) / n,
            }
        )

    agg_csv = outdir / "summary_agg.csv"
    with agg_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "corpus",
                "prior_weight",
                "n_runs",
                "mean_mmse_ber",
                "mean_mmse_prior_ber",
                "mean_diff_ber",
                "mean_delta_diff_minus_mmse_prior",
                "mean_delta_diff_minus_mmse",
            ],
        )
        writer.writeheader()
        writer.writerows(agg)

    manifest = {
        "config": args.config,
        "checkpoint": str(checkpoint),
        "seeds": seeds,
        "prior_weights": weights,
        "max_bytes": int(args.max_bytes),
        "corpora": [
            {"name": name, "path": path, "start_byte": start}
            for (name, path, start) in corpora
        ],
        "summary_csv": str(summary_csv),
        "summary_agg_csv": str(agg_csv),
    }
    with (outdir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {summary_csv}")
    print(f"Wrote {agg_csv}")
    print(f"Wrote {outdir / 'manifest.json'}")


if __name__ == "__main__":
    main()
