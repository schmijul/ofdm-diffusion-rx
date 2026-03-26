#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.study_utils import load_csv_rows, summarize_delta_curve


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--uniform-config", default="config/exp_uniform_fast.yaml")
    p.add_argument("--non-iid-config", default="config/exp_non_iid_fast.yaml")
    p.add_argument("--uniform-name", default="uniform")
    p.add_argument("--non-iid-name", default="non_iid")
    p.add_argument("--outdir", default="results/regime_study")
    p.add_argument("--n-frames", type=int, default=30)
    p.add_argument("--seeds", default="1,2")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-val", type=int, default=None)
    p.add_argument("--force-train", action="store_true")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_trained(config_path: str, outdir: Path, args) -> Path:
    checkpoint = outdir / "best_model.pt"
    if checkpoint.exists() and not args.force_train:
        print(f"Reusing checkpoint: {checkpoint}")
        return checkpoint

    cmd = [sys.executable, "scripts/train.py", "--config", config_path, "--outdir", str(outdir)]
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.n_train is not None:
        cmd.extend(["--n-train", str(args.n_train)])
    if args.n_val is not None:
        cmd.extend(["--n-val", str(args.n_val)])
    run(cmd)
    return checkpoint


def run_benchmark(config_path: str, checkpoint: Path, outdir: Path, n_frames: int, seeds: str) -> Path:
    run(
        [
            sys.executable,
            "scripts/benchmark.py",
            "--config",
            config_path,
            "--checkpoint",
            str(checkpoint),
            "--outdir",
            str(outdir),
            "--n-frames",
            str(n_frames),
            "--seeds",
            seeds,
        ]
    )
    run(
        [
            sys.executable,
            "scripts/plot_benchmark.py",
            "--csv",
            str(outdir / "benchmark_summary.csv"),
            "--outdir",
            str(outdir),
        ]
    )
    return outdir / "benchmark_summary.csv"


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    uniform_dir = outdir / args.uniform_name
    non_iid_dir = outdir / args.non_iid_name
    uniform_dir.mkdir(parents=True, exist_ok=True)
    non_iid_dir.mkdir(parents=True, exist_ok=True)

    ckpt_uniform = ensure_trained(args.uniform_config, uniform_dir, args)
    ckpt_non_iid = ensure_trained(args.non_iid_config, non_iid_dir, args)

    csv_uniform = run_benchmark(args.uniform_config, ckpt_uniform, uniform_dir, args.n_frames, args.seeds)
    csv_non_iid = run_benchmark(args.non_iid_config, ckpt_non_iid, non_iid_dir, args.n_frames, args.seeds)

    run(
        [
            sys.executable,
            "scripts/plot_regime_comparison.py",
            "--uniform-csv",
            str(csv_uniform),
            "--non-iid-csv",
            str(csv_non_iid),
            "--outdir",
            str(outdir),
        ]
    )

    rows_uniform = load_csv_rows(csv_uniform)
    rows_non_iid = load_csv_rows(csv_non_iid)
    summary_uniform = summarize_delta_curve(rows_uniform)
    summary_non_iid = summarize_delta_curve(rows_non_iid)

    summary_path = outdir / "regime_summary.md"
    summary_path.write_text(
        "\n".join(
            [
                "# Regime Study Summary",
                "",
                f"- Uniform avg delta: {summary_uniform['avg_delta']:.4e}",
                f"- Uniform best delta: {summary_uniform['best_delta']:.4e}",
                f"- Uniform worst delta: {summary_uniform['worst_delta']:.4e}",
                f"- Non-IID avg delta: {summary_non_iid['avg_delta']:.4e}",
                f"- Non-IID best delta: {summary_non_iid['best_delta']:.4e}",
                f"- Non-IID worst delta: {summary_non_iid['worst_delta']:.4e}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
