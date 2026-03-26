#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.study_utils import load_csv_rows, parse_int_list, summarize_delta_curve


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
    if args.n_frames <= 0:
        raise ValueError("--n-frames must be positive")
    parse_int_list(args.seeds)
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("--epochs must be positive when provided")
    if args.n_train is not None and args.n_train <= 0:
        raise ValueError("--n-train must be positive when provided")
    if args.n_val is not None and args.n_val <= 0:
        raise ValueError("--n-val must be positive when provided")

    study_root_dir = Path(args.outdir)
    study_root_dir.mkdir(parents=True, exist_ok=True)

    uniform_dir = study_root_dir / args.uniform_name
    non_iid_dir = study_root_dir / args.non_iid_name
    uniform_dir.mkdir(parents=True, exist_ok=True)
    non_iid_dir.mkdir(parents=True, exist_ok=True)

    uniform_checkpoint = ensure_trained(args.uniform_config, uniform_dir, args)
    non_iid_checkpoint = ensure_trained(args.non_iid_config, non_iid_dir, args)

    uniform_summary_csv = run_benchmark(args.uniform_config, uniform_checkpoint, uniform_dir, args.n_frames, args.seeds)
    non_iid_summary_csv = run_benchmark(args.non_iid_config, non_iid_checkpoint, non_iid_dir, args.n_frames, args.seeds)

    run(
        [
            sys.executable,
            "scripts/plot_regime_comparison.py",
            "--uniform-csv",
            str(uniform_summary_csv),
            "--non-iid-csv",
            str(non_iid_summary_csv),
            "--outdir",
            str(study_root_dir),
        ]
    )

    uniform_rows = load_csv_rows(uniform_summary_csv)
    non_iid_rows = load_csv_rows(non_iid_summary_csv)
    uniform_summary = summarize_delta_curve(uniform_rows)
    non_iid_summary = summarize_delta_curve(non_iid_rows)

    summary_path = study_root_dir / "regime_summary.md"
    summary_csv_path = study_root_dir / "regime_summary.csv"
    supports_hypothesis = (non_iid_summary["avg_delta"] < 0.0) and (uniform_summary["avg_delta"] >= 0.0)

    summary_path.write_text(
        "\n".join(
            [
                "# Regime Study Summary",
                "",
                f"- Uniform avg delta: {uniform_summary['avg_delta']:.4e}",
                f"- Uniform best delta: {uniform_summary['best_delta']:.4e}",
                f"- Uniform worst delta: {uniform_summary['worst_delta']:.4e}",
                f"- Non-IID avg delta: {non_iid_summary['avg_delta']:.4e}",
                f"- Non-IID best delta: {non_iid_summary['best_delta']:.4e}",
                f"- Non-IID worst delta: {non_iid_summary['worst_delta']:.4e}",
                f"- Supports non-IID gain hypothesis: {'yes' if supports_hypothesis else 'no'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary_csv_path.write_text(
        "\n".join(
            [
                "regime,avg_delta,best_delta,worst_delta,snr_min_db,snr_max_db,n_snrs",
                (
                    f"uniform,{uniform_summary['avg_delta']},{uniform_summary['best_delta']},"
                    f"{uniform_summary['worst_delta']},{uniform_summary['snr_min_db']},"
                    f"{uniform_summary['snr_max_db']},{uniform_summary['n_snrs']}"
                ),
                (
                    f"non_iid,{non_iid_summary['avg_delta']},{non_iid_summary['best_delta']},"
                    f"{non_iid_summary['worst_delta']},{non_iid_summary['snr_min_db']},"
                    f"{non_iid_summary['snr_max_db']},{non_iid_summary['n_snrs']}"
                ),
                f"hypothesis_support,{int(supports_hypothesis)},,,,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
