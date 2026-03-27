#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import subprocess
import time
from pathlib import Path


START_MARKER = "<!-- PAPER_LONG_RUN_STATUS_START -->"
END_MARKER = "<!-- PAPER_LONG_RUN_STATUS_END -->"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="results/paper_long_run")
    p.add_argument("--readme", default="README.md")
    p.add_argument("--interval-sec", type=int, default=180)
    p.add_argument("--once", action="store_true")
    return p.parse_args()


def _safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def read_summary_agg(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_summary(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def process_running(pattern: str) -> bool:
    out = subprocess.run(
        ["/bin/bash", "-lc", f"ps -ef | rg '{pattern}' | rg -v rg"],
        capture_output=True,
        text=True,
    )
    return bool(out.stdout.strip())


def build_status_block(run_dir: Path) -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = read_summary(run_dir / "summary.csv")
    summary_agg = read_summary_agg(run_dir / "summary_agg.csv")
    train_ckpt = run_dir / "train" / "best_model.pt"
    is_running = process_running("scripts/paper_fair_ablation.py")

    lines: list[str] = []
    lines.append(START_MARKER)
    lines.append("### Live Long-Run Status")
    lines.append("")
    lines.append(f"- Last update: `{now}`")
    lines.append(f"- Runner active: `{'yes' if is_running else 'no'}`")
    lines.append(f"- Checkpoint present: `{'yes' if train_ckpt.exists() else 'no'}`")
    lines.append(f"- Run dir: `{run_dir}`")
    lines.append("")

    if not summary and not summary_agg:
        lines.append("No benchmark summaries yet. Training or first benchmarks are still running.")
    else:
        lines.append("Current aggregated results (`summary_agg.csv`):")
        lines.append("")
        lines.append("| corpus | weight | n_runs | mmse | mmse+prior | diff | diff-mmse+prior |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in sorted(summary_agg, key=lambda r: (r.get("corpus", ""), _safe_float(r.get("prior_weight", "nan")))):
            lines.append(
                "| "
                + f"{row.get('corpus','')} | "
                + f"{_safe_float(row.get('prior_weight','nan')):.2f} | "
                + f"{int(float(row.get('n_runs','0') or 0))} | "
                + f"{_safe_float(row.get('mean_mmse_ber','nan')):.4f} | "
                + f"{_safe_float(row.get('mean_mmse_prior_ber','nan')):.4f} | "
                + f"{_safe_float(row.get('mean_diff_ber','nan')):.4f} | "
                + f"{_safe_float(row.get('mean_delta_diff_minus_mmse_prior','nan')):+.4f} |"
            )

        best_rows = sorted(summary_agg, key=lambda r: _safe_float(r.get("mean_delta_diff_minus_mmse_prior", "nan")))
        if best_rows:
            best = best_rows[0]
            lines.append("")
            lines.append(
                "Current best (lowest `diff - mmse+prior`): "
                + f"`{best.get('corpus','')}`, "
                + f"`w={_safe_float(best.get('prior_weight','nan')):.2f}`, "
                + f"`delta={_safe_float(best.get('mean_delta_diff_minus_mmse_prior','nan')):+.4f}`"
            )

        if summary:
            lines.append("")
            lines.append(f"- Completed run files in `summary.csv`: `{len(summary)}`")

    lines.append(END_MARKER)
    return "\n".join(lines) + "\n"


def update_readme(readme_path: Path, block: str) -> None:
    text = readme_path.read_text(encoding="utf-8")
    if START_MARKER not in text or END_MARKER not in text:
        append = (
            "\n\n## Long-Run Monitor\n\n"
            + block
        )
        readme_path.write_text(text + append, encoding="utf-8")
        return
    start = text.index(START_MARKER)
    end = text.index(END_MARKER) + len(END_MARKER)
    new_text = text[:start] + block.rstrip("\n") + text[end:]
    readme_path.write_text(new_text, encoding="utf-8")


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    readme_path = Path(args.readme)
    while True:
        block = build_status_block(run_dir)
        update_readme(readme_path, block)
        if args.once:
            break
        time.sleep(max(args.interval_sec, 30))


if __name__ == "__main__":
    main()
