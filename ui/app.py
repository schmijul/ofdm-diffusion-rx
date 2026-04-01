from __future__ import annotations

import csv
import datetime as dt
import html
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import streamlit as st
import yaml


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
REGISTRY_PATH = RESULTS_DIR / "ui" / "runs.json"


st.set_page_config(
    page_title="OFDM Diffusion UI",
    page_icon="O",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ui-bg: #f3efe7;
          --ui-panel: rgba(255, 255, 255, 0.86);
          --ui-panel-strong: rgba(255, 255, 255, 0.96);
          --ui-border: rgba(15, 23, 42, 0.10);
          --ui-text: #10212d;
          --ui-muted: #5d6472;
          --ui-accent: #1d4ed8;
          --ui-accent-2: #0f766e;
          --ui-warn: #b45309;
          --ui-good: #047857;
        }

        [data-testid="stAppViewContainer"] {
          background:
            radial-gradient(circle at top left, rgba(29, 78, 216, 0.08), transparent 30%),
            radial-gradient(circle at top right, rgba(15, 118, 110, 0.08), transparent 28%),
            linear-gradient(180deg, #f7f3eb 0%, #fbfaf7 45%, #eef3f8 100%);
          color: var(--ui-text);
        }

        .block-container {
          padding-top: 1.4rem;
          padding-bottom: 2.5rem;
          max-width: 1280px;
        }

        .ui-hero {
          background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 54%, #0f766e 100%);
          color: white;
          border-radius: 22px;
          padding: 1.2rem 1.4rem;
          box-shadow: 0 18px 60px rgba(15, 23, 42, 0.18);
          margin-bottom: 1rem;
        }

        .ui-hero h1 {
          margin: 0 0 0.25rem 0;
          font-size: 2rem;
          letter-spacing: -0.03em;
        }

        .ui-hero p {
          margin: 0.2rem 0 0 0;
          max-width: 76ch;
          color: rgba(255,255,255,0.88);
          line-height: 1.45;
        }

        .ui-card {
          background: var(--ui-panel);
          backdrop-filter: blur(10px);
          border: 1px solid var(--ui-border);
          border-radius: 18px;
          padding: 1rem 1.05rem;
          box-shadow: 0 8px 26px rgba(15, 23, 42, 0.08);
          margin-bottom: 0.9rem;
        }

        .ui-card-strong {
          background: var(--ui-panel-strong);
        }

        .ui-section-title {
          font-size: 0.88rem;
          text-transform: uppercase;
          letter-spacing: 0.12em;
          color: var(--ui-muted);
          margin-bottom: 0.5rem;
        }

        .ui-kv {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
          gap: 0.55rem;
        }

        .ui-kv-item {
          border: 1px solid var(--ui-border);
          border-radius: 14px;
          padding: 0.65rem 0.8rem;
          background: rgba(255,255,255,0.7);
        }

        .ui-kv-label {
          font-size: 0.75rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: var(--ui-muted);
          margin-bottom: 0.2rem;
        }

        .ui-kv-value {
          font-size: 1.02rem;
          font-weight: 650;
          color: var(--ui-text);
          word-break: break-word;
        }

        .ui-chip {
          display: inline-flex;
          align-items: center;
          gap: 0.35rem;
          border-radius: 999px;
          padding: 0.18rem 0.6rem;
          border: 1px solid var(--ui-border);
          background: rgba(255,255,255,0.75);
          font-size: 0.82rem;
          color: var(--ui-text);
          margin-right: 0.35rem;
        }

        .ui-chip.good { color: var(--ui-good); }
        .ui-chip.warn { color: var(--ui-warn); }
        .ui-chip.accent { color: var(--ui-accent); }

        .ui-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 0.9rem;
        }

        .ui-table th, .ui-table td {
          border-bottom: 1px solid rgba(15, 23, 42, 0.08);
          padding: 0.42rem 0.55rem;
          text-align: left;
          vertical-align: top;
        }

        .ui-table th {
          background: rgba(15, 23, 42, 0.04);
          font-size: 0.76rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: var(--ui-muted);
        }

        .ui-log {
          font-size: 0.82rem;
          line-height: 1.35;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def short_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_registry() -> list[dict[str, Any]]:
    data = load_json(REGISTRY_PATH, [])
    return data if isinstance(data, list) else []


def save_registry(entries: list[dict[str, Any]]) -> None:
    atomic_write_json(REGISTRY_PATH, entries)


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def now_human() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_csv_list(text: str, cast=float) -> list[Any]:
    items = [part.strip() for part in text.split(",") if part.strip()]
    return [cast(item) for item in items]


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def is_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def tail_text(path: Path, max_lines: int = 40) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(lines[-max_lines:])


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(text: str | None, default: float = float("nan")) -> float:
    if text is None or text == "":
        return default
    try:
        return float(text)
    except Exception:
        return default


def fmt_float(value: Any, digits: int = 4) -> str:
    try:
        if value is None:
            return "n/a"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def fmt_signed(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):+.{digits}f}"
    except Exception:
        return "n/a"


def qam_frame_count(cfg: dict[str, Any]) -> int:
    n_sc = int(cfg["ofdm"]["n_subcarriers"])
    n_p = int(cfg["ofdm"]["n_pilot_subcarriers"])
    n_sym = int(cfg["ofdm"]["n_ofdm_symbols"])
    return (n_sc - n_p) * n_sym * 4


def snr_point_count(cfg: dict[str, Any]) -> int:
    snr_range = cfg.get("snr_range_db", [0.0, 0.0])
    if not isinstance(snr_range, (list, tuple)) or len(snr_range) < 2:
        return 1
    start = float(snr_range[0])
    end = float(snr_range[1])
    step = float(cfg.get("snr_step_db", 6.0))
    if step <= 0:
        return 1
    return int(round((end - start) / step)) + 1


def fair_expected_total(seeds_text: str, weights_text: str) -> int:
    seeds = parse_csv_list(seeds_text, int)
    weights = parse_csv_list(weights_text, float)
    return max(len(seeds) * len(weights) * 2, 0)


def launch_detached(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return int(proc.pid)


def register_run(entry: dict[str, Any]) -> None:
    entries = load_registry()
    entries = [e for e in entries if e.get("id") != entry["id"]]
    entries.append(entry)
    save_registry(entries)


def launch_fair_run(
    *,
    config_path: str,
    outdir: str,
    train_texts: str,
    max_bytes_per_text: int,
    seeds_text: str,
    weights_text: str,
    max_bytes: int,
    grundgesetz_start_byte: int,
    text8_start_byte: int,
    force_train: bool,
) -> dict[str, Any]:
    run_dir = Path(outdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "fair_ablation.py"),
        "--config",
        config_path,
        "--outdir",
        outdir,
        "--train-texts",
        train_texts,
        "--max-bytes-per-text",
        str(max_bytes_per_text),
        "--seeds",
        seeds_text,
        "--diff-prior-weights",
        weights_text,
        "--max-bytes",
        str(max_bytes),
        "--grundgesetz-start-byte",
        str(grundgesetz_start_byte),
        "--text8-start-byte",
        str(text8_start_byte),
    ]
    if force_train:
        cmd.append("--force-train")

    pid = launch_detached(cmd, run_dir / "ui_launch.log")
    entry = {
        "id": f"fair-{now_stamp()}",
        "kind": "fair_ablation",
        "title": "Fair ablation",
        "config_path": config_path,
        "outdir": outdir,
        "log_path": short_path(run_dir / "ui_launch.log"),
        "command": cmd,
        "pid": pid,
        "launched_at": now_human(),
        "expected_total": fair_expected_total(seeds_text, weights_text),
        "params": {
            "train_texts": train_texts,
            "max_bytes_per_text": max_bytes_per_text,
            "seeds": seeds_text,
            "diff_prior_weights": weights_text,
            "max_bytes": max_bytes,
            "grundgesetz_start_byte": grundgesetz_start_byte,
            "text8_start_byte": text8_start_byte,
            "force_train": force_train,
        },
    }
    register_run(entry)
    return entry


def launch_text_benchmark(
    *,
    config_path: str,
    text_path: str,
    train_texts: str,
    checkpoint: str,
    outdir: str,
    seed: int,
    diff_prior_weight: float,
    mmse_prior_weight: float,
    max_bytes: int,
    start_byte: int,
) -> dict[str, Any]:
    run_dir = Path(outdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "text_benchmark.py"),
        "--text",
        text_path,
        "--train-texts",
        train_texts,
        "--config",
        config_path,
        "--checkpoint",
        checkpoint,
        "--outdir",
        outdir,
        "--seed",
        str(seed),
        "--diff-prior-weight",
        str(diff_prior_weight),
        "--mmse-prior-weight",
        str(mmse_prior_weight),
    ]
    if max_bytes > 0:
        cmd += ["--max-bytes", str(max_bytes)]
    if start_byte > 0:
        cmd += ["--start-byte", str(start_byte)]

    pid = launch_detached(cmd, run_dir / "ui_launch.log")
    cfg = load_yaml(Path(config_path))
    expected_total = snr_point_count(cfg) if cfg else 1
    entry = {
        "id": f"text-{now_stamp()}",
        "kind": "text_benchmark",
        "title": Path(text_path).name,
        "config_path": config_path,
        "outdir": outdir,
        "log_path": short_path(run_dir / "ui_launch.log"),
        "command": cmd,
        "pid": pid,
        "launched_at": now_human(),
        "expected_total": expected_total,
        "params": {
            "text_path": text_path,
            "train_texts": train_texts,
            "checkpoint": checkpoint,
            "seed": seed,
            "diff_prior_weight": diff_prior_weight,
            "mmse_prior_weight": mmse_prior_weight,
            "max_bytes": max_bytes,
            "start_byte": start_byte,
        },
    }
    register_run(entry)
    return entry


def benchmark_progress(outdir: Path) -> tuple[int, int]:
    bench_dir = outdir / "bench"
    n_done = len(list(bench_dir.rglob("text_metrics.csv"))) if bench_dir.exists() else 0
    return n_done, 0


def parse_training_progress(log_path: Path) -> tuple[str | None, str | None]:
    text = tail_text(log_path, 60)
    matches = list(re.finditer(r"epoch\s+(\d+)/(\d+).*?train_loss=([0-9.eE+-]+).*?val_loss=([0-9.eE+-]+)", text))
    if matches:
        m = matches[-1]
        return f"{m.group(1)}/{m.group(2)}", f"train {m.group(3)} | val {m.group(4)}"
    return None, None


def collect_candidate_run_dirs() -> list[Path]:
    dirs: dict[Path, float] = {}
    for pattern in ("summary_agg.csv", "summary.csv", "text_metrics.csv"):
        for path in RESULTS_DIR.rglob(pattern):
            dirs[path.parent] = max(dirs.get(path.parent, 0.0), path.stat().st_mtime)
    for entry in load_registry():
        outdir = entry.get("outdir")
        if outdir:
            p = Path(outdir)
            if p.exists():
                dirs[p] = max(dirs.get(p, 0.0), p.stat().st_mtime)
    return [p for p, _ in sorted(dirs.items(), key=lambda item: item[1], reverse=True)]


def html_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int = 50) -> str:
    head = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body_rows = []
    for row in rows[:max_rows]:
        cells = []
        for col in columns:
            value = row.get(col, "")
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    body = "".join(body_rows)
    return f'<table class="ui-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def show_metric_cards(items: list[tuple[str, str]]) -> None:
    cols = st.columns(min(len(items), 4) if items else 1)
    for idx, (label, value) in enumerate(items):
        with cols[idx % len(cols)]:
            st.markdown(
                f"""
                <div class="ui-kv-item">
                  <div class="ui-kv-label">{html.escape(label)}</div>
                  <div class="ui-kv-value">{html.escape(value)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def plot_fair_summary(summary_agg: list[dict[str, str]]):
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in summary_agg:
        corpus = row.get("corpus", "unknown")
        grouped[corpus].append(
            (
                safe_float(row.get("prior_weight")),
                safe_float(row.get("mean_delta_diff_minus_mmse_prior")),
            )
        )
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    palette = ["#1d4ed8", "#0f766e", "#b45309", "#7c3aed"]
    for idx, (corpus, points) in enumerate(sorted(grouped.items())):
        points = sorted(points, key=lambda x: x[0])
        if not points:
            continue
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        ax.plot(xs, ys, marker="o", linewidth=2.0, color=palette[idx % len(palette)], label=corpus)
    ax.axhline(0.0, color="#334155", linestyle="--", linewidth=1)
    ax.set_xlabel("prior weight")
    ax.set_ylabel("diff - mmse+prior BER")
    ax.set_title("Fair ablation gap")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_text_metrics(rows: list[dict[str, str]]):
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    xs = [safe_float(row.get("snr_db")) for row in rows]
    series = [
        ("mmse_ber", "LS+MMSE", "#1d4ed8"),
        ("mmse_prior_ber", "LS+MMSE+prior", "#0f766e"),
        ("diff_ber", "Diffusion+MMSE", "#b45309"),
    ]
    for key, label, color in series:
        ys = [safe_float(row.get(key)) for row in rows]
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=label, color=color)
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("BER")
    ax.set_title("Text benchmark BER vs SNR")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def render_run_summary(run_dir: Path) -> None:
    summary_agg_path = run_dir / "summary_agg.csv"
    summary_path = run_dir / "summary.csv"
    text_metrics_path = run_dir / "text_metrics.csv"
    train_log_path = run_dir / "train" / "train_log.csv"
    train_png = run_dir / "train" / "training_curves.png"
    benchmark_png = run_dir / "text_ber_vs_snr.png"
    byte_png = run_dir / "text_byte_error_vs_snr.png"

    if summary_agg_path.exists():
        rows = csv_rows(summary_agg_path)
        st.markdown('<div class="ui-card ui-card-strong">', unsafe_allow_html=True)
        st.markdown('<div class="ui-section-title">Fair summary</div>', unsafe_allow_html=True)
        show_metric_cards(
            [
                ("Aggregated rows", str(len(rows))),
                ("Best gap", fmt_signed(min((safe_float(r.get("mean_delta_diff_minus_mmse_prior")) for r in rows), default=0.0))),
                ("Run dir", short_path(run_dir)),
            ]
        )
        st.markdown(html_table(rows, ["corpus", "prior_weight", "n_runs", "mean_mmse_ber", "mean_mmse_prior_ber", "mean_diff_ber", "mean_delta_diff_minus_mmse_prior"], max_rows=50), unsafe_allow_html=True)
        st.pyplot(plot_fair_summary(rows), clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if text_metrics_path.exists():
        rows = csv_rows(text_metrics_path)
        st.markdown('<div class="ui-card ui-card-strong">', unsafe_allow_html=True)
        st.markdown('<div class="ui-section-title">Text benchmark summary</div>', unsafe_allow_html=True)
        show_metric_cards(
            [
                ("SNR points", str(len(rows))),
                ("Best diffusion BER", fmt_float(min((safe_float(r.get("diff_ber")) for r in rows), default=0.0))),
                ("Run dir", short_path(run_dir)),
            ]
        )
        st.markdown(
            html_table(
                rows,
                ["snr_db", "mmse_ber", "mmse_prior_ber", "diff_ber", "mmse_byte_error", "mmse_prior_byte_error", "diff_byte_error"],
                max_rows=20,
            ),
            unsafe_allow_html=True,
        )
        st.pyplot(plot_text_metrics(rows), clear_figure=True)
        if benchmark_png.exists():
            st.image(str(benchmark_png), caption="Generated text BER plot", use_container_width=True)
        if byte_png.exists():
            st.image(str(byte_png), caption="Generated byte error plot", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if train_log_path.exists():
        rows = csv_rows(train_log_path)
        st.markdown('<div class="ui-card">', unsafe_allow_html=True)
        st.markdown('<div class="ui-section-title">Training log</div>', unsafe_allow_html=True)
        st.markdown(html_table(rows, ["epoch", "train_loss", "val_loss"], max_rows=50), unsafe_allow_html=True)
        if train_png.exists():
            st.image(str(train_png), caption="Training curves", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    inject_css()

    auto_refresh_sec = st.sidebar.slider("Auto refresh seconds", 0, 30, 8, 1)
    if auto_refresh_sec > 0:
        st.markdown(
            f'<meta http-equiv="refresh" content="{auto_refresh_sec}">',
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="ui-hero">
          <h1>OFDM Diffusion Control Deck</h1>
          <p>Launch benchmark jobs, watch live progress, and inspect generated results without staying in the terminal.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    registry = load_registry()
    active_count = sum(1 for entry in registry if is_running(entry.get("pid")))
    completed_count = sum(1 for entry in registry if not is_running(entry.get("pid")))

    st.markdown(
        """
        <div class="ui-card">
        <div class="ui-section-title">Workspace status</div>
        """,
        unsafe_allow_html=True,
    )
    show_metric_cards(
        [
            ("Tracked runs", str(len(registry))),
            ("Active processes", str(active_count)),
            ("Completed processes", str(completed_count)),
            ("Registry path", short_path(REGISTRY_PATH)),
        ]
    )
    st.markdown("</div>", unsafe_allow_html=True)

    launch_tab, live_tab, results_tab = st.tabs(["Launch", "Live", "Results"])

    with launch_tab:
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown('<div class="ui-card ui-card-strong">', unsafe_allow_html=True)
            st.markdown('<div class="ui-section-title">Fair ablation</div>', unsafe_allow_html=True)
            with st.form("launch_fair_form", clear_on_submit=False):
                config_path = st.text_input("Config", value="config/compare_text_real_followup_adapt.yaml")
                outdir = st.text_input("Output dir", value=f"results/ui/fair_{now_stamp()}")
                train_texts = st.text_input("Train texts", value="data/grundgesetz.txt,data/text8.txt")
                max_bytes_per_text = st.number_input("Max bytes per train text", min_value=1, value=2_000_000, step=100_000)
                seeds_text = st.text_input("Seeds", value="1,2,3")
                weights_text = st.text_input("Diff prior weights", value="0.55,0.65,0.75")
                max_bytes = st.number_input("Eval max bytes", min_value=1, value=20_000, step=1_000)
                grundgesetz_start_byte = st.number_input("Grundgesetz start byte", min_value=0, value=0, step=1_000)
                text8_start_byte = st.number_input("text8 start byte", min_value=0, value=1_000_000, step=1_000)
                force_train = st.checkbox("Force retrain", value=False)
                submitted = st.form_submit_button("Launch fair ablation")
            if submitted:
                entry = launch_fair_run(
                    config_path=config_path,
                    outdir=outdir,
                    train_texts=train_texts,
                    max_bytes_per_text=int(max_bytes_per_text),
                    seeds_text=seeds_text,
                    weights_text=weights_text,
                    max_bytes=int(max_bytes),
                    grundgesetz_start_byte=int(grundgesetz_start_byte),
                    text8_start_byte=int(text8_start_byte),
                    force_train=force_train,
                )
                st.success(f"Launched PID {entry['pid']} for fair ablation")
                st.code(" ".join(entry["command"]), language="bash")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="ui-card ui-card-strong">', unsafe_allow_html=True)
            st.markdown('<div class="ui-section-title">Text benchmark</div>', unsafe_allow_html=True)
            with st.form("launch_text_form", clear_on_submit=False):
                text_path = st.text_input("Test text", value="data/grundgesetz.txt")
                config_path = st.text_input("Config ", value="config/compare_text_real_followup_adapt.yaml", key="text_config")
                checkpoint = st.text_input("Checkpoint", value="results/fair_followup_adapt/train/best_model.pt")
                outdir = st.text_input("Output dir ", value=f"results/ui/text_{now_stamp()}", key="text_outdir")
                train_texts = st.text_input("Train texts ", value="data/grundgesetz.txt,data/text8.txt", key="text_train_texts")
                seed = st.number_input("Seed", min_value=0, value=1, step=1)
                diff_prior_weight = st.number_input("Diff prior weight", min_value=0.0, value=0.55, step=0.05, format="%.2f")
                mmse_prior_weight = st.number_input("MMSE prior weight", min_value=0.0, value=0.55, step=0.05, format="%.2f")
                max_bytes = st.number_input("Max bytes", min_value=0, value=20_000, step=1_000)
                start_byte = st.number_input("Start byte", min_value=0, value=0, step=1_000)
                submitted = st.form_submit_button("Launch text benchmark")
            if submitted:
                entry = launch_text_benchmark(
                    config_path=config_path,
                    text_path=text_path,
                    train_texts=train_texts,
                    checkpoint=checkpoint,
                    outdir=outdir,
                    seed=int(seed),
                    diff_prior_weight=float(diff_prior_weight),
                    mmse_prior_weight=float(mmse_prior_weight),
                    max_bytes=int(max_bytes),
                    start_byte=int(start_byte),
                )
                st.success(f"Launched PID {entry['pid']} for text benchmark")
                st.code(" ".join(entry["command"]), language="bash")
            st.markdown("</div>", unsafe_allow_html=True)

    with live_tab:
        registry = load_registry()
        st.markdown('<div class="ui-card ui-card-strong">', unsafe_allow_html=True)
        st.markdown('<div class="ui-section-title">Tracked runs</div>', unsafe_allow_html=True)
        if not registry:
            st.info("No tracked runs yet. Use the Launch tab to start one.")
        else:
            for entry in sorted(registry, key=lambda e: e.get("launched_at", ""), reverse=True):
                pid = entry.get("pid")
                run_dir = Path(entry.get("outdir", ""))
                active = is_running(pid)
                expected = int(entry.get("expected_total") or 0)
                completed = 0
                if run_dir.exists():
                    if entry.get("kind") == "fair_ablation":
                        completed = len(list((run_dir / "bench").rglob("text_metrics.csv"))) if (run_dir / "bench").exists() else 0
                    elif entry.get("kind") == "text_benchmark":
                        completed = 1 if (run_dir / "text_metrics.csv").exists() else 0
                progress = (completed / expected) if expected else (1.0 if completed else 0.0)
                epoch_hint, loss_hint = parse_training_progress(run_dir / "ui_launch.log")
                label = f"{entry.get('title', entry.get('kind', 'run'))} | {short_path(run_dir)}"
                with st.expander(label, expanded=active):
                    chips = [
                        ("kind", entry.get("kind", "run")),
                        ("pid", str(pid) if pid else "n/a"),
                        ("status", "running" if active else "done"),
                        ("completed", f"{completed}/{expected if expected else 1}"),
                        ("launched", entry.get("launched_at", "n/a")),
                    ]
                    show_metric_cards(chips)
                    st.progress(min(max(progress, 0.0), 1.0))
                    if epoch_hint or loss_hint:
                        st.caption(f"Training progress: {epoch_hint or 'n/a'} | {loss_hint or ''}")
                    if run_dir.exists():
                        if (run_dir / "summary_agg.csv").exists():
                            rows = csv_rows(run_dir / "summary_agg.csv")
                            st.markdown(html_table(rows, ["corpus", "prior_weight", "n_runs", "mean_mmse_ber", "mean_mmse_prior_ber", "mean_diff_ber", "mean_delta_diff_minus_mmse_prior"], max_rows=20), unsafe_allow_html=True)
                        if (run_dir / "text_metrics.csv").exists():
                            rows = csv_rows(run_dir / "text_metrics.csv")
                            st.markdown(html_table(rows, ["snr_db", "mmse_ber", "mmse_prior_ber", "diff_ber"], max_rows=20), unsafe_allow_html=True)
                    log_path = run_dir / "ui_launch.log"
                    if log_path.exists():
                        st.markdown('<div class="ui-log">', unsafe_allow_html=True)
                        st.code(tail_text(log_path, 40), language="text")
                        st.markdown("</div>", unsafe_allow_html=True)
                    cmd = entry.get("command", [])
                    if cmd:
                        st.code(" ".join(cmd), language="bash")
        st.markdown("</div>", unsafe_allow_html=True)

    with results_tab:
        st.markdown('<div class="ui-card ui-card-strong">', unsafe_allow_html=True)
        st.markdown('<div class="ui-section-title">Results browser</div>', unsafe_allow_html=True)
        options = collect_candidate_run_dirs()
        default_idx = 0
        if registry:
            latest = Path(sorted(registry, key=lambda e: e.get("launched_at", ""), reverse=True)[0].get("outdir", ""))
            for idx, option in enumerate(options):
                if option == latest:
                    default_idx = idx
                    break
        if options:
            selected = st.selectbox(
                "Run directory",
                options=options,
                index=default_idx,
                format_func=lambda p: short_path(p),
            )
            render_run_summary(selected)
        else:
            st.info("No result directories found yet.")
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
