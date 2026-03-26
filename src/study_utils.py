import csv
from pathlib import Path


def parse_float_list(spec: str) -> list[float]:
    values = [item.strip() for item in spec.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value")
    return [float(item) for item in values]


def prior_slug(bit_one_prob: float) -> str:
    return f"p{int(round(bit_one_prob * 100)):03d}"


def load_csv_rows(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_delta_curve(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Expected at least one benchmark row")

    deltas = [float(row["delta_diff_minus_mmse_mean"]) for row in rows]
    snrs = [float(row["snr_db"]) for row in rows]
    return {
        "snr_min_db": min(snrs),
        "snr_max_db": max(snrs),
        "avg_delta": sum(deltas) / len(deltas),
        "best_delta": min(deltas),
        "worst_delta": max(deltas),
        "n_snrs": len(deltas),
    }
