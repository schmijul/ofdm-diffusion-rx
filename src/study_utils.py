import csv
from pathlib import Path


def parse_int_list(spec: str) -> list[int]:
    values = [item.strip() for item in spec.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return [int(item) for item in values]


def parse_float_list(spec: str) -> list[float]:
    values = [item.strip() for item in spec.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value")
    return [float(item) for item in values]


def prior_slug(bit_one_prob: float) -> str:
    return f"p{int(round(bit_one_prob * 100)):03d}"


def validate_bit_prior(bit_one_prob: float) -> float:
    value = float(bit_one_prob)
    if not (0.0 < value < 1.0):
        raise ValueError(f"bit_one_prob must be in (0,1), got {value}")
    return value


def normalize_unique_bit_priors(priors: list[float]) -> list[float]:
    validated = [validate_bit_prior(v) for v in priors]
    return sorted(set(validated))


def load_csv_rows(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _extract_delta(row: dict) -> float:
    if "diffusion_gain_vs_mmse_mean" in row and row["diffusion_gain_vs_mmse_mean"] != "":
        return float(row["diffusion_gain_vs_mmse_mean"])
    if "delta_diff_minus_mmse_mean" in row and row["delta_diff_minus_mmse_mean"] != "":
        return float(row["delta_diff_minus_mmse_mean"])
    raise KeyError("Expected either diffusion_gain_vs_mmse_mean or delta_diff_minus_mmse_mean in CSV row")


def summarize_delta_curve(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Expected at least one benchmark row")

    deltas = [_extract_delta(row) for row in rows]
    snrs = [float(row["snr_db"]) for row in rows]
    return {
        "snr_min_db": min(snrs),
        "snr_max_db": max(snrs),
        "avg_delta": sum(deltas) / len(deltas),
        "best_delta": min(deltas),
        "worst_delta": max(deltas),
        "n_snrs": len(deltas),
    }


def linear_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have identical length")
    if len(xs) < 2:
        raise ValueError("Need at least 2 points to fit a slope")

    n = float(len(xs))
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sum_xx - sum_x * sum_x
    if denom == 0.0:
        raise ValueError("Cannot fit slope: degenerate x values")
    return (n * sum_xy - sum_x * sum_y) / denom
