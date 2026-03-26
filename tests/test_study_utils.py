import pytest

from src.study_utils import load_csv_rows, parse_float_list, prior_slug, summarize_delta_curve


def test_parse_float_list_parses_csv_values():
    assert parse_float_list("0.1, 0.2,0.5") == [0.1, 0.2, 0.5]


def test_parse_float_list_rejects_empty_input():
    with pytest.raises(ValueError):
        parse_float_list(" , ")


def test_prior_slug_is_stable():
    assert prior_slug(0.1) == "p010"
    assert prior_slug(0.5) == "p050"


def test_summarize_delta_curve_extracts_key_stats(tmp_path):
    csv_path = tmp_path / "benchmark_summary.csv"
    csv_path.write_text(
        "snr_db,delta_diff_minus_mmse_mean\n0,-0.05\n4,-0.03\n8,0.01\n",
        encoding="utf-8",
    )

    rows = load_csv_rows(csv_path)
    summary = summarize_delta_curve(rows)

    assert summary["snr_min_db"] == 0.0
    assert summary["snr_max_db"] == 8.0
    assert abs(summary["avg_delta"] - (-0.07 / 3.0)) < 1e-9
    assert summary["best_delta"] == -0.05
    assert summary["worst_delta"] == 0.01
    assert summary["n_snrs"] == 3
