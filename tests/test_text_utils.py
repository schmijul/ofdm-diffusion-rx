from pathlib import Path

import pytest

from src.text_utils import (
    assert_disjoint_test_file,
    bits_to_bytes,
    bytes_to_bits,
    byte_error_rate,
    char_mismatch_rate,
    estimate_qam16_bit_priors_from_bytes,
    estimate_qam16_bit_priors_from_text_files,
)


def test_bits_bytes_roundtrip():
    data = b"Hello OFDM"
    bits = bytes_to_bits(data)
    out = bits_to_bytes(bits)
    assert out == data


def test_disjoint_hash_guard(tmp_path: Path):
    train = tmp_path / "train.txt"
    test = tmp_path / "test.txt"
    train.write_text("abc", encoding="utf-8")
    test.write_text("abc", encoding="utf-8")

    with pytest.raises(ValueError):
        assert_disjoint_test_file(test, [train])


def test_error_metrics_in_range():
    ref = b"abcdef"
    hyp = b"abcxef"
    ber = byte_error_rate(ref, hyp)
    cer = char_mismatch_rate("abcdef", "abcxef")
    assert 0.0 < ber < 1.0
    assert 0.0 < cer < 1.0


def test_estimate_qam16_bit_priors_from_bytes_shape():
    p_global, p_pos = estimate_qam16_bit_priors_from_bytes(b"ABCD")
    assert 0.0 < p_global < 1.0
    assert len(p_pos) == 4
    assert all(0.0 <= p <= 1.0 for p in p_pos)


def test_estimate_qam16_bit_priors_from_text_files(tmp_path: Path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("AAAA", encoding="utf-8")
    b.write_text("BBBB", encoding="utf-8")

    p_global, p_pos, n_bytes = estimate_qam16_bit_priors_from_text_files([a, b])
    assert n_bytes == 8
    assert 0.0 < p_global < 1.0
    assert len(p_pos) == 4
