import torch

from src.demapper import bits_to_qam16, qam16_to_bits, qam16_to_bits_with_priors


def test_qam16_roundtrip_hard_demapper():
    bits = torch.tensor(
        [
            0, 0, 0, 0,
            0, 1, 1, 0,
            1, 0, 0, 1,
            1, 1, 1, 1,
        ],
        dtype=torch.long,
    )

    symbols = bits_to_qam16(bits)
    out = qam16_to_bits(symbols)

    assert torch.equal(out, bits)


def test_prior_demapper_matches_hard_demapper_when_disabled():
    bits = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1], dtype=torch.long)
    symbols = bits_to_qam16(bits)

    out = qam16_to_bits_with_priors(symbols, [0.2, 0.8, 0.7, 0.3], prior_weight=0.0)

    assert torch.equal(out, bits)


def test_prior_demapper_can_break_near_boundary_toward_likely_bits():
    symbols = torch.tensor([0.0 + 0.0j])

    out = qam16_to_bits_with_priors(symbols, [0.9, 0.9, 0.9, 0.9], prior_weight=1.0)

    assert torch.equal(out, torch.tensor([1, 1, 1, 1]))
