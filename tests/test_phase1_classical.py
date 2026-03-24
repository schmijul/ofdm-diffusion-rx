import copy

import torch

from src.classical_receiver import run_classical_frame, run_receiver_on_frame, simulate_received_frame
from src.demapper import bits_to_qam16, qam16_to_bits
from src.estimation import interpolate_channel, ls_channel_estimate
from src.ofdm import get_pilot_indices
from src.utils import load_config, set_seed


def test_qam16_roundtrip_bits():
    set_seed(0)
    bits = torch.randint(0, 2, (4000,), dtype=torch.long)
    symbols = bits_to_qam16(bits)
    decoded = qam16_to_bits(symbols)
    assert torch.equal(bits, decoded)


def test_awgn_high_snr_near_zero_ber():
    set_seed(1)
    cfg = load_config()
    cfg = copy.deepcopy(cfg)
    cfg["channel"]["n_taps"] = 1
    cfg["ofdm"]["n_ofdm_symbols"] = 10

    out = run_classical_frame(cfg, snr_db=30.0, method="perfect_mmse", perfect_csi=True)
    assert out["ber"] < 1e-3


def test_ls_mse_scales_with_noise_power():
    set_seed(2)
    n_sc = 64
    n_p = 8
    n_sym = 4
    pilot_idx = get_pilot_indices(n_sc, n_p)

    h_true = torch.ones((n_sym, n_sc), dtype=torch.complex64)
    y_clean = h_true.clone()

    mse_values = []
    for sigma in [0.02, 0.1]:
        noise = sigma * (torch.randn_like(y_clean.real) + 1j * torch.randn_like(y_clean.imag))
        y_noisy = y_clean + noise.to(torch.complex64)
        h_p = ls_channel_estimate(y_noisy, pilot_idx)
        h_hat = interpolate_channel(h_p, pilot_idx, n_sc)
        mse = (h_hat - h_true).abs().pow(2).mean().item()
        mse_values.append(mse)

    assert mse_values[1] > mse_values[0]


def test_baseline_ordering_average_ber():
    set_seed(3)
    cfg = load_config()
    cfg = copy.deepcopy(cfg)
    cfg["ofdm"]["n_ofdm_symbols"] = 8

    ber = {"ls_zf": [], "ls_mmse": [], "perfect_mmse": []}
    for _ in range(10):
        ber["ls_zf"].append(run_classical_frame(cfg, snr_db=10.0, method="ls_zf", perfect_csi=False)["ber"])
        ber["ls_mmse"].append(run_classical_frame(cfg, snr_db=10.0, method="ls_mmse", perfect_csi=False)["ber"])
        ber["perfect_mmse"].append(run_classical_frame(cfg, snr_db=10.0, method="perfect_mmse", perfect_csi=True)["ber"])

    zf_mean = sum(ber["ls_zf"]) / len(ber["ls_zf"])
    mmse_mean = sum(ber["ls_mmse"]) / len(ber["ls_mmse"])
    genie_mean = sum(ber["perfect_mmse"]) / len(ber["perfect_mmse"])

    assert genie_mean <= mmse_mean
    # LS interpolation errors can make MMSE vs ZF close; allow small tolerance.
    assert mmse_mean <= zf_mean + 0.06


def test_methods_share_bits_on_same_frame():
    set_seed(5)
    cfg = load_config()
    cfg = copy.deepcopy(cfg)
    cfg["ofdm"]["n_ofdm_symbols"] = 4

    frame = simulate_received_frame(cfg, snr_db=8.0)
    out_zf = run_receiver_on_frame(frame, method="ls_zf", perfect_csi=False)
    out_mmse = run_receiver_on_frame(frame, method="ls_mmse", perfect_csi=False)
    out_genie = run_receiver_on_frame(frame, method="perfect_mmse", perfect_csi=True)

    assert torch.equal(out_zf["bits_tx"], out_mmse["bits_tx"])
    assert torch.equal(out_zf["bits_tx"], out_genie["bits_tx"])


def test_cyclic_interpolation_handles_band_edge():
    n_sc = 8
    pilot_idx = torch.tensor([0, 4], dtype=torch.long)
    h_p = torch.tensor([[0.0 + 0.0j, 4.0 + 0.0j]], dtype=torch.complex64)
    h_full = interpolate_channel(h_p, pilot_idx, n_sc)

    # Between k=4 and k=0 (wrapped via k=8), interpolation should decrease linearly: k=6 -> 2.
    assert torch.isclose(h_full[0, 6].real, torch.tensor(2.0), atol=1e-4)
