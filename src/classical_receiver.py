import torch

from src.channel import (
    add_awgn,
    apply_multipath,
    channel_frequency_response,
    generate_rayleigh_taps,
    generate_tdl_a_taps,
)
from src.demapper import bits_to_qam16, qam16_to_bits
from src.equalization import mmse_equalize, zf_equalize
from src.estimation import interpolate_channel, ls_channel_estimate
from src.metrics import bit_error_rate
from src.ofdm import (
    build_ofdm_grid,
    get_data_indices,
    get_pilot_indices,
    ofdm_demodulate,
    ofdm_modulate,
)


def _get_device(cfg: dict) -> torch.device:
    use_cuda = torch.cuda.is_available() and cfg.get("device", {}).get("prefer_cuda", True)
    return torch.device("cuda" if use_cuda else "cpu")


def simulate_received_frame(cfg: dict, snr_db: float) -> dict:
    n_sc = cfg["ofdm"]["n_subcarriers"]
    cp = cfg["ofdm"]["cp_length"]
    n_sym = cfg["ofdm"]["n_ofdm_symbols"]
    n_p = cfg["ofdm"]["n_pilot_subcarriers"]
    exp_decay = cfg["channel"].get("exp_decay", 0.7)
    n_taps = cfg["channel"]["n_taps"]
    channel_model = str(cfg.get("channel", {}).get("model", "rayleigh")).lower()

    device = _get_device(cfg)

    pilot_idx = get_pilot_indices(n_sc, n_p).to(device)
    data_idx = get_data_indices(n_sc, pilot_idx)
    n_data = data_idx.numel() * n_sym

    bits_tx = torch.randint(0, 2, (n_data * 4,), device=device)
    data_symbols = bits_to_qam16(bits_tx)

    tx_grid = build_ofdm_grid(data_symbols, n_sc, n_sym, pilot_idx)
    tx_time = ofdm_modulate(tx_grid, cp)

    if channel_model == "tdl_a":
        taps = generate_tdl_a_taps(n_taps, device)
    else:
        taps = generate_rayleigh_taps(n_taps, exp_decay, device)
    rx_time = apply_multipath(tx_time, taps)
    rx_time, noise_power = add_awgn(rx_time, snr_db)

    rx_grid = ofdm_demodulate(rx_time, n_sc, cp)
    h_true = channel_frequency_response(taps, n_sc).unsqueeze(0).repeat(n_sym, 1)

    return {
        "cfg": cfg,
        "snr_db": snr_db,
        "pilot_idx": pilot_idx,
        "data_idx": data_idx,
        "bits_tx": bits_tx,
        "tx_symbols": data_symbols,
        "rx_grid": rx_grid,
        "h_true": h_true,
        "noise_power": noise_power,
    }


def run_receiver_on_frame(frame: dict, method: str = "ls_mmse", perfect_csi: bool = False) -> dict:
    rx_grid = frame["rx_grid"]
    h_true = frame["h_true"]
    pilot_idx = frame["pilot_idx"]
    data_idx = frame["data_idx"]
    bits_tx = frame["bits_tx"]
    tx_symbols = frame["tx_symbols"]
    noise_power = frame["noise_power"]

    if perfect_csi:
        h_hat = h_true
    else:
        h_p = ls_channel_estimate(rx_grid, pilot_idx)
        h_hat = interpolate_channel(h_p, pilot_idx, rx_grid.shape[-1])

    if method == "ls_zf":
        x_hat_grid = zf_equalize(rx_grid, h_hat)
    elif method == "ls_mmse" or method == "perfect_mmse":
        x_hat_grid = mmse_equalize(rx_grid, h_hat, noise_power)
    else:
        raise ValueError(f"Unknown method: {method}")

    rx_data = rx_grid[:, data_idx].reshape(-1)
    x_hat_data = x_hat_grid[:, data_idx].reshape(-1)
    bits_rx = qam16_to_bits(x_hat_data)
    ber = bit_error_rate(bits_tx.long().cpu(), bits_rx.long().cpu())

    return {
        "bits_tx": bits_tx.detach().cpu(),
        "bits_rx": bits_rx.detach().cpu(),
        "tx_symbols": tx_symbols.detach().cpu(),
        "rx_symbols": rx_data.detach().cpu(),
        "equalized_symbols": x_hat_data.detach().cpu(),
        "ber": ber,
        "h_true": h_true.detach().cpu(),
        "h_hat": h_hat.detach().cpu(),
        "noise_power": noise_power,
        "snr_db": frame["snr_db"],
    }


def run_classical_frame(cfg: dict, snr_db: float, method: str = "ls_mmse", perfect_csi: bool = False) -> dict:
    frame = simulate_received_frame(cfg, snr_db)
    return run_receiver_on_frame(frame, method=method, perfect_csi=perfect_csi)
