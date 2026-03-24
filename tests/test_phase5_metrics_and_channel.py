import copy

import torch

from src.classical_receiver import run_classical_frame
from src.diffusion.ddpm import DDPM
from src.diffusion.model import ResidualMLPDenoiser
from src.diffusion.noise_schedule import NoiseSchedule
from src.utils import load_config, set_seed


def test_tdl_a_channel_mode_runs():
    set_seed(6)
    cfg = copy.deepcopy(load_config())
    cfg["channel"]["model"] = "tdl_a"
    cfg["channel"]["n_taps"] = 6
    out = run_classical_frame(cfg, snr_db=10.0, method="ls_mmse", perfect_csi=False)
    assert 0.0 <= out["ber"] <= 1.0


def test_ddpm_respects_inference_steps_shape():
    device = torch.device("cpu")
    model = ResidualMLPDenoiser(input_dim=2, hidden_dim=64, n_res_blocks=2, time_dim=32).to(device)
    sched = NoiseSchedule(n_timesteps=32, beta_start=1e-4, beta_end=0.02, schedule="linear")
    ddpm = DDPM(model, sched, device, inference_steps=4)

    x_eq = torch.randn(10, 2)
    snr = torch.full((10, 1), 12.0)
    out = ddpm.denoise_from_equalized(x_eq, snr)
    assert out.shape == x_eq.shape
