import torch

from src.diffusion.ddpm import DDPM
from src.diffusion.model import ResidualMLPDenoiser
from src.diffusion.noise_schedule import NoiseSchedule


def test_noise_schedule_monotonic_alpha_bar():
    sched = NoiseSchedule(n_timesteps=64, beta_start=1e-4, beta_end=0.02, schedule="linear")
    diff = sched.alpha_bars[1:] - sched.alpha_bars[:-1]
    assert torch.all(diff <= 1e-7)


def test_ddpm_shapes_and_sampling():
    device = torch.device("cpu")
    model = ResidualMLPDenoiser(input_dim=2, hidden_dim=64, n_res_blocks=2, time_dim=32).to(device)
    sched = NoiseSchedule(n_timesteps=32, beta_start=1e-4, beta_end=0.02, schedule="linear")
    ddpm = DDPM(model, sched, device)

    x0 = torch.randn(16, 2)
    snr = torch.full((16, 1), 10.0)

    loss = ddpm.p_losses(x0, snr)
    assert loss.ndim == 0

    x_dn = ddpm.denoise_from_equalized(x0, snr)
    assert x_dn.shape == x0.shape
