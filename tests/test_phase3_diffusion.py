import torch

from src.diffusion.ddpm import DDPM
from src.diffusion.model import ResidualMLPDenoiser, build_denoiser_from_config
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


def test_denoiser_factory_supports_variants():
    variants = [
        {"type": "residual_mlp", "hidden_dim": 32, "n_res_blocks": 2, "time_embedding_dim": 16, "context_dim": 4},
        {"type": "film_residual_mlp", "hidden_dim": 32, "n_res_blocks": 2, "time_embedding_dim": 16, "context_dim": 4},
        {"type": "gated_residual_mlp", "hidden_dim": 32, "n_res_blocks": 2, "time_embedding_dim": 16, "context_dim": 4},
    ]
    device = torch.device("cpu")
    x = torch.randn(8, 2)
    t = torch.randint(0, 10, (8,))
    snr = torch.full((8, 1), 10.0)
    context = torch.full((8, 4), 0.5)

    for cfg in variants:
        model = build_denoiser_from_config(cfg).to(device)
        out = model(x, t, snr, context)
        assert out.shape == x.shape


def test_residual_variance_estimator_positive():
    x_eq = torch.tensor(
        [
            [0.95 / (10.0**0.5), 1.02 / (10.0**0.5)],
            [3.3 / (10.0**0.5), -2.7 / (10.0**0.5)],
            [-1.2 / (10.0**0.5), -0.8 / (10.0**0.5)],
        ],
        dtype=torch.float32,
    )
    sigma2 = DDPM.estimate_residual_variance(x_eq)
    assert sigma2.shape == (3,)
    assert torch.all(sigma2 > 0.0)
