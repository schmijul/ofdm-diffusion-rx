import math

import torch
import torch.nn as nn

_FREQ_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def _sinusoidal_freqs(half_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (half_dim, str(device), dtype)
    cached = _FREQ_CACHE.get(key)
    if cached is not None and cached.device == device:
        return cached
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half_dim, device=device, dtype=dtype) / max(half_dim - 1, 1)
    )
    _FREQ_CACHE[key] = freqs
    return freqs


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    device = timesteps.device
    dtype = torch.float32
    freqs = _sinusoidal_freqs(half, device, dtype)
    args = timesteps.to(dtype=dtype).unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class ResBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class ResidualMLPDenoiser(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 256, n_res_blocks: int = 6, time_dim: int = 128):
        super().__init__()
        self.time_dim = time_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_proj = nn.Sequential(nn.Linear(time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

        self.snr_to_gamma = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.snr_to_beta = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(n_res_blocks)])
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, input_dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        if t.numel() > 1 and torch.all(t == t[0]):
            # Benchmark inference repeatedly uses a single timestep per batch.
            base_emb = sinusoidal_time_embedding(t[:1], self.time_dim)
            t_emb = base_emb.expand(t.shape[0], -1)
        else:
            t_emb = sinusoidal_time_embedding(t, self.time_dim)
        h = h + self.time_proj(t_emb)

        gamma = self.snr_to_gamma(snr_db)
        beta = self.snr_to_beta(snr_db)
        h = h * (1.0 + gamma) + beta

        for block in self.blocks:
            h = block(h)

        return self.out(h)
