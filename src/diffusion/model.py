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
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 256,
        n_res_blocks: int = 6,
        time_dim: int = 128,
        context_dim: int = 0,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.context_dim = int(context_dim)

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_proj = nn.Sequential(nn.Linear(time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.context_proj = (
            nn.Sequential(nn.Linear(self.context_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
            if self.context_dim > 0
            else None
        )

        self.snr_to_gamma = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.snr_to_beta = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(n_res_blocks)])
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, input_dim))

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, snr_db: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.input_proj(x)
        if t.numel() > 1 and torch.all(t == t[0]):
            # Benchmark inference repeatedly uses a single timestep per batch.
            base_emb = sinusoidal_time_embedding(t[:1], self.time_dim)
            t_emb = base_emb.expand(t.shape[0], -1)
        else:
            t_emb = sinusoidal_time_embedding(t, self.time_dim)
        h = h + self.time_proj(t_emb)
        if self.context_proj is not None:
            if context is None:
                context = torch.zeros((x.shape[0], self.context_dim), device=x.device, dtype=x.dtype)
            h = h + self.context_proj(context)

        gamma = self.snr_to_gamma(snr_db)
        beta = self.snr_to_beta(snr_db)
        h = h * (1.0 + gamma) + beta

        for block in self.blocks:
            h = block(h)

        return self.out(h)


def _prepare_context(
    context: torch.Tensor | None,
    *,
    batch_size: int,
    context_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if context_dim <= 0:
        return None
    if context is None:
        return torch.zeros((batch_size, context_dim), device=device, dtype=dtype)
    if context.dim() == 1:
        context = context.unsqueeze(0)
    if context.dim() != 2:
        raise ValueError(f"Expected context rank 1 or 2, got shape {tuple(context.shape)}")
    if context.shape[1] != context_dim:
        raise ValueError(f"Expected context_dim={context_dim}, got shape {tuple(context.shape)}")
    if context.shape[0] == 1 and batch_size != 1:
        context = context.expand(batch_size, -1)
    elif context.shape[0] != batch_size:
        raise ValueError(f"Expected batch size {batch_size}, got context batch {context.shape[0]}")
    return context.to(device=device, dtype=dtype)


class FiLMResBlock(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mod = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.mod(cond).chunk(2, dim=-1)
        h = self.net(self.norm(x))
        return x + h * (1.0 + torch.tanh(scale)) + shift


class GatedResBlock(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.bias = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.net(self.norm(x))
        gate = torch.sigmoid(self.gate(cond))
        bias = self.bias(cond)
        return x + h * gate + bias


class _ConditionedDenoiserBase(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, time_dim: int, context_dim: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.context_dim = int(context_dim)

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_proj = nn.Sequential(nn.Linear(time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.snr_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.context_proj = (
            nn.Sequential(nn.Linear(self.context_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
            if self.context_dim > 0
            else None
        )

    def _condition(self, t: torch.Tensor, snr_db: torch.Tensor, context: torch.Tensor | None, batch_size: int) -> torch.Tensor:
        if t.numel() > 1 and torch.all(t == t[0]):
            base_emb = sinusoidal_time_embedding(t[:1], self.time_dim)
            t_emb = base_emb.expand(batch_size, -1)
        else:
            t_emb = sinusoidal_time_embedding(t, self.time_dim)
        cond = self.time_proj(t_emb) + self.snr_proj(snr_db)
        if self.context_proj is not None:
            ctx = _prepare_context(
                context,
                batch_size=batch_size,
                context_dim=self.context_dim,
                device=t.device,
                dtype=cond.dtype,
            )
            if ctx is None:
                ctx = torch.zeros((batch_size, self.context_dim), device=t.device, dtype=cond.dtype)
            cond = cond + self.context_proj(ctx)
        return cond


class FiLMResidualMLPDenoiser(_ConditionedDenoiserBase):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 256,
        n_res_blocks: int = 6,
        time_dim: int = 128,
        context_dim: int = 0,
    ):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, time_dim=time_dim, context_dim=context_dim)
        self.blocks = nn.ModuleList([FiLMResBlock(hidden_dim, hidden_dim) for _ in range(n_res_blocks)])
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, input_dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor, snr_db: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        h = self.input_proj(x)
        cond = self._condition(t, snr_db, context, x.shape[0])
        for block in self.blocks:
            h = block(h, cond)
        return self.out(h)


class GatedResidualMLPDenoiser(_ConditionedDenoiserBase):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 256,
        n_res_blocks: int = 6,
        time_dim: int = 128,
        context_dim: int = 0,
    ):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, time_dim=time_dim, context_dim=context_dim)
        self.blocks = nn.ModuleList([GatedResBlock(hidden_dim, hidden_dim) for _ in range(n_res_blocks)])
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, input_dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor, snr_db: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        h = self.input_proj(x)
        cond = self._condition(t, snr_db, context, x.shape[0])
        for block in self.blocks:
            h = block(h, cond)
        return self.out(h)


def build_prior_context_from_config(cfg: dict, device: torch.device) -> torch.Tensor | None:
    modulation_cfg = cfg.get("modulation", {})
    per_pos = modulation_cfg.get("bit_one_prob_per_position", None)
    if isinstance(per_pos, list) and len(per_pos) == 4:
        return torch.tensor(per_pos, device=device, dtype=torch.float32)
    p = modulation_cfg.get("bit_one_prob", None)
    if p is None:
        return None
    return torch.full((4,), float(p), device=device, dtype=torch.float32)


def build_denoiser_from_config(model_cfg: dict) -> nn.Module:
    model_type = str(model_cfg.get("type", "residual_mlp")).lower()
    hidden_dim = int(model_cfg["hidden_dim"])
    n_res_blocks = int(model_cfg["n_res_blocks"])
    time_dim = int(model_cfg["time_embedding_dim"])
    context_dim = int(model_cfg.get("context_dim", 0))

    if model_type in {"residual_mlp", "residual", "base"}:
        return ResidualMLPDenoiser(
            input_dim=2,
            hidden_dim=hidden_dim,
            n_res_blocks=n_res_blocks,
            time_dim=time_dim,
            context_dim=context_dim,
        )
    if model_type in {"film_residual_mlp", "film_mlp", "film"}:
        return FiLMResidualMLPDenoiser(
            input_dim=2,
            hidden_dim=hidden_dim,
            n_res_blocks=n_res_blocks,
            time_dim=time_dim,
            context_dim=context_dim,
        )
    if model_type in {"gated_residual_mlp", "gated_mlp", "gated"}:
        return GatedResidualMLPDenoiser(
            input_dim=2,
            hidden_dim=hidden_dim,
            n_res_blocks=n_res_blocks,
            time_dim=time_dim,
            context_dim=context_dim,
        )
    raise ValueError(f"Unknown diffusion.model.type: {model_type}")
