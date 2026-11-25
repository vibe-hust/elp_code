"""Inter-pass state update rules for frozen language models (Euler, RK, Anderson, Geodesic)."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import torch
from torch import Tensor

TEXT_STRATEGIES = (
    "naive", "euler", "anchored_euler", "midpoint", "heun", "rk4",
    "heavy_ball", "anderson", "norm_stab", "explicit_rk",
)


@dataclass(frozen=True)
class TextLoopConfig:
    """Configuration of one contiguous decoder layer window for looped LLM inference."""

    start_layer: int
    end_layer: int
    num_loops: int = 3
    strategy: str = "euler"
    step_size: float | None = None
    anchor_weight: float = 0.0
    momentum: float = 0.3
    anderson_history: int = 2
    anderson_beta: float = 0.5
    anderson_regularization: float = 0.0
    rk_a: tuple[tuple[float, ...], ...] = ()
    rk_b: tuple[float, ...] = ()
    norm_gamma: float = 0.0
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if not 0 <= self.start_layer <= self.end_layer:
            raise ValueError("window must satisfy 0 <= start_layer <= end_layer")
        if self.num_loops < 1:
            raise ValueError("num_loops must be >= 1")
        if self.strategy not in TEXT_STRATEGIES:
            raise ValueError(f"Unknown text loop strategy: {self.strategy!r}")
        if self.strategy == "explicit_rk":
            if len(self.rk_a) != self.num_loops or len(self.rk_b) != self.num_loops:
                raise ValueError("explicit_rk requires K rows in rk_a and K weights in rk_b")
            if any(len(row) != i for i, row in enumerate(self.rk_a)):
                raise ValueError("rk_a must be strictly lower-triangular")


def _fp32(x: Tensor) -> Tensor:
    return x if x.dtype == torch.float64 else x.float()


def explicit_runge_kutta(
    g: Callable[[Tensor], Tensor],
    x: Tensor,
    a: Sequence[Sequence[float]],
    b: Sequence[float],
    *,
    h: float = 1.0,
    norm_gamma: float = 0.0,
    epsilon: float = 1e-8,
) -> Tensor:
    """Execute one explicit Runge-Kutta step on F(u) = g(u) - u with optional RMS tapering."""
    if len(a) != len(b) or not b or any(len(row) != i for i, row in enumerate(a)):
        raise ValueError("Invalid strictly lower-triangular Butcher tableau")
    x0_norm = torch.linalg.vector_norm(_fp32(x), dim=-1, keepdim=True) if norm_gamma else None
    increments: list[Tensor] = []
    for row in a:
        stage = x
        for weight, inc in zip(row, increments):
            stage = stage + h * weight * inc
        increments.append(g(stage) - stage)
    out = x
    for weight, inc in zip(b, increments):
        out = out + h * weight * inc
    if norm_gamma and x0_norm is not None:
        out_fp = _fp32(out)
        out_norm = torch.linalg.vector_norm(out_fp, dim=-1, keepdim=True).clamp_min(epsilon)
        target_norm = (1.0 - norm_gamma) * out_norm + norm_gamma * x0_norm
        out = (out_fp * (target_norm / out_norm)).to(x.dtype)
    return out


def _anderson_step(xs: list[Tensor], fs: list[Tensor], y: Tensor, cfg: TextLoopConfig) -> Tensor:
    x, f = xs[-1], fs[-1]
    if len(xs) == 1:
        return x + cfg.anderson_beta * f
    dx = torch.stack([_fp32(v) - _fp32(u) for u, v in zip(xs, xs[1:])], dim=-1)
    df = torch.stack([_fp32(v) - _fp32(u) for u, v in zip(fs, fs[1:])], dim=-1)
    residual = _fp32(f).unsqueeze(-1)
    if cfg.anderson_regularization > 0:
        df_t = df.transpose(-1, -2)
        gram = df_t @ df
        eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
        gamma = torch.linalg.solve(gram + cfg.anderson_regularization * eye, df_t @ residual)
    else:
        gamma = torch.linalg.pinv(df, rtol=cfg.epsilon) @ residual
    dx_g = (dx @ gamma).squeeze(-1)
    df_g = (df @ gamma).squeeze(-1)
    beta = cfg.anderson_beta
    return ((1.0 - beta) * (_fp32(x) - dx_g) + beta * (_fp32(y) - df_g)).to(x.dtype)


def iterate_window(g: Callable[[Tensor], Tensor], x: Tensor, config: TextLoopConfig) -> Tensor:
    """Iterate decoder window `g` on hidden state `x` per the configured integration strategy."""
    if config.strategy == "explicit_rk":
        return explicit_runge_kutta(
            g, x, config.rk_a, config.rk_b, norm_gamma=config.norm_gamma, epsilon=config.epsilon
        )

    h = 1.0 / config.num_loops
    alpha = config.step_size if config.step_size is not None else h
    x0, prev, anchor = x, x, None
    xs: list[Tensor] = []
    fs: list[Tensor] = []

    for _ in range(config.num_loops):
        old = x
        y = g(x)
        if anchor is None:
            anchor = y
        f = y - x
        if config.strategy == "naive":
            x = y
        elif config.strategy in ("euler", "anchored_euler"):
            x = y if alpha == 1.0 else x + alpha * f
        elif config.strategy == "norm_stab":
            cand = y if alpha == 1.0 else x + alpha * f
            target = torch.linalg.vector_norm(_fp32(x0), dim=-1, keepdim=True)
            norm = torch.linalg.vector_norm(_fp32(cand), dim=-1, keepdim=True)
            scaled = (_fp32(cand) * target / norm.clamp_min(config.epsilon)).to(x.dtype)
            x = torch.where(norm > config.epsilon, scaled, x0)
        elif config.strategy == "heavy_ball":
            x = x + alpha * f + config.momentum * (x - prev)
        elif config.strategy == "anderson":
            xs.append(x)
            fs.append(f)
            keep = config.anderson_history + 1
            xs, fs = xs[-keep:], fs[-keep:]
            x = _anderson_step(xs, fs, y, config)
        elif config.strategy == "midpoint":
            probe = x + (h / 2.0) * f
            x = x + h * (g(probe) - probe)
        elif config.strategy == "heun":
            probe = x + h * f
            x = x + (h / 2.0) * (f + g(probe) - probe)
        elif config.strategy == "rk4":
            probe = x + (h / 2.0) * f
            k2 = g(probe) - probe
            probe = x + (h / 2.0) * k2
            k3 = g(probe) - probe
            probe = x + h * k3
            k4 = g(probe) - probe
            x = x + (h / 6.0) * (f + 2.0 * k2 + 2.0 * k3 + k4)
        prev = old

    if config.strategy == "anchored_euler" and anchor is not None and config.anchor_weight > 0.0:
        w = config.anchor_weight
        x = w * anchor + (1.0 - w) * x
    return x


def geodesic_extrapolate(
    h_prev: Tensor,
    h_curr: Tensor,
    scale: float,
    *,
    max_angle: float = 0.35,
    epsilon: float = 1e-8,
) -> Tensor:
    """Extrapolate hidden states along the per-token hyperspherical geodesic past `h_curr`."""
    if scale == 0.0:
        return h_curr
    p = _fp32(h_prev)
    c = _fp32(h_curr)
    p_norm = torch.linalg.vector_norm(p, dim=-1, keepdim=True)
    c_norm = torch.linalg.vector_norm(c, dim=-1, keepdim=True)
    u0 = p / p_norm.clamp_min(epsilon)
    u1 = c / c_norm.clamp_min(epsilon)

    cos_t = (u0 * u1).sum(dim=-1, keepdim=True).clamp(-1.0 + epsilon, 1.0 - epsilon)
    theta = torch.acos(cos_t)
    sin_t = torch.sqrt((1.0 - cos_t.square()).clamp_min(epsilon))

    delta = (float(scale) * theta).clamp(-max_angle, max_angle)
    tangent = (cos_t * u1 - u0) / sin_t
    extrap_dir = torch.cos(delta) * u1 + torch.sin(delta) * tangent
    result = c_norm * extrap_dir
    valid = (theta > 1e-5) & (p_norm > epsilon) & (c_norm > epsilon)
    return torch.where(valid, result, c).to(h_curr.dtype)
