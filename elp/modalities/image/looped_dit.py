"""Curvature-warped schedule, adaptive loop depth, residual reuse, and AB2 extrapolation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class LoopedDiTScheduleConfig:
    """Configuration for adaptive loop reallocation on pretrained looped DiTs."""

    step_count: int = 14
    snr_curvature: float = 1.38
    refresh_steps: tuple[int, ...] = (0, 3, 6, 9, 12)
    low_t_threshold: float = 0.25
    full_loops: int = 4
    reduced_loops: int = 2
    extrap_alpha: float = 0.20
    extrap_t_min: float = 0.15
    extrap_t_max: float = 0.90
    reuse_max_age: int = 2
    reuse_weight: float = 1.0
    ab2_beta: float = 0.32


def warped_snr_schedule(steps: int, curvature: float = 1.38) -> list[tuple[float, float]]:
    """Return `steps` pairs of `(t_i, dt_i)` over [0, 1] warped by `u ** curvature`."""
    if steps < 1 or curvature <= 0:
        raise ValueError("steps must be >= 1 and curvature > 0")
    grid = [(i / steps) ** curvature for i in range(steps + 1)]
    return [(grid[i], grid[i + 1] - grid[i]) for i in range(steps)]


class AdaptiveLoopedDiTSampler:
    """Sample with adaptive K_t in {4, 2}, dual-stream extrapolation, bounded delta reuse, and AB2."""

    def __init__(self, config: LoopedDiTScheduleConfig | None = None) -> None:
        self.config = config or LoopedDiTScheduleConfig()

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        txt_init: Tensor,
        *,
        pre_blocks: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
        core_step: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
        post_blocks: Callable[[Tensor, Tensor], Tensor],
    ) -> tuple[Tensor, dict[str, int]]:
        """Run a full adaptive trajectory with frozen weights, returning latent and pass count."""
        cfg = self.config
        schedule = warped_snr_schedule(cfg.step_count, cfg.snr_curvature)
        refresh_set = set(cfg.refresh_steps)

        x = x_init
        prev_pred: Tensor | None = None
        res_cache: tuple[Tensor, Tensor] | None = None
        cache_step: int = -10**6
        total_core_passes = 0

        for step_idx, (t0, dt) in enumerate(schedule):
            n_loops = (
                cfg.full_loops
                if (step_idx in refresh_set or t0 < cfg.low_t_threshold)
                else cfg.reduced_loops
            )
            img, txt = pre_blocks(x, txt_init)
            hist_img, hist_txt = [img], [txt]
            prev_img, prev_txt = img, txt

            for _ in range(n_loops):
                prev_img, prev_txt = img, txt
                img, txt = core_step(img, txt)
                hist_img.append(img)
                hist_txt.append(txt)
                total_core_passes += 1

            if n_loops >= cfg.full_loops:
                if cfg.extrap_t_min <= t0 < cfg.extrap_t_max and cfg.extrap_alpha != 0.0:
                    img = img + cfg.extrap_alpha * (img - prev_img)
                    txt = txt + cfg.extrap_alpha * (txt - prev_txt)
                anchor_idx = min(cfg.reduced_loops, len(hist_img) - 1)
                res_cache = ((img - hist_img[anchor_idx]).detach(), (txt - hist_txt[anchor_idx]).detach())
                cache_step = step_idx
            elif res_cache is not None and (step_idx - cache_step) <= cfg.reuse_max_age:
                d_img, d_txt = res_cache
                img = img + cfg.reuse_weight * d_img
                txt = txt + cfg.reuse_weight * d_txt

            pred = post_blocks(img, txt)
            if prev_pred is not None and cfg.ab2_beta != 0.0:
                taper = max(0.0, 1.0 - t0)
                pred = pred + (cfg.ab2_beta * taper * dt) * (pred - prev_pred)
            prev_pred = pred

            denom = max(0.05, 1.0 - t0)
            v = (pred - x) / denom
            x = x + v * dt

        return x, {"core_passes": total_core_passes, "steps": len(schedule)}
