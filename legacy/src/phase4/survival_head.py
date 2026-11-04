"""Weibull survival head for Remaining-Useful-Life (RUL) prediction.

Reframes the prognostics task as time-to-event (cycles until SOH crosses
the End-of-Life threshold, default 80%) instead of point-in-time SOH
regression. Returns a full Weibull distribution rather than a scalar:

    P(RUL <= t | x) = 1 - exp(-(t / scale)^shape)

with scale (lambda) > 0 and shape (k) > 0 predicted by the network.
This naturally yields:

    - Median RUL = scale * (ln 2)^(1/shape)
    - Quantiles at any confidence level
    - A proper likelihood objective with right-censoring support
      (cycles where the battery hasn't yet reached EoL count as "censored")

Right-censoring matters here: for ongoing batteries we know RUL > T_observed
but not the exact value, and ignoring this biases the model toward
under-estimating life.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


EOL_THRESHOLD_DEFAULT = 80.0  # SOH (%) at which a battery is considered end-of-life


def compute_rul_targets(
    df: pd.DataFrame,
    eol_threshold: float = EOL_THRESHOLD_DEFAULT,
    soh_col: str = "soh",
    cycle_col: str = "cycle_number",
    battery_col: str = "battery_id",
) -> pd.DataFrame:
    """Add RUL ('rul') and event indicator ('event') columns to a feature frame.

    For each battery, find the first cycle where SOH crosses below
    eol_threshold. RUL at any earlier cycle is (eol_cycle - current_cycle).
    Cycles after EoL get RUL=0. If a battery never reaches EoL within the
    recording, all its cycles are censored and assigned RUL = (last_cycle -
    current_cycle) with event=0; the model should treat these as lower
    bounds via the survival likelihood.
    """
    out_frames = []
    for bat_id, group in df.groupby(battery_col):
        g = group.sort_values(cycle_col).copy()
        below = g[g[soh_col] < eol_threshold]
        if len(below):
            eol_cycle = int(below[cycle_col].iloc[0])
            event = (g[cycle_col] <= eol_cycle).astype(int)
            rul = (eol_cycle - g[cycle_col]).clip(lower=0)
        else:
            last_cycle = int(g[cycle_col].iloc[-1])
            event = pd.Series(0, index=g.index)
            rul = (last_cycle - g[cycle_col]).clip(lower=0)
        g = g.assign(rul=rul.astype(float).values, event=event.astype(int).values)
        out_frames.append(g)
    return pd.concat(out_frames, ignore_index=True)


class WeibullHead(nn.Module):
    """Two-output head producing positive Weibull (scale, shape) parameters.

    Both outputs use softplus to keep them strictly positive, with small
    epsilons that empirically stabilise the negative-log-likelihood. Shape
    is additionally clamped to a sensible range (avoids degenerate gradients
    when the network briefly predicts shape near zero or above 50 early in
    training).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        scale_init: float = 100.0,
    ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.scale_head = nn.Linear(hidden_dim, 1)
        self.shape_head = nn.Linear(hidden_dim, 1)
        # Bias the scale output so initial predictions sit near scale_init,
        # which (with softplus) drastically improves early-training stability.
        with torch.no_grad():
            self.scale_head.bias.fill_(math.log(math.exp(scale_init) - 1.0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        scale = F.softplus(self.scale_head(h)).squeeze(-1) + 1e-3
        shape = F.softplus(self.shape_head(h)).squeeze(-1).clamp(min=1e-2, max=50.0) + 1e-3
        return scale, shape


def weibull_nll(
    scale: torch.Tensor,
    shape: torch.Tensor,
    rul: torch.Tensor,
    event: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Right-censoring-aware Weibull negative log-likelihood.

    For uncensored observations (event == 1):
        log f(t) = log(k/lambda) + (k-1) log(t/lambda) - (t/lambda)^k
    For censored observations (event == 0):
        log S(t) = -(t/lambda)^k

    Aggregates as the mean per-sample NLL.
    """
    t = rul.clamp(min=eps)
    lam = scale.clamp(min=eps)
    k = shape.clamp(min=eps)
    log_t_over_lam = torch.log(t) - torch.log(lam)
    hazard_term = torch.exp(k * log_t_over_lam)  # (t/lambda)^k
    log_pdf = torch.log(k) - torch.log(lam) + (k - 1.0) * log_t_over_lam - hazard_term
    log_surv = -hazard_term
    log_lik = event.float() * log_pdf + (1.0 - event.float()) * log_surv
    return -log_lik.mean()


def weibull_quantile(scale: torch.Tensor, shape: torch.Tensor, q: float) -> torch.Tensor:
    """Inverse CDF of Weibull at probability ``q``."""
    if not 0 < q < 1:
        raise ValueError("q must be in (0, 1)")
    return scale * (-math.log(1.0 - q)) ** (1.0 / shape)


def weibull_median(scale: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    return weibull_quantile(scale, shape, 0.5)


def weibull_mean(scale: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    """Mean of a Weibull(scale, shape): scale * Gamma(1 + 1/shape)."""
    one_over_k = 1.0 / shape
    return scale * torch.exp(torch.lgamma(1.0 + one_over_k))


@dataclass
class RULPrediction:
    median: np.ndarray
    p10: np.ndarray
    p90: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    shape: np.ndarray


def summarise_predictions(scale: torch.Tensor, shape: torch.Tensor) -> RULPrediction:
    with torch.no_grad():
        med = weibull_median(scale, shape).cpu().numpy()
        p10 = weibull_quantile(scale, shape, 0.10).cpu().numpy()
        p90 = weibull_quantile(scale, shape, 0.90).cpu().numpy()
        mean = weibull_mean(scale, shape).cpu().numpy()
    return RULPrediction(
        median=med,
        p10=p10,
        p90=p90,
        mean=mean,
        scale=scale.detach().cpu().numpy(),
        shape=shape.detach().cpu().numpy(),
    )


class CrossAttentionWithRULHead(nn.Module):
    """Wraps an existing CrossAttentionSOHModel and adds a Weibull RUL head.

    The base model is reused as-is. The fused embedding is recomputed inside
    forward() so the RUL head trains on the same fused representation as
    the SOH regressor. This means we can freeze the base network and train
    only the RUL head, or fine-tune both end-to-end with a multi-task loss.
    """

    def __init__(
        self,
        base_model: nn.Module,
        fused_dim: Optional[int] = None,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.base = base_model
        if fused_dim is None:
            fused_dim = self._infer_fused_dim()
        self.rul_head = WeibullHead(fused_dim, hidden_dim=head_hidden_dim, dropout=head_dropout)

    def _infer_fused_dim(self) -> int:
        # CrossAttentionSOHModel.regressor[0] is Linear(in=fused_dim, out=hidden//2)
        first_linear = self.base.regressor[0]
        return first_linear.in_features

    def _compute_fused(self, numeric_seq: torch.Tensor, semantic_emb: torch.Tensor) -> torch.Tensor:
        numeric_pooled, _ = self.base.numeric_encoder(numeric_seq)
        semantic_adapted = self.base.semantic_adapter(semantic_emb)
        if getattr(self.base, "use_cross_attention", False):
            numeric_attended, _ = self.base.cross_attention(numeric_pooled, semantic_adapted, False)
        else:
            numeric_attended = numeric_pooled
        if getattr(self.base, "use_gated_fusion", False):
            fused, _ = self.base.gated_fusion(numeric_attended, semantic_adapted)
        else:
            fused = torch.cat([numeric_attended, semantic_adapted], dim=-1)
        return fused

    def forward(
        self, numeric_seq: torch.Tensor, semantic_emb: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        fused = self._compute_fused(numeric_seq, semantic_emb)
        soh = self.base.regressor(fused).squeeze(-1)
        scale, shape = self.rul_head(fused)
        return {"soh": soh, "rul_scale": scale, "rul_shape": shape}

    def freeze_base(self) -> None:
        for p in self.base.parameters():
            p.requires_grad = False
