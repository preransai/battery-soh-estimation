"""Physics-informed loss terms for SOH regression.

Adds three soft constraints on top of the usual MSE objective so the model
respects degradation physics rather than fitting the data alone:

1. Monotonicity penalty (MonotonicityLoss)
   For two cycles from the same battery with cycle_a < cycle_b, the
   predicted SOH(a) should be >= SOH(b). Penalises any reversal.

2. Arrhenius temperature coupling (ArrheniusLoss)
   Hotter cycles should fade faster than cooler cycles. Operationalised
   as a Spearman-style rank consistency penalty between cycle temperature
   and predicted local fade rate.

3. Capacity-resistance coupling (CapacityResistanceLoss)
   Capacity loss and DCIR rise are physically coupled (Butler-Volmer-like
   degradation). Penalises predictions that contradict the observed
   capacity-DCIR co-trend.

The PINNCompositeLoss wraps the standard MSE plus weighted physics terms
into a drop-in replacement for ``nn.MSELoss`` inside CrossAttentionTrainer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn


def _pairwise_diff(x: torch.Tensor) -> torch.Tensor:
    """Returns x.unsqueeze(0) - x.unsqueeze(1), shape (n, n)."""
    return x.unsqueeze(0) - x.unsqueeze(1)


class MonotonicityLoss(nn.Module):
    """Penalises predicted-SOH increases between later cycles of the same battery.

    Hinge form: max(0, soh_pred[b] - soh_pred[a]) when cycle_b > cycle_a.
    Operates per-battery within the minibatch; if a battery has only one
    sample in the batch the term contributes zero.
    """

    def __init__(self, margin: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.margin = float(margin)
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    def forward(
        self,
        soh_pred: torch.Tensor,
        cycle_number: torch.Tensor,
        battery_id: torch.Tensor,
    ) -> torch.Tensor:
        device = soh_pred.device
        violations = []
        unique_batteries = torch.unique(battery_id)
        for bat in unique_batteries:
            mask = battery_id == bat
            if mask.sum() < 2:
                continue
            cyc = cycle_number[mask].float()
            pred = soh_pred[mask]
            cycle_gap = _pairwise_diff(cyc)        # cyc_b - cyc_a
            pred_gap = _pairwise_diff(pred)         # pred_b - pred_a
            later_pairs = cycle_gap > 0
            if not later_pairs.any():
                continue
            violation = torch.relu(pred_gap[later_pairs] + self.margin)
            violations.append(violation)
        if not violations:
            return torch.zeros((), device=device)
        flat = torch.cat(violations)
        return flat.mean() if self.reduction == "mean" else flat.sum()


class ArrheniusLoss(nn.Module):
    """Rank-consistency penalty between mean temperature and predicted fade.

    For each battery in the batch, compute the local fade signal
        fade = soh_at_first_cycle - soh_pred  (per sample)
    and its mean temperature ``temp_mean``. Penalise any pair (i, j) within
    the same battery where temp_i > temp_j but fade_i < fade_j (i.e. the
    hotter cycle aged less). Uses a soft hinge so gradients flow.
    """

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = float(margin)

    def forward(
        self,
        soh_pred: torch.Tensor,
        temp_mean: torch.Tensor,
        battery_id: torch.Tensor,
    ) -> torch.Tensor:
        device = soh_pred.device
        violations = []
        unique_batteries = torch.unique(battery_id)
        for bat in unique_batteries:
            mask = battery_id == bat
            if mask.sum() < 2:
                continue
            pred = soh_pred[mask]
            temp = temp_mean[mask].float()
            fade = pred.max().detach() - pred           # higher = more degraded
            temp_gap = _pairwise_diff(temp)              # T_i - T_j
            fade_gap = _pairwise_diff(fade)              # fade_i - fade_j
            # Want: temp_gap > 0 implies fade_gap >= 0. Penalise the opposite.
            hot_pairs = temp_gap > 0
            if not hot_pairs.any():
                continue
            violation = torch.relu(self.margin - fade_gap[hot_pairs])
            violations.append(violation)
        if not violations:
            return torch.zeros((), device=device)
        flat = torch.cat(violations)
        return flat.mean()


class CapacityResistanceLoss(nn.Module):
    """Penalises predictions that disagree with the capacity-DCIR co-trend.

    Lower observed capacity tends to be paired with higher observed DCIR.
    For each battery, we form a normalised "physical fade index"
        physical_fade = z(-capacity) + z(dcir)
    (z is per-battery z-score) and penalise the squared deviation between
    its rank order and the rank order of the model's predicted SOH.

    Implemented as a soft pairwise loss: for pairs with physical_fade_i >
    physical_fade_j, penalise (soh_pred_i - soh_pred_j) > 0 (i.e. the
    model thinks the more-degraded cycle is healthier).
    """

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = float(margin)

    @staticmethod
    def _zscore(x: torch.Tensor) -> torch.Tensor:
        std = x.std(unbiased=False)
        if std.item() < 1e-8:
            return torch.zeros_like(x)
        return (x - x.mean()) / std

    def forward(
        self,
        soh_pred: torch.Tensor,
        capacity: torch.Tensor,
        dcir: torch.Tensor,
        battery_id: torch.Tensor,
    ) -> torch.Tensor:
        device = soh_pred.device
        violations = []
        unique_batteries = torch.unique(battery_id)
        for bat in unique_batteries:
            mask = battery_id == bat
            if mask.sum() < 2:
                continue
            pred = soh_pred[mask]
            cap = capacity[mask].float()
            res = dcir[mask].float()
            phys = self._zscore(-cap) + self._zscore(res)
            phys_gap = _pairwise_diff(phys)
            pred_gap = _pairwise_diff(pred)
            degraded_more = phys_gap > 0
            if not degraded_more.any():
                continue
            violation = torch.relu(pred_gap[degraded_more] + self.margin)
            violations.append(violation)
        if not violations:
            return torch.zeros((), device=device)
        flat = torch.cat(violations)
        return flat.mean()


@dataclass
class PINNWeights:
    """Loss weights. Defaults are conservative; tune on a validation set."""

    mse: float = 1.0
    monotonicity: float = 0.05
    arrhenius: float = 0.02
    capacity_resistance: float = 0.05


@dataclass
class PINNDiagnostics:
    mse: float = 0.0
    monotonicity: float = 0.0
    arrhenius: float = 0.0
    capacity_resistance: float = 0.0
    total: float = 0.0


class PINNCompositeLoss(nn.Module):
    """MSE + weighted physics penalties. Drop-in for nn.MSELoss.

    Forward signature differs from nn.MSELoss because it needs side information
    (cycle number, temperature, capacity, DCIR, battery id). Use it from a
    custom training loop or a thin trainer adapter rather than the stock one.
    """

    def __init__(self, weights: Optional[PINNWeights] = None):
        super().__init__()
        self.weights = weights or PINNWeights()
        self.mse = nn.MSELoss()
        self.monotonicity = MonotonicityLoss()
        self.arrhenius = ArrheniusLoss()
        self.capacity_resistance = CapacityResistanceLoss()
        self.last: PINNDiagnostics = PINNDiagnostics()

    def forward(
        self,
        soh_pred: torch.Tensor,
        soh_true: torch.Tensor,
        cycle_number: torch.Tensor,
        temp_mean: torch.Tensor,
        capacity: torch.Tensor,
        dcir: torch.Tensor,
        battery_id: torch.Tensor,
    ) -> torch.Tensor:
        mse = self.mse(soh_pred, soh_true)
        mono = self.monotonicity(soh_pred, cycle_number, battery_id)
        arr = self.arrhenius(soh_pred, temp_mean, battery_id)
        capres = self.capacity_resistance(soh_pred, capacity, dcir, battery_id)
        w = self.weights
        total = (
            w.mse * mse
            + w.monotonicity * mono
            + w.arrhenius * arr
            + w.capacity_resistance * capres
        )
        self.last = PINNDiagnostics(
            mse=mse.item(),
            monotonicity=mono.item(),
            arrhenius=arr.item(),
            capacity_resistance=capres.item(),
            total=total.item(),
        )
        return total
