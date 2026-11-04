"""Synthetic 18650 cycling data with analytically known ground truth.

The generator exists so that every health indicator can be tested against an
exact answer instead of against another implementation. It simulates a
lumped equivalent-circuit cell:

* **Open-circuit voltage.** :class:`OCVCurve` defines the state of charge as
  an explicit, strictly increasing function of the open-circuit voltage,
  ``SOC(V) = background ramp + sum of logistic steps``. Each logistic step is
  a phase transition of the electrode and shows up as a peak of the
  incremental capacity ``dQ/dV = C * dSOC/dV`` at its centre voltage. The
  default parameters give an NCA/LCO-like curve between 2.5 V and 4.2 V with
  a low-SOC knee and the charge-side peak at 3.90 V OCV (about 4.0 V terminal
  at 1.5 A, as in the NASA PCoE cells).
* **Electrical model.** Terminal voltage ``V = OCV(SOC) + I * R0 + U1`` with a
  series resistance ``R0`` and an optional RC branch (``R1``, ``tau1``;
  ``dU1/dt = (I * R1 - U1) / tau1``). Current is positive while charging and
  negative while discharging (NASA sign convention).
* **Thermal model.** One lumped heat capacity with Newton cooling to ambient,
  heated by the irreversible loss ``I * (V - OCV)``.
* **Protocols.** Constant-current discharge to a cutoff voltage, and the NASA
  constant-current / constant-voltage charge (1.5 A to 4.2 V, then 4.2 V until
  the current falls to 20 mA), each with optional rest periods before and
  after, sampled at a fixed interval with optional seeded Gaussian noise.

The state ``(OCV, U1, T - T_ambient)`` is integrated with an adaptive
Runge-Kutta solver at tight tolerances, so sampled values are exact far below
sensor resolution. Because ``SOC(V)`` is explicit, the quantities the feature
extractors estimate have closed forms (:meth:`CellModel.cc_window`,
:meth:`CellModel.cc_ica_peak`), and any other event time can be located on
the noise-free trajectory (:meth:`SyntheticSegment.crossing_time`).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar

from battery_soh._typing import FloatArray
from battery_soh.config import RANDOM_SEED
from battery_soh.data.schema import Segment

#: Integration tolerances; sampled values are exact to ~1e-7 V / 1e-7 Ah.
_RTOL = 1e-9
_ATOL = 1e-11


# --------------------------------------------------------------------------- cell model


@dataclass(frozen=True)
class LogisticStep:
    """One phase transition of the OCV curve (a peak of dSOC/dV).

    Attributes:
        center_v: Open-circuit voltage of the transition (peak position).
        width_v: Logistic scale ``s``; the dSOC/dV peak has a FWHM of ``3.53 * s``.
        weight: Share of the SOC range contributed by the transition (relative).
    """

    center_v: float
    width_v: float
    weight: float


#: Default transitions: low-SOC knee, main discharge plateau, the charge-side
#: peak (3.90 V OCV, ~4.0 V terminal at 1.5 A) and a high-SOC step.
_DEFAULT_STEPS = (
    LogisticStep(center_v=3.42, width_v=0.070, weight=0.10),
    LogisticStep(center_v=3.63, width_v=0.050, weight=0.38),
    LogisticStep(center_v=3.90, width_v=0.025, weight=0.24),
    LogisticStep(center_v=4.07, width_v=0.030, weight=0.14),
)


def _sigmoid(x: FloatArray) -> FloatArray:
    return np.asarray(0.5 * (1.0 + np.tanh(0.5 * x)), dtype=np.float64)


@dataclass(frozen=True)
class OCVCurve:
    """Strictly increasing open-circuit-voltage curve defined through ``SOC(V)``.

    ``raw(V) = background * (V - v_min) / (v_max - v_min) + sum_k w_k * sigmoid((V - c_k) / s_k)``
    and ``SOC(V) = (raw(V) - raw(v_min)) / (raw(v_max) - raw(v_min))``, so
    ``SOC(v_min) = 0`` and ``SOC(v_max) = 1``. A positive ``background`` makes
    the curve strictly increasing, hence invertible.

    Attributes:
        v_min: OCV at SOC 0 (V).
        v_max: OCV at SOC 1 (V).
        background: Weight of the linear ramp (> 0).
        steps: Logistic phase transitions.
    """

    v_min: float = 2.5
    v_max: float = 4.2
    background: float = 0.10
    steps: tuple[LogisticStep, ...] = _DEFAULT_STEPS

    def __post_init__(self) -> None:
        values = [self.v_min, self.v_max, self.background]
        values += [x for s in self.steps for x in (s.center_v, s.width_v, s.weight)]
        if not all(math.isfinite(x) for x in values):
            raise ValueError("OCV curve parameters must be finite")
        if not self.v_min < self.v_max:
            raise ValueError(f"need v_min < v_max, got {self.v_min} and {self.v_max}")
        if self.background <= 0:
            raise ValueError("background must be > 0 (keeps the curve strictly increasing)")
        if any(s.width_v <= 0 or s.weight < 0 for s in self.steps):
            raise ValueError("logistic steps need width_v > 0 and weight >= 0")

    def _raw(self, v: FloatArray) -> FloatArray:
        out = self.background * (v - self.v_min) / (self.v_max - self.v_min)
        for s in self.steps:
            out = out + s.weight * _sigmoid((v - s.center_v) / s.width_v)
        return np.asarray(out, dtype=np.float64)

    @cached_property
    def _bounds(self) -> tuple[float, float]:
        lo, hi = self._raw(np.array([self.v_min, self.v_max]))
        return float(lo), float(hi - lo)

    def soc(self, voltage_v: ArrayLike) -> FloatArray:
        """State of charge at an open-circuit voltage (0 at ``v_min``, 1 at ``v_max``).

        Outside ``[v_min, v_max]`` the formula is extrapolated (values < 0 or > 1).
        """
        base, span = self._bounds
        return (self._raw(np.asarray(voltage_v, dtype=np.float64)) - base) / span

    def dsoc_dv(self, voltage_v: ArrayLike) -> FloatArray:
        """Derivative ``dSOC/dV`` (1/V); times the capacity it is dQ/dV in Ah/V."""
        v = np.asarray(voltage_v, dtype=np.float64)
        out = np.full(v.shape, self.background / (self.v_max - self.v_min))
        for s in self.steps:
            sig = _sigmoid((v - s.center_v) / s.width_v)
            out = out + s.weight * sig * (1.0 - sig) / s.width_v
        return np.asarray(out / self._bounds[1], dtype=np.float64)

    def dsoc_dv_scalar(self, voltage_v: float) -> float:
        """Scalar fast path of :meth:`dsoc_dv` (used inside the ODE right-hand side)."""
        out = self.background / (self.v_max - self.v_min)
        for s in self.steps:
            sig = 0.5 * (1.0 + math.tanh(0.5 * (voltage_v - s.center_v) / s.width_v))
            out += s.weight * sig * (1.0 - sig) / s.width_v
        return out / self._bounds[1]

    def ocv(self, soc: float) -> float:
        """Open-circuit voltage at a state of charge in ``[0, 1]`` (exact inverse).

        Raises:
            ValueError: If ``soc`` is outside ``[0, 1]``.
        """
        if not 0.0 <= soc <= 1.0:
            raise ValueError(f"soc must be within [0, 1], got {soc}")
        return float(
            brentq(
                lambda v: float(self.soc(v)) - soc,
                self.v_min - 1e-9,
                self.v_max + 1e-9,
                xtol=1e-13,
            )
        )


@dataclass(frozen=True)
class CellModel:
    """Lumped electro-thermal model of one 18650 cell.

    Attributes:
        capacity_ah: Charge between SOC 0 and SOC 1 (Ah).
        r0_ohm: Series (ohmic) resistance; must be > 0 (the CV phase needs it).
        r1_ohm: Resistance of the optional RC polarisation branch (0 disables it).
        tau1_s: Time constant of the RC branch (s).
        heat_capacity_j_per_k: Lumped heat capacity (~45 g x ~0.93 J/(g K)).
        heat_transfer_w_per_k: Convective conductance to ambient (~10 W/(m2 K) x 42 cm2).
        ocv: Open-circuit voltage curve.
    """

    capacity_ah: float = 2.0
    r0_ohm: float = 0.07
    r1_ohm: float = 0.0
    tau1_s: float = 20.0
    heat_capacity_j_per_k: float = 42.0
    heat_transfer_w_per_k: float = 0.042
    ocv: OCVCurve = field(default_factory=OCVCurve)

    def __post_init__(self) -> None:
        values = (
            self.capacity_ah,
            self.r0_ohm,
            self.r1_ohm,
            self.tau1_s,
            self.heat_capacity_j_per_k,
            self.heat_transfer_w_per_k,
        )
        if not all(math.isfinite(x) for x in values):
            raise ValueError("cell parameters must be finite")
        if self.capacity_ah <= 0 or self.r0_ohm <= 0 or self.tau1_s <= 0:
            raise ValueError("capacity_ah, r0_ohm and tau1_s must be > 0")
        if self.r1_ohm < 0:
            raise ValueError("r1_ohm must be >= 0")
        if self.heat_capacity_j_per_k <= 0 or self.heat_transfer_w_per_k <= 0:
            raise ValueError("thermal parameters must be > 0")

    @property
    def steady_resistance_ohm(self) -> float:
        """DC resistance once the RC branch has settled (``R0 + R1``)."""
        return self.r0_ohm + self.r1_ohm

    def aged(self, soh: float, *, resistance_growth: float = 1.5) -> CellModel:
        """Return the same cell after ageing to a state of health ``soh`` (fraction).

        Capacity scales with ``soh``; both resistances grow linearly with the
        capacity fade: ``R -> R * (1 + resistance_growth * (1 - soh))``.

        Raises:
            ValueError: If ``soh`` is not in ``(0, 1]`` or the growth is negative.
        """
        if not 0.0 < soh <= 1.0:
            raise ValueError(f"soh must be in (0, 1], got {soh}")
        if resistance_growth < 0:
            raise ValueError("resistance_growth must be >= 0")
        factor = 1.0 + resistance_growth * (1.0 - soh)
        return replace(
            self,
            capacity_ah=self.capacity_ah * soh,
            r0_ohm=self.r0_ohm * factor,
            r1_ohm=self.r1_ohm * factor,
        )

    # ------------------------------------------------------------ closed-form ground truth

    def cc_window(self, v_from: float, v_to: float, current_a: float) -> tuple[float, float]:
        """Charge (Ah) and time (s) for the terminal voltage to go ``v_from -> v_to`` under CC.

        Exact when ``r1_ohm == 0`` and otherwise once the RC branch has settled:
        the terminal voltage is then ``OCV + I * (R0 + R1)``, so the window is
        the OCV window shifted by ``-I * (R0 + R1)`` and holds
        ``C * |SOC(v_to - I R) - SOC(v_from - I R)|``.

        Args:
            v_from: Terminal voltage at the start of the window.
            v_to: Terminal voltage at the end of the window.
            current_a: Signed current (positive charge, negative discharge).

        Raises:
            ValueError: If ``current_a`` is zero.
        """
        if current_a == 0:
            raise ValueError("current_a must be non-zero")
        shift = current_a * self.steady_resistance_ohm
        soc = self.ocv.soc(np.array([v_from - shift, v_to - shift]))
        charge_ah = abs(float(soc[1] - soc[0])) * self.capacity_ah
        return charge_ah, charge_ah * 3600.0 / abs(current_a)

    def cc_ica_peak(
        self, current_a: float, v_lo: float = 3.90, v_hi: float = 4.18
    ) -> tuple[float, float]:
        """Height (Ah/V) and terminal voltage of the CC-charge dQ/dV maximum in a window.

        During CC charge with a settled RC branch,
        ``dQ/dV(V) = C * dSOC/dV(V - I * (R0 + R1))``.

        Args:
            current_a: Charge current (A, > 0).
            v_lo: Lower terminal voltage of the search window.
            v_hi: Upper terminal voltage of the search window.

        Raises:
            ValueError: If ``current_a <= 0`` or ``v_lo >= v_hi``.
        """
        if current_a <= 0 or not v_lo < v_hi:
            raise ValueError("need current_a > 0 and v_lo < v_hi")
        shift = current_a * self.steady_resistance_ohm
        grid = np.linspace(v_lo, v_hi, 20_001)
        j = int(np.argmax(self.ocv.dsoc_dv(grid - shift)))
        v_peak = float(grid[j])
        if 0 < j < grid.size - 1:
            res = minimize_scalar(
                lambda v: -self.ocv.dsoc_dv_scalar(v - shift),
                bounds=(float(grid[j - 1]), float(grid[j + 1])),
                method="bounded",
                options={"xatol": 1e-10},
            )
            v_peak = float(res.x)
        return self.capacity_ah * self.ocv.dsoc_dv_scalar(v_peak - shift), v_peak


@dataclass(frozen=True)
class Noise:
    """Standard deviations of additive Gaussian sensor noise."""

    voltage_v: float = 0.0
    current_a: float = 0.0
    temperature_c: float = 0.0

    def __post_init__(self) -> None:
        values = (self.voltage_v, self.current_a, self.temperature_c)
        if not all(math.isfinite(x) and x >= 0 for x in values):
            raise ValueError("noise levels must be finite and >= 0")


#: Realistic sensor noise: 1 mV, 2 mA, 0.02 degC (NASA logs are at least this clean).
SENSOR_NOISE = Noise(voltage_v=1e-3, current_a=2e-3, temperature_c=0.02)


# --------------------------------------------------------------------------- trajectories

PhaseMode = Literal["rest", "cc", "cv"]


@dataclass(frozen=True)
class _Law:
    """Current law of one protocol phase.

    ``rest``: no current; ``cc``: constant signed current ``value`` (A);
    ``cv``: terminal voltage held at ``value`` (V), so ``I = (value - OCV - U1) / R0``.
    """

    mode: PhaseMode
    value: float
    r0_ohm: float

    def current(self, ocv_v: FloatArray, u1_v: FloatArray) -> FloatArray:
        if self.mode == "rest":
            return np.zeros_like(ocv_v)
        if self.mode == "cc":
            return np.full_like(ocv_v, self.value)
        return (self.value - ocv_v - u1_v) / self.r0_ohm

    def current_scalar(self, ocv_v: float, u1_v: float) -> float:
        if self.mode == "rest":
            return 0.0
        if self.mode == "cc":
            return self.value
        return (self.value - ocv_v - u1_v) / self.r0_ohm


@dataclass(frozen=True)
class _Phase:
    law: _Law
    t0: float
    t1: float
    solution: Callable[[FloatArray], FloatArray]  # times -> states (3, n): OCV, U1, theta


@dataclass(frozen=True)
class SyntheticSegment:
    """A sampled synthetic segment plus its noise-free model.

    Attributes:
        segment: The sampled (optionally noisy) data, NASA sign convention.
        truth: Exact event times and totals of the simulation: ``t_load_on_s``,
            ``t_load_off_s``, ``current_a``, ``delivered_ah`` (discharge) or
            ``t_cc_start_s``, ``t_cv_start_s``, ``t_cv_end_s``, ``current_a``,
            ``charged_ah`` (charge).
        cell: The simulated cell.
        ambient_c: Ambient temperature (degC).
    """

    segment: Segment
    truth: Mapping[str, float]
    cell: CellModel
    ambient_c: float
    _phases: tuple[_Phase, ...] = field(repr=False, compare=False)

    def noiseless(self, t: ArrayLike) -> dict[str, FloatArray]:
        """Evaluate the noise-free model at times ``t`` (s); NaN outside the record.

        A phase owns ``[t0, t1)`` (the last one also ``t1``), so a sample taken
        exactly at a switching instant already sees the new current.

        Returns:
            ``voltage_v``, ``current_a``, ``temperature_c``, ``ocv_v`` and ``soc`` arrays.
        """
        times = np.atleast_1d(np.asarray(t, dtype=np.float64))
        ocv_v, current, u1, theta = (np.full(times.shape, np.nan) for _ in range(4))
        last = len(self._phases) - 1
        for n, ph in enumerate(self._phases):
            upper = times <= ph.t1 if n == last else times < ph.t1
            mask = (times >= ph.t0) & upper
            if mask.any():
                ocv_v[mask], u1[mask], theta[mask] = ph.solution(times[mask])
                current[mask] = ph.law.current(ocv_v[mask], u1[mask])
        return {
            "voltage_v": ocv_v + current * self.cell.r0_ohm + u1,
            "current_a": current,
            "temperature_c": self.ambient_c + theta,
            "ocv_v": ocv_v,
            "soc": self.cell.ocv.soc(ocv_v),
        }

    def crossing_time(
        self,
        signal: Literal["voltage_v", "current_a", "temperature_c"],
        level: float,
        *,
        rising: bool,
        t_min: float = 0.0,
        resolution_s: float = 0.5,
    ) -> float:
        """First time at or after ``t_min`` at which the noise-free ``signal`` reaches ``level``.

        The crossing is bracketed on a ``resolution_s`` grid and refined with
        Brent's method inside one phase (where the signal is continuous); a
        jump at a phase switch (e.g. the load-on voltage step) counts as a
        crossing at the switching instant. Returns NaN if the level is never
        reached.
        """
        t_end = self._phases[-1].t1
        grid = np.append(np.arange(t_min, t_end, resolution_s), t_end)
        values = self.noiseless(grid)[signal]
        idx = np.flatnonzero(values >= level if rising else values <= level)
        if idx.size == 0:
            return math.nan
        j = int(idx[0])
        if j == 0:
            return float(grid[0])

        def gap(x: float) -> float:
            return float(self.noiseless(np.array([x]))[signal][0]) - level

        def reached(x: float) -> bool:
            return gap(x) >= 0 if rising else gap(x) <= 0

        cuts = [ph.t0 for ph in self._phases if grid[j - 1] < ph.t0 <= grid[j]]
        bounds = [float(grid[j - 1]), *cuts, float(grid[j])]
        for lo, hi in itertools.pairwise(bounds):
            if reached(lo):
                return lo  # jump at a phase switch
            hi_inside = hi - 1e-9 if hi in cuts else hi  # left limit before a switch
            if hi_inside > lo and reached(hi_inside):
                return float(brentq(gap, lo, hi_inside, xtol=1e-10))
        return float(grid[j])  # pragma: no cover - unreachable: grid[j] is reached


class _Simulator:
    """Integrates the cell state through a sequence of protocol phases."""

    def __init__(self, cell: CellModel, soc_start: float) -> None:
        self.cell = cell
        self.t = 0.0
        self.y = np.array([cell.ocv.ocv(soc_start), 0.0, 0.0])  # relaxed, at ambient
        self.phases: list[_Phase] = []

    def terminal_voltage(self, law: _Law, y: FloatArray) -> float:
        i = law.current_scalar(float(y[0]), float(y[1]))
        return float(y[0] + i * self.cell.r0_ohm + y[1])

    def run(
        self,
        law: _Law,
        duration_s: float,
        *,
        stop: Callable[[float, FloatArray], float] | None = None,
        direction: float = 0.0,
    ) -> None:
        """Advance by ``duration_s`` or until ``stop`` crosses zero in ``direction``."""
        if duration_s <= 0:
            return
        cell = self.cell
        inv_q = 1.0 / (3600.0 * cell.capacity_ah)

        def rhs(_t: float, y: FloatArray) -> FloatArray:
            i = law.current_scalar(float(y[0]), float(y[1]))
            return np.array(
                [
                    i * inv_q / cell.ocv.dsoc_dv_scalar(float(y[0])),
                    (i * cell.r1_ohm - y[1]) / cell.tau1_s,
                    (i * (i * cell.r0_ohm + y[1]) - cell.heat_transfer_w_per_k * y[2])
                    / cell.heat_capacity_j_per_k,
                ]
            )

        events = None
        if stop is not None:
            setattr(stop, "terminal", True)  # noqa: B010 - solve_ivp reads these attributes
            setattr(stop, "direction", direction)  # noqa: B010
            events = [stop]
        t0 = self.t
        sol = solve_ivp(
            rhs,
            (t0, t0 + duration_s),
            self.y,
            method="RK45",
            rtol=_RTOL,
            atol=_ATOL,
            dense_output=True,
            events=events,
        )
        if not sol.success:  # pragma: no cover - smooth ODE
            raise RuntimeError(f"simulation of a {law.mode} phase failed: {sol.message}")
        t1 = float(sol.t[-1])
        dense = sol.sol

        def solution(t: FloatArray) -> FloatArray:
            return np.asarray(dense(np.clip(t, t0, t1)), dtype=np.float64).reshape(3, -1)

        self.phases.append(_Phase(law=law, t0=t0, t1=t1, solution=solution))
        self.t, self.y = t1, np.asarray(sol.y[:, -1], dtype=np.float64)

    def sample(
        self,
        kind: Literal["charge", "discharge"],
        *,
        sample_s: float,
        ambient_c: float,
        noise: Noise | None,
        seed: int,
        truth: dict[str, float],
        capacity_ah: float | None,
    ) -> SyntheticSegment:
        """Sample every ``sample_s`` seconds from 0 to the end of the last phase."""
        model = SyntheticSegment(
            segment=Segment(kind=kind),
            truth=MappingProxyType(dict(truth)),
            cell=self.cell,
            ambient_c=ambient_c,
            _phases=tuple(self.phases),
        )
        times = np.arange(0.0, self.t + 1e-9, sample_s)
        values = model.noiseless(times)
        v, i, temp = values["voltage_v"], values["current_a"], values["temperature_c"]
        if noise is not None:
            rng = np.random.default_rng(seed)
            v = v + rng.normal(0.0, noise.voltage_v, times.size)
            i = i + rng.normal(0.0, noise.current_a, times.size)
            temp = temp + rng.normal(0.0, noise.temperature_c, times.size)
        segment = Segment(
            kind=kind,
            time_s=times,
            voltage_v=v,
            current_a=i,
            temperature_c=temp,
            ambient_c=ambient_c,
            capacity_ah=capacity_ah,
        )
        return replace(model, segment=segment)


def _check_common(
    current_a: float, soc_start: float, sample_s: float, rest_before_s: float, rest_after_s: float
) -> None:
    if not (math.isfinite(current_a) and current_a > 0):
        raise ValueError(f"current_a must be a positive magnitude, got {current_a}")
    if not (math.isfinite(sample_s) and sample_s > 0):
        raise ValueError(f"sample_s must be > 0, got {sample_s}")
    if not (rest_before_s >= 0 and rest_after_s >= 0):
        raise ValueError("rest durations must be >= 0")
    if not 0.0 <= soc_start <= 1.0:
        raise ValueError(f"soc_start must be within [0, 1], got {soc_start}")


# --------------------------------------------------------------------------- protocols


def simulate_discharge(
    cell: CellModel,
    *,
    current_a: float = 2.0,
    cutoff_v: float = 2.7,
    soc_start: float = 1.0,
    rest_before_s: float = 20.0,
    rest_after_s: float = 300.0,
    max_load_s: float | None = None,
    sample_s: float = 10.0,
    ambient_c: float = 24.0,
    noise: Noise | None = None,
    seed: int = RANDOM_SEED,
) -> SyntheticSegment:
    """Simulate rest -> constant-current discharge to ``cutoff_v`` -> rest.

    The load switches on at ``rest_before_s``. When that is a multiple of
    ``sample_s`` the first loaded sample is taken at the switch-on instant,
    before the OCV or the RC branch have moved, so ``V[k0-1] - V[k0] = I * R0``
    exactly. The recorded current is negative (NASA convention).

    Args:
        cell: Cell to simulate.
        current_a: Load current magnitude (A, > 0).
        cutoff_v: Terminal voltage that ends the load.
        soc_start: Initial state of charge (cell relaxed, at ambient).
        rest_before_s: Rest before the load (s).
        rest_after_s: Rest after the load, i.e. the relaxation tail (s).
        max_load_s: Optional cap on the load duration (a partial discharge).
        sample_s: Sampling interval (s).
        ambient_c: Ambient temperature (degC).
        noise: Optional sensor noise.
        seed: Seed of the noise generator.

    Returns:
        The sampled segment (``capacity_ah`` = delivered charge) and its truth:
        ``t_load_on_s``, ``t_load_off_s``, ``current_a``, ``delivered_ah``.

    Raises:
        ValueError: If a parameter is invalid, the loaded cell starts below the
            cutoff, or the cutoff lies below the OCV range (over-discharge).
    """
    _check_common(current_a, soc_start, sample_s, rest_before_s, rest_after_s)
    if cell.ocv.ocv(soc_start) - current_a * cell.r0_ohm <= cutoff_v:
        raise ValueError("the loaded cell starts at or below the cutoff voltage")
    if cutoff_v + current_a * cell.r0_ohm <= cell.ocv.v_min:
        raise ValueError("cutoff_v is unreachable inside the OCV range (over-discharge)")
    if max_load_s is not None and not max_load_s > 0:
        raise ValueError("max_load_s must be > 0")

    sim = _Simulator(cell, soc_start)
    sim.run(_Law("rest", 0.0, cell.r0_ohm), rest_before_s)
    load = _Law("cc", -current_a, cell.r0_ohm)

    def cutoff(_t: float, y: FloatArray) -> float:
        return sim.terminal_voltage(load, y) - cutoff_v

    full_s = 1.5 * 3600.0 * cell.capacity_ah / current_a  # upper bound; the cutoff stops it
    t_on = sim.t
    sim.run(
        load,
        full_s if max_load_s is None else min(full_s, max_load_s),
        stop=cutoff,
        direction=-1.0,
    )
    t_off = sim.t
    sim.run(_Law("rest", 0.0, cell.r0_ohm), rest_after_s)
    delivered = current_a * (t_off - t_on) / 3600.0
    truth = {
        "t_load_on_s": t_on,
        "t_load_off_s": t_off,
        "current_a": current_a,
        "delivered_ah": delivered,
    }
    return sim.sample(
        "discharge",
        sample_s=sample_s,
        ambient_c=ambient_c,
        noise=noise,
        seed=seed,
        truth=truth,
        capacity_ah=delivered,
    )


def simulate_charge(
    cell: CellModel,
    *,
    current_a: float = 1.5,
    v_max: float = 4.2,
    cutoff_a: float = 0.02,
    soc_start: float = 0.05,
    rest_before_s: float = 20.0,
    rest_after_s: float = 0.0,
    sample_s: float = 10.0,
    ambient_c: float = 24.0,
    noise: Noise | None = None,
    seed: int = RANDOM_SEED,
) -> SyntheticSegment:
    """Simulate rest -> CC charge to ``v_max`` -> CV at ``v_max`` until ``cutoff_a`` -> rest.

    Defaults follow the NASA PCoE protocol (1.5 A to 4.2 V, then 4.2 V until
    20 mA). The recorded current is positive (NASA convention).

    Args:
        cell: Cell to simulate.
        current_a: CC current (A, > 0).
        v_max: Charge voltage limit (CC -> CV transition and CV set point).
        cutoff_a: CV current that terminates the charge (A).
        soc_start: Initial state of charge (cell relaxed, at ambient).
        rest_before_s: Rest before the charger switches on (s).
        rest_after_s: Rest after the charge (s).
        sample_s: Sampling interval (s).
        ambient_c: Ambient temperature (degC).
        noise: Optional sensor noise.
        seed: Seed of the noise generator.

    Returns:
        The sampled segment and its truth: ``t_cc_start_s``, ``t_cv_start_s``
        (terminal voltage reaches ``v_max``), ``t_cv_end_s`` (current falls to
        ``cutoff_a``), ``current_a`` and ``charged_ah``.

    Raises:
        ValueError: If a parameter is invalid.
    """
    _check_common(current_a, soc_start, sample_s, rest_before_s, rest_after_s)
    if not 0 < cutoff_a < current_a:
        raise ValueError("need 0 < cutoff_a < current_a")
    if not cell.ocv.ocv(soc_start) + current_a * cell.r0_ohm < v_max <= cell.ocv.v_max:
        raise ValueError("v_max must lie above the CC start voltage and within the OCV range")

    sim = _Simulator(cell, soc_start)
    sim.run(_Law("rest", 0.0, cell.r0_ohm), rest_before_s)
    cc = _Law("cc", current_a, cell.r0_ohm)
    cv = _Law("cv", v_max, cell.r0_ohm)

    def reach_vmax(_t: float, y: FloatArray) -> float:
        return sim.terminal_voltage(cc, y) - v_max

    def taper_end(_t: float, y: FloatArray) -> float:
        return cv.current_scalar(float(y[0]), float(y[1])) - cutoff_a

    full_s = 1.5 * 3600.0 * cell.capacity_ah / current_a
    t_cc = sim.t
    sim.run(cc, full_s, stop=reach_vmax, direction=1.0)
    t_cv = sim.t
    sim.run(cv, 20.0 * full_s, stop=taper_end, direction=-1.0)
    t_end = sim.t
    soc_end = float(cell.ocv.soc(sim.y[0]))
    sim.run(_Law("rest", 0.0, cell.r0_ohm), rest_after_s)
    truth = {
        "t_cc_start_s": t_cc,
        "t_cv_start_s": t_cv,
        "t_cv_end_s": t_end,
        "current_a": current_a,
        "charged_ah": (soc_end - soc_start) * cell.capacity_ah,
    }
    return sim.sample(
        "charge",
        sample_s=sample_s,
        ambient_c=ambient_c,
        noise=noise,
        seed=seed,
        truth=truth,
        capacity_ah=None,
    )
