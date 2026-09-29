"""Introspection layer over ``Model_Calibration.event_models``.

The GUI never hard-codes a model list: everything (available models, their
parameters, bounds and initial guesses) is read from the registry, so a model
added to ``event_models.py`` shows up in the interface automatically.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math

import numpy as np

from Model_Calibration.event_models import get_event_model


# Canonical names, ordered by increasing complexity. Probed against the
# registry at import time so a typo or a removed model degrades gracefully.
_CANDIDATE_NAMES: Tuple[str, ...] = (
    "single_exp",
    "alpha",
    "gamma",
    "bilinear",
    "double_exp",
    "cooperative",
    "two_step_binding",
    "binding_kinetics",
    "iglusnfr",
    "two_component",
    "desensitization",
    "coop_plus_linear",
    "diffusion_clearance",
    "double_cooperative",
    "iglusnfr_tri",
    "hetero_coop",
    "two_comp_coop",
)

# Models that extract_metrics fits with a tau-varying kernel; the rest are used
# as fixed-shape templates whose amplitude alone is solved per pulse.
TAU_VARYING: frozenset = frozenset({"double_exp", "cooperative", "bilinear"})

# Keys that additionally shape the kinetics search grids inside extract_metrics
# (they are validated strictly: finite, > 0 and lower < upper).
GRID_BOUND_KEYS: frozenset = frozenset({"tau_rise", "tau_decay_fast", "tau_decay", "tau_decay_slow"})

ONE_LINERS: Dict[str, str] = {
    "single_exp": "Instantaneous rise, single exponential decay. The minimal baseline model.",
    "alpha": "Alpha function: one time constant sets both rise and decay.",
    "gamma": "Gamma function; the shape exponent n decouples rise sharpness from decay.",
    "bilinear": "Linear rise then exponential decay. Rise/decay are given in ms, not seconds.",
    "double_exp": "Classic difference of exponentials, peak-normalised so amp stays interpretable.",
    "cooperative": "Hill-like cooperative rise times an exponential decay.",
    "two_step_binding": "Binding followed by a conformational step, then dissociation.",
    "binding_kinetics": "Explicit on/off rates plus a clearance term.",
    "iglusnfr": "Bi-exponential iGluSnFR: fast and slow decay sharing one rise.",
    "two_component": "Two amplitudes sharing a rise, each with its own decay.",
    "desensitization": "Rise/decay with an added desensitisation and recovery term.",
    "coop_plus_linear": "Cooperative component plus a slower linear component.",
    "diffusion_clearance": "Diffusion-limited rise with bi-exponential clearance.",
    "double_cooperative": "Sum of two independent cooperative components.",
    "iglusnfr_tri": "Tri-exponential iGluSnFR. The super-slow term is applied via NNLS template variants.",
    "hetero_coop": "Two cooperative populations mixed by frac1.",
    "two_comp_coop": "Two cooperative components with independent amplitudes.",
}


class ModelInfo:
    """A registry entry plus the metadata the GUI needs to render it."""

    __slots__ = ("name", "func", "params", "lower", "upper", "p0_func",
                 "complexity", "progression_rules", "summary")

    def __init__(self, name: str, spec: Dict) -> None:
        lb, ub = spec["bounds"]
        self.name = str(spec.get("name", name))
        self.func = spec["func"]
        self.params: List[str] = list(spec["params"])
        self.lower: List[float] = [float(v) for v in lb]
        self.upper: List[float] = [float(v) for v in ub]
        self.p0_func = spec["p0_func"]
        self.complexity = int(spec.get("complexity", len(self.params)))
        self.progression_rules: Dict[str, str] = dict(spec.get("progression_rules", {}) or {})
        self.summary = ONE_LINERS.get(self.name, "")

    @property
    def tau_varying(self) -> bool:
        return self.name in TAU_VARYING

    def bounds_for(self, param: str) -> Tuple[float, float]:
        i = self.params.index(param)
        return self.lower[i], self.upper[i]

    def initial_guess(self, y: np.ndarray, t_ms: np.ndarray) -> List[float]:
        """Registry p0, clipped into the registry bounds."""
        try:
            p0 = [float(v) for v in self.p0_func(y, t_ms)]
        except Exception:
            p0 = [float(v) for v in self.display_defaults()]
        out = []
        for value, lo, hi in zip(p0, self.lower, self.upper):
            if not math.isfinite(value):
                value = lo if math.isfinite(lo) else 0.0
            out.append(min(max(value, lo), hi if math.isfinite(hi) else value))
        return out

    def display_defaults(self) -> List[float]:
        """Finite fallback values: bound midpoint, or the lower bound if open."""
        out = []
        for lo, hi in zip(self.lower, self.upper):
            if math.isfinite(lo) and math.isfinite(hi):
                out.append(0.5 * (lo + hi))
            elif math.isfinite(lo):
                out.append(max(lo, 1.0))
            else:
                out.append(0.0)
        return out

    def slider_bounds(self, param: str, value: Optional[float] = None) -> Tuple[float, float]:
        """Finite bounds usable by a slider (``amp`` has an infinite upper bound)."""
        lo, hi = self.bounds_for(param)
        if not math.isfinite(hi):
            base = abs(float(value)) if value not in (None, 0) and math.isfinite(float(value)) else 1.0
            hi = max(base * 4.0, lo + 1.0)
        if not math.isfinite(lo):
            lo = 0.0
        return lo, hi


_CACHE: Dict[str, ModelInfo] = {}
_NAMES: List[str] = []

for _name in _CANDIDATE_NAMES:
    try:
        _CACHE[_name] = ModelInfo(_name, get_event_model(_name))
        _NAMES.append(_name)
    except Exception:  # pragma: no cover - a model removed from the registry
        continue


def model_names() -> List[str]:
    """Every model the registry actually resolves, simplest first."""
    return list(_NAMES)


def model(name: str) -> ModelInfo:
    key = str(name).strip().lower()
    if key.startswith("library:"):
        key = key.split(":", 1)[1].strip()
    if key in _CACHE:
        return _CACHE[key]
    info = ModelInfo(key, get_event_model(key))  # raises ValueError if unknown
    _CACHE[key] = info
    return info


def label_for(name: str) -> str:
    """Menu label: name, parameter count and complexity."""
    info = model(name)
    kind = "tau-varying" if info.tau_varying else "template"
    return f"{info.name}  ({len(info.params)}p, {kind})"


def name_from_label(label: str) -> str:
    return str(label).split("(")[0].strip()


# --- parameter presentation -------------------------------------------------

_UNIT_OVERRIDES: Dict[str, str] = {
    "t_onset": "ms",
    "t_rise": "ms",
    "t_decay": "ms",
    "kon": "1/s",
    "koff": "1/s",
}

_HELP_OVERRIDES: Dict[str, str] = {
    "amp": "Peak amplitude of the event, in trace units.",
    "t_onset": "Event onset relative to the stimulus, in milliseconds. Negative values start before the stimulus.",
    "tau_rise": "Rise time constant. Shorter values give a sharper onset.",
    "tau_decay": "Decay time constant of the single decay component.",
    "tau_decay_fast": "Fast decay component. Dominates the first ~10 ms after the peak.",
    "tau_decay_slow": "Slow decay component. Sets how much signal is carried into the next pulse.",
    "tau_decay_superslow": "Super-slow accumulation component; applied through NNLS template variants.",
    "frac_fast": "Weight of the fast decay component (0-1). The remainder goes to the slow component.",
    "frac_slow": "Weight of the slow component in the tri-exponential mix.",
    "n_coop": "Hill coefficient of the cooperative rise. Higher values give a sigmoidal onset.",
    "desens_factor": "Fraction of the response lost to desensitisation (0-1).",
    "kon": "Association rate constant.",
    "koff": "Dissociation rate constant.",
    "tau_clear": "Clearance time constant of free ligand.",
    "tau_diff": "Diffusion-limited rise time constant.",
    "tau_recovery": "Recovery time constant from the desensitised state.",
    "tau_bind": "Binding step time constant.",
    "tau_conform": "Conformational-change time constant.",
    "tau_dissoc": "Dissociation time constant.",
}


def param_unit(name: str) -> str:
    if name in _UNIT_OVERRIDES:
        return _UNIT_OVERRIDES[name]
    if name.startswith("tau"):
        return "s"
    if name.startswith("amp"):
        return ""
    return ""  # n, frac_*, desens_factor are dimensionless


def param_help(name: str) -> str:
    if name in _HELP_OVERRIDES:
        return _HELP_OVERRIDES[name]
    if name.startswith("amp"):
        return f"Amplitude of the '{name.replace('amp_', '')}' component, in trace units."
    if name.startswith("tau"):
        return f"Time constant '{name}', in seconds."
    if name.startswith("frac"):
        return f"Mixing fraction '{name}' (0-1)."
    if name.startswith("n"):
        return f"Shape/cooperativity exponent '{name}'."
    return name


def is_amplitude(name: str) -> bool:
    return name.startswith("amp")


# --- bridge to extract_metrics parameter_bounds -----------------------------

# Model parameter -> the key extract_metrics uses when building its kinetics
# grids. Unmapped parameters are still passed through: apply_parameter_bounds()
# clips any fitted parameter whose name appears in parameter_bounds.
_CANONICAL: Dict[str, str] = {
    "tau_rise": "tau_rise",
    "tau_rise_coop": "tau_rise",
    "tau_rise_fast": "tau_rise",
    "tau_rise1": "tau_rise",
    "tau_bind": "tau_rise",
    "tau_diff": "tau_rise",
    "tau_decay": "tau_decay_fast",
    "tau_decay_fast": "tau_decay_fast",
    "tau_fast": "tau_decay_fast",
    "tau_decay1": "tau_decay_fast",
    "tau_decay_coop": "tau_decay_fast",
    "tau_clear1": "tau_decay_fast",
    "tau_conform": "tau_decay_fast",
    "tau_decay_slow": "tau_decay_slow",
    "tau_slow": "tau_decay_slow",
    "tau_decay2": "tau_decay_slow",
    "tau_decay_linear": "tau_decay_slow",
    "tau_clear2": "tau_decay_slow",
    "tau_dissoc": "tau_decay_slow",
    "tau_decay_superslow": "tau_decay_superslow",
}


def canonical_bound_key(param: str) -> Optional[str]:
    """The extract_metrics ``parameter_bounds`` key a model parameter maps to."""
    return _CANONICAL.get(param)


def build_parameter_bounds(model_name: str, bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    """Translate per-model bounds into an extract_metrics ``parameter_bounds`` dict.

    Amplitude and onset bounds are dropped (they are not kinetics). Keys that
    feed the search grids are sanitised: strictly positive, lower < upper,
    because extract_metrics rejects anything else outright.
    """
    out: Dict[str, Tuple[float, float]] = {}
    for param, (lo, hi) in bounds.items():
        if param in ("t_onset", "t_rise", "t_decay") or is_amplitude(param):
            continue
        if lo is None or hi is None:
            continue
        lo, hi = float(lo), float(hi)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            continue
        key = canonical_bound_key(param) or param
        if key in GRID_BOUND_KEYS:
            if lo <= 0:
                lo = max(hi * 1e-3, 1e-6)
            if hi <= lo:
                hi = lo * 1.05 + 1e-9
        elif hi < lo:
            lo, hi = hi, lo
        # A later, more specific parameter should not clobber an earlier one.
        if key in out:
            prev_lo, prev_hi = out[key]
            lo, hi = min(prev_lo, lo), max(prev_hi, hi)
        out[key] = (lo, hi)
    return out
