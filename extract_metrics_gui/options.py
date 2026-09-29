"""Declarative schema for the ``extract_metrics`` option surface.

Every option the fitter actually reads is described once here - type, default,
choices and a one-line explanation - so the GUI can build its panels, tooltips
and preset menu without duplicating knowledge. Defaults track
``Feature_extraction.extract_metrics.DEFAULTS``; the explanations follow the
annotated preset in ``Feature_extraction/demo_batch_process.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from Feature_extraction.extract_metrics import DEFAULTS


@dataclass(frozen=True)
class OptionSpec:
    key: str
    label: str
    kind: str  # 'bool' | 'choice' | 'float' | 'int' | 'special'
    group: str
    default: Any = None
    choices: Sequence[str] = field(default_factory=tuple)
    help: str = ""
    advanced: bool = False

    def coerce(self, raw: Any) -> Any:
        """Turn a widget string into the value extract_metrics expects."""
        if self.kind == "bool":
            return bool(raw)
        text = "" if raw is None else str(raw).strip()
        if self.kind == "choice":
            return text
        if text == "" or text.lower() in ("none", "null"):
            return None
        if self.kind == "int":
            return int(float(text))
        if self.kind == "float":
            return float(text)
        # 'special': accepts a keyword or a number (e.g. 'auto' / 'best' / 0.05)
        low = text.lower()
        if low in ("auto", "best"):
            return low
        return float(text)

    def to_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)


def _d(key: str, fallback: Any = None) -> Any:
    return DEFAULTS.get(key, fallback)


GROUPS: Tuple[str, ...] = (
    "Preprocessing",
    "Kinetics",
    "Event snippets",
    "NNLS fitting",
    "Template variants",
    "Peak measurement",
    "Thresholds & failures",
    "Display window",
)


SPECS: Tuple[OptionSpec, ...] = (
    # --- Preprocessing ------------------------------------------------------
    OptionSpec("bleach", "Bleach correction", "bool", "Preprocessing", True,
               help="Fit and remove a photobleaching decay before measuring events."),
    OptionSpec("normalize_dff", "Normalise to dF/F0", "bool", "Preprocessing", True,
               help="Express traces as dF/F0 using the pre-train baseline."),
    OptionSpec("f0_window_s", "F0 window", "float", "Preprocessing", _d("f0_window_s", 0.4),
               help="Length of the pre-train window (s) averaged to obtain F0."),
    OptionSpec("sg_window", "Savitzky-Golay window", "int", "Preprocessing", _d("sg_window", 9),
               help="Sample count of the Savitzky-Golay smoothing window. Must be odd.", advanced=True),
    OptionSpec("sg_poly", "Savitzky-Golay order", "int", "Preprocessing", _d("sg_poly", 2),
               help="Polynomial order of the Savitzky-Golay filter.", advanced=True),
    OptionSpec("bleach_huber_delta", "Bleach Huber delta", "float", "Preprocessing",
               _d("bleach_huber_delta", 3.0),
               help="Robustness of the bleach fit, in noise standard deviations.", advanced=True),
    OptionSpec("bleach_n_tau", "Bleach tau candidates", "int", "Preprocessing", _d("bleach_n_tau", 25),
               help="Number of bleach time constants tried during the search.", advanced=True),

    # --- Kinetics -----------------------------------------------------------
    OptionSpec("fit_source", "Kinetics source", "choice", "Kinetics", _d("fit_source", "global"),
               choices=("global", "average", "individual"),
               help="Where the event kinetics come from. 'global' fits one template from all trials "
                    "(recut median), 'average' fits the multi-trial average, 'individual' fits each "
                    "trial then takes the median."),
    OptionSpec("decay_progression_mode", "Decay progression", "choice", "Kinetics",
               _d("decay_progression_mode", "linear"),
               choices=("linear", "fixed", "free_monotonic", "none"),
               help="How the decay constant is allowed to evolve across the train. 'fixed' uses one "
                    "value throughout, 'linear' fits a non-negative slope, 'free_monotonic' "
                    "interpolates between the first and last estimate."),
    OptionSpec("anchor_first_tau", "Anchor first tau", "bool", "Kinetics", _d("anchor_first_tau", False),
               help="Treat the first event's tau as the fastest decay in the train."),
    OptionSpec("anchor_final_tau", "Anchor final tau", "bool", "Kinetics", _d("anchor_final_tau", True),
               help="Treat the last event's tau as the slowest decay. The last event has no following "
                    "event, so its estimate is usually the most reliable."),
    OptionSpec("early_events_only", "Fit kinetics on first N events", "int", "Kinetics",
               _d("early_events_only", 0),
               help="Restrict the kinetics fit to the first N events (0 uses all of them). Useful when "
                    "later events are contaminated by accumulation."),

    # --- Event snippets (recut) --------------------------------------------
    OptionSpec("recut_projection", "Snippet projection", "choice", "Event snippets",
               _d("recut_projection", "median"), choices=("median", "mean", "std"),
               help="How individual event snippets are combined into the average event."),
    OptionSpec("recut_oversample", "Snippet oversampling", "int", "Event snippets",
               _d("recut_oversample", 1),
               help="Factor by which snippets are projected onto a finer time grid before averaging. "
                    "Higher values sharpen the event fit but cost time."),
    OptionSpec("recut_peak_recenter", "Recentre on peak", "int", "Event snippets",
               _d("recut_peak_recenter", 0),
               help="Re-align each snippet on its own peak before averaging (0 disables).", advanced=True),
    OptionSpec("onset_method", "Onset detection", "choice", "Event snippets",
               _d("onset_method", "inflection"),
               choices=("inflection", "baseline_threshold", "none"),
               help="How the event onset is located inside each snippet. 'baseline_threshold' is more "
                    "robust for high-frequency trains with a contaminated baseline."),
    OptionSpec("onset_baseline_threshold", "Onset threshold", "float", "Event snippets",
               _d("onset_baseline_threshold", 0.15),
               help="Fraction of the peak used by the 'baseline_threshold' onset method (0.1-0.3)."),

    # --- NNLS fitting -------------------------------------------------------
    OptionSpec("nnls_fit_mode", "NNLS mode", "choice", "NNLS fitting", _d("nnls_fit_mode", "simultaneous"),
               choices=("simultaneous", "sequential"),
               help="'simultaneous' solves all events jointly. 'sequential' is a greedy forward pass "
                    "that subtracts each fitted event before the next, which resolves fast/super-slow "
                    "degeneracy in strongly overlapping trains."),
    OptionSpec("nnls_weight_mode", "Weighting", "choice", "NNLS fitting", _d("nnls_weight_mode", "uniform"),
               choices=("uniform", "linear", "exponential", "savgol", "peak"),
               help="Sample weighting used by the non-negative fit."),
    OptionSpec("nnls_weight_tau_s", "Weight tau", "float", "NNLS fitting", _d("nnls_weight_tau_s"),
               help="Time constant (s) for exponential weighting, or slope for linear. Blank = automatic.",
               advanced=True),
    OptionSpec("nnls_peak_window_s", "Peak-emphasis window", "float", "NNLS fitting",
               _d("nnls_peak_window_s", 0.010),
               help="Length (s) of the emphasised window after each stimulus.", advanced=True),
    OptionSpec("nnls_peak_weight", "Peak-emphasis weight", "float", "NNLS fitting",
               _d("nnls_peak_weight", 3.0),
               help="Weight multiplier applied inside the peak-emphasis window.", advanced=True),
    OptionSpec("huber_delta", "Huber delta", "float", "NNLS fitting", _d("huber_delta", 5.5),
               help="Robust-loss threshold in noise standard deviations. Blank disables robust fitting."),
    OptionSpec("irls_iters", "IRLS iterations", "int", "NNLS fitting", _d("irls_iters", 6),
               help="Reweighting iterations used by the robust fit.", advanced=True),
    OptionSpec("nnls_last_event_tail_tau_s", "Last-event tail tau", "special", "NNLS fitting",
               _d("nnls_last_event_tail_tau_s"),
               help="Down-weights the tail of the final event to prevent overshoot. Blank disables it, "
                    "'auto' uses the ISI, 'best' searches for the optimum, or give a value in seconds."),
    OptionSpec("nnls_two_pass_guard", "Two-pass guard", "bool", "NNLS fitting",
               _d("nnls_two_pass_guard", False),
               help="Reject the smoothing pass when it worsens the RMS residual.", advanced=True),
    OptionSpec("allow_shift", "Allow pulse micro-shift", "bool", "NNLS fitting", _d("allow_shift", True),
               help="Let each event slide by a fraction of a millisecond to absorb stimulus jitter."),
    OptionSpec("delta_max_s", "Max micro-shift", "float", "NNLS fitting", _d("delta_max_s", 0.002),
               help="Largest permitted per-event shift, in seconds.", advanced=True),

    # --- Template variants --------------------------------------------------
    OptionSpec("use_template_variants", "Use template variants", "bool", "Template variants",
               _d("use_template_variants", False),
               help="Screen a family of templates with different slow-component weights, letting the "
                    "data choose the mixture per event."),
    OptionSpec("template_variant_select", "Variant selection", "choice", "Template variants",
               _d("template_variant_select", "soft"), choices=("soft", "hard"),
               help="'soft' blends the candidate templates, 'hard' picks the single best."),
    OptionSpec("superslow_min_ratio", "Super-slow min ratio", "float", "Template variants",
               _d("superslow_min_ratio", 1.0),
               help="Disable super-slow variants when tau_superslow is below tau_slow times this ratio.",
               advanced=True),
    OptionSpec("allow_tau_slow_override", "Allow tau_slow override", "bool", "Template variants",
               _d("allow_tau_slow_override", True),
               help="Let the last-event decay replace the recut slow tau when it is shorter.",
               advanced=True),

    # --- Peak measurement ---------------------------------------------------
    OptionSpec("peak_window_ms", "Peak window", "float", "Peak measurement", _d("peak_window_ms", 25.0),
               help="Search window (ms) after each stimulus in which the peak is located. Automatically "
                    "capped to the ISI."),
    OptionSpec("peak_avg_points", "Peak average points", "int", "Peak measurement",
               _d("peak_avg_points", 5),
               help="Samples averaged around each peak to measure its amplitude. Values above 5 are "
                    "clamped to 3 by the fitter."),
    OptionSpec("pre_peak_ms", "Pre-peak baseline", "float", "Peak measurement", _d("pre_peak_ms", 0.0),
               help="Window (ms) before each peak used as its local baseline."),

    # --- Thresholds & failures ---------------------------------------------
    OptionSpec("measurement", "Measurement", "choice", "Thresholds & failures", _d("measurement", "NNLS"),
               choices=("NNLS", "SAVGOL", "RAW"),
               help="Which amplitude series drives p-values and failure classification."),
    OptionSpec("fail_method", "Failure method", "choice", "Thresholds & failures", "",
               choices=("", "NNLS", "SAVGOL", "RAW"),
               help="Amplitude series used to build the null distribution. Blank follows 'Measurement'."),
    OptionSpec("threshold_mode", "Threshold mode", "choice", "Thresholds & failures",
               _d("threshold_mode", "auto"), choices=("auto", "mad", "sd"),
               help="'auto' uses MAD for an NNLS null and SD for SAVGOL/RAW."),
    OptionSpec("null_N", "Null multiplier", "float", "Thresholds & failures", _d("null_N", 3.0),
               help="Multiplier applied to the null spread to set the detection threshold."),
    OptionSpec("null_sim_max_points", "Null sample cap", "int", "Thresholds & failures",
               _d("null_sim_max_points", 1000),
               help="Maximum number of samples drawn for the null distribution.", advanced=True),
    OptionSpec("amplitude_floor_to_noise", "Floor amplitudes to noise", "bool", "Thresholds & failures",
               _d("amplitude_floor_to_noise", False),
               help="Clamp per-trial amplitudes below threshold before computing PPR, which prevents "
                    "division by near-zero values."),

    # --- Display window -----------------------------------------------------
    OptionSpec("pre_zoom_s", "Pre-event window", "float", "Display window", _d("pre_zoom_s", 0.15),
               help="Time (s) shown and snipped before each event. Does not affect fitting."),
    OptionSpec("post_zoom_s", "Post-event window", "float", "Display window", _d("post_zoom_s", 0.60),
               help="Time (s) shown and snipped after each event. Does not affect fitting."),
)


BY_KEY: Dict[str, OptionSpec] = {spec.key: spec for spec in SPECS}


def specs_in_group(group: str, *, advanced: bool = True) -> List[OptionSpec]:
    return [s for s in SPECS if s.group == group and (advanced or not s.advanced)]


# --- presets ----------------------------------------------------------------

PRESETS: Dict[str, Dict[str, Any]] = {
    "extract_metrics defaults": {},
    "iGluSnFR optimised": {
        "event_model": "iglusnfr_tri",
        "fit_source": "global",
        "decay_progression_mode": "linear",
        "anchor_final_tau": False,
        "anchor_first_tau": False,
        "recut_projection": "mean",
        "recut_oversample": 20,
        "onset_method": "baseline_threshold",
        "onset_baseline_threshold": 0.10,
        "amplitude_floor_to_noise": True,
        "nnls_fit_mode": "sequential",
        "nnls_weight_mode": "savgol",
        "nnls_peak_window_s": 0.010,
        "nnls_peak_weight": 3.0,
        "huber_delta": 2.5,
        "irls_iters": 20,
        "nnls_last_event_tail_tau_s": "best",
        "nnls_two_pass_guard": False,
        "f0_window_s": 1.0,
        "peak_avg_points": 3,
        "pre_peak_ms": 1.0,
        "measurement": "SAVGOL",
        "fail_method": "SAVGOL",
        "threshold_mode": "auto",
        "null_N": 1.0,
        "use_template_variants": True,
        "template_variant_select": "soft",
        # Mirrors demo_batch_process.py, which bounds only the fast and slow
        # decays; the super-slow component keeps the registry's own range.
        "parameter_bounds": {
            "tau_rise": (0.0005, 0.003),
            "tau_decay_fast": (0.003, 0.008),
            "tau_decay_slow": (0.008, 0.035),
        },
    },
    "double-exp simple": {
        "event_model": "double_exp",
        "fit_source": "global",
        "decay_progression_mode": "linear",
        "measurement": "NNLS",
        "nnls_fit_mode": "simultaneous",
        "nnls_weight_mode": "uniform",
        "use_template_variants": False,
        "recut_oversample": 4,
        "parameter_bounds": {
            "tau_rise": (0.0005, 0.003),
            "tau_decay_fast": (0.003, 0.030),
        },
    },
}


def preset_names() -> List[str]:
    return list(PRESETS)


def defaults() -> Dict[str, Any]:
    """Schema defaults as a plain dict."""
    return {spec.key: spec.default for spec in SPECS}
