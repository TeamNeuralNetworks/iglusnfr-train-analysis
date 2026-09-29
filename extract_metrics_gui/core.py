"""Fitting engine behind the GUI.

Two loops with very different costs:

* the **inner loop** (`fit_event`) fits one library model to the average event
  snippet. It takes milliseconds, so it can run on every slider move;
* the **outer loop** (`run_train_fit`) runs the full `extract_metrics` train
  extraction. It takes seconds, so the GUI runs it on a worker thread.

`build_average_event` bridges them: it recuts the event snippets directly from
the loaded traces, so the inner loop is usable before any train fit has run.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.optimize import curve_fit

from Feature_extraction.extract_metrics import extract_metrics

from . import registry
from .loaders import TraceData
from .theme import style_axes


# --- inner loop: the average event and one model fitted to it ---------------

@dataclass
class AverageEvent:
    """Event snippets recut from the raw traces, plus their projection."""
    t_ms: np.ndarray            # relative time grid, milliseconds
    average: np.ndarray
    snippets: np.ndarray        # (n_snippets, n_samples)
    n_used: int
    projection: str

    @property
    def ok(self) -> bool:
        return self.t_ms.size > 4 and np.isfinite(self.average).sum() > 4


def build_average_event(data: TraceData, *, train_start: float, isi: float, n_pulses: int,
                        oversample: int = 8, projection: str = "median",
                        pre_ms: float = 5.0, max_pulses: Optional[int] = None) -> AverageEvent:
    """Recut every stimulus-locked snippet onto a common oversampled grid."""
    t = np.asarray(data.time_s, float)
    trials = np.asarray(data.trials, float)
    if trials.ndim == 1:
        trials = trials[:, None]

    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.0
    if not np.isfinite(dt) or dt <= 0 or isi <= 0:
        return AverageEvent(np.array([]), np.array([]), np.zeros((0, 0)), 0, projection)

    pre_s = min(pre_ms / 1000.0, 0.4 * isi)
    post_s = isi
    step = dt / max(1, int(oversample))
    grid = np.arange(-pre_s, post_s, step)
    if grid.size < 5:
        return AverageEvent(np.array([]), np.array([]), np.zeros((0, 0)), 0, projection)

    limit = int(n_pulses if max_pulses is None else min(n_pulses, max_pulses))
    baseline_mask = grid < 0

    collected: List[np.ndarray] = []
    for pulse in range(max(1, limit)):
        stim = float(train_start) + pulse * float(isi)
        for col in range(trials.shape[1]):
            y = trials[:, col]
            finite = np.isfinite(y)
            if finite.sum() < 8:
                continue
            snippet = np.interp(stim + grid, t[finite], y[finite], left=np.nan, right=np.nan)
            if not np.isfinite(snippet).all():
                continue
            if baseline_mask.any():
                snippet = snippet - np.nanmedian(snippet[baseline_mask])
            collected.append(snippet)

    if not collected:
        return AverageEvent(np.array([]), np.array([]), np.zeros((0, 0)), 0, projection)

    stack = np.vstack(collected)
    if projection == "mean":
        average = np.nanmean(stack, axis=0)
    elif projection == "std":
        average = np.nanstd(stack, axis=0)
    else:
        average = np.nanmedian(stack, axis=0)

    return AverageEvent(
        t_ms=grid * 1000.0, average=np.asarray(average, float),
        snippets=stack, n_used=int(stack.shape[0]), projection=projection,
    )


@dataclass
class EventFit:
    """Result of fitting one library model to the average event."""
    model: str
    params: Dict[str, float]
    bounds: Dict[str, Tuple[float, float]]
    locked: Tuple[str, ...]
    t_ms: np.ndarray
    yhat: np.ndarray
    residual: np.ndarray
    r2: float
    rms: float
    aic: float
    pinned: Tuple[str, ...]
    converged: bool
    message: str = ""

    @property
    def quality(self) -> str:
        if not self.converged:
            return "bad"
        if self.r2 >= 0.98 and not self.pinned:
            return "good"
        if self.r2 >= 0.90:
            return "warn"
        return "bad"


def _pinned_params(values: Dict[str, float], bounds: Dict[str, Tuple[float, float]],
                   skip: Sequence[str] = ()) -> Tuple[str, ...]:
    out = []
    for name, value in values.items():
        if name in skip:
            continue
        lo, hi = bounds.get(name, (None, None))
        if lo is None or hi is None or not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            continue
        tol = 1e-9 + 0.01 * (hi - lo)
        if value <= lo + tol or value >= hi - tol:
            out.append(name)
    return tuple(out)


def fit_event(event: AverageEvent, model_name: str, *,
              start: Optional[Dict[str, float]] = None,
              bounds: Optional[Dict[str, Tuple[float, float]]] = None,
              locked: Sequence[str] = (),
              refit: bool = True) -> EventFit:
    """Fit `model_name` to the average event.

    With `refit=False` the model is only *evaluated* at `start`, which is what
    the GUI does while a slider is being dragged.
    """
    info = registry.model(model_name)
    t_ms = np.asarray(event.t_ms, float)
    y = np.asarray(event.average, float)
    finite = np.isfinite(t_ms) & np.isfinite(y)
    t_fit, y_fit = t_ms[finite], y[finite]

    guess = info.initial_guess(y_fit, t_fit) if t_fit.size > 4 else info.display_defaults()
    values = {name: float(v) for name, v in zip(info.params, guess)}
    if start:
        values.update({k: float(v) for k, v in start.items() if k in values})

    effective: Dict[str, Tuple[float, float]] = {}
    for name in info.params:
        lo, hi = info.bounds_for(name)
        if bounds and name in bounds and bounds[name][0] is not None and bounds[name][1] is not None:
            lo, hi = float(bounds[name][0]), float(bounds[name][1])
        effective[name] = (lo, hi)

    locked = tuple(name for name in locked if name in values)

    def evaluate(params: Dict[str, float]) -> np.ndarray:
        ordered = [params[name] for name in info.params]
        return np.asarray(info.func(t_fit, *ordered), float)

    converged, message = True, ""
    free = [name for name in info.params if name not in locked]

    if refit and t_fit.size > len(free) + 2 and free:
        def model_fn(_t, *free_values):
            merged = dict(values)
            merged.update(dict(zip(free, free_values)))
            ordered = [merged[name] for name in info.params]
            return np.asarray(info.func(_t, *ordered), float)

        p0, lower, upper = [], [], []
        for name in free:
            lo, hi = effective[name]
            p0.append(min(max(values[name], lo), hi if math.isfinite(hi) else values[name]))
            lower.append(lo)
            upper.append(hi)
        try:
            popt, _ = curve_fit(model_fn, t_fit, y_fit, p0=p0, bounds=(lower, upper), maxfev=12000)
            values.update({name: float(v) for name, v in zip(free, popt)})
        except Exception as exc:
            converged = False
            message = f"{type(exc).__name__}: {exc}"

    yhat = evaluate(values)
    residual = y_fit - yhat
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rms = float(np.sqrt(ss_res / residual.size)) if residual.size else float("nan")
    n, k = residual.size, len(free)
    aic = float(n * np.log(ss_res / n) + 2 * k) if n > k and ss_res > 0 else float("nan")

    return EventFit(
        model=info.name, params=values, bounds=effective, locked=locked,
        t_ms=t_fit, yhat=yhat, residual=residual, r2=r2, rms=rms, aic=aic,
        pinned=_pinned_params(values, effective, skip=locked),
        converged=converged, message=message,
    )


# --- outer loop: the full extract_metrics train fit -------------------------

@dataclass
class TrainConfig:
    source: str
    train_start: float
    isi: float
    n_pulses: int
    event_model: str
    options: Dict[str, Any] = field(default_factory=dict)
    parameter_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    time_column_hint: str = ""

    def to_json(self) -> str:
        payload = {
            "source": self.source,
            "train_start": self.train_start,
            "isi": self.isi,
            "n_pulses": self.n_pulses,
            "event_model": self.event_model,
            "time_column_hint": self.time_column_hint,
            "options": {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.options.items()},
            "parameter_bounds": {k: list(v) for k, v in self.parameter_bounds.items()},
        }
        return json.dumps(payload, indent=2, default=str)

    @staticmethod
    def from_json(text: str) -> Dict[str, Any]:
        return json.loads(text)


def build_options(config: TrainConfig) -> Dict[str, Any]:
    """Assemble the option dict handed to extract_metrics."""
    options: Dict[str, Any] = {k: v for k, v in config.options.items() if v is not None or k == "fail_method"}
    options = {k: v for k, v in options.items() if not (k == "fail_method" and not v)}
    options["event_model"] = str(config.event_model)
    options["plot"] = {"enabled": False}
    options["recut_snippets"] = True
    options.setdefault("recut_oversample", 8)
    if config.parameter_bounds:
        options["parameter_bounds"] = {k: tuple(v) for k, v in config.parameter_bounds.items()}

    # peak_window_ms must stay below the ISI or the fitter silently overrides it.
    isi_ms = float(config.isi) * 1000.0
    if "peak_window_ms" in options and options["peak_window_ms"]:
        options["peak_window_ms"] = min(float(options["peak_window_ms"]), max(1.0, isi_ms - 1.0))
    return options


@dataclass
class TrainResult:
    config: TrainConfig
    data: TraceData
    raw: Dict[str, Any]
    pulses: pd.DataFrame
    metrics: pd.DataFrame
    event_fit: Optional[EventFit] = None

    @property
    def amp1(self) -> float:
        if self.pulses.empty:
            return float("nan")
        return float(self.pulses.loc[0, "amp_corrected"])

    @property
    def train_r2(self) -> float:
        avg = self.raw.get("average", {})
        y = np.asarray(avg.get("y_avg", []), float)
        yhat = np.asarray(avg.get("yhat_avg", []), float)
        if y.size == 0 or y.size != yhat.size:
            return float("nan")
        finite = np.isfinite(y) & np.isfinite(yhat)
        if finite.sum() < 4:
            return float("nan")
        resid = y[finite] - yhat[finite]
        ss_tot = float(np.sum((y[finite] - np.mean(y[finite])) ** 2))
        return 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")


_MEASUREMENT_KEYS = {
    "SAVGOL": ("amp_savgol", "amp_savgol_corr"),
    "RAW": ("amp_raw", "amp_raw_corr"),
    "NNLS": ("amp_nnls", "amp_nnls_corr"),
}


def _series(avg: Dict[str, Any], key: str, n: int) -> np.ndarray:
    values = np.asarray(avg.get(key, []), float).ravel()
    out = np.full(n, np.nan)
    out[: min(n, values.size)] = values[: min(n, values.size)]
    return out


def _pulse_table(result: Dict[str, Any], measurement: str, n_pulses: int) -> pd.DataFrame:
    avg = result.get("average", {})
    uncorr_key, corr_key = _MEASUREMENT_KEYS.get(str(measurement).upper(), _MEASUREMENT_KEYS["NNLS"])
    uncorrected = _series(avg, uncorr_key, n_pulses)
    corrected = _series(avg, corr_key, n_pulses)
    if np.all(np.isnan(corrected)):
        corrected = uncorrected
    ppr = _series(avg, "ppr_nnls_corr", n_pulses)
    if np.all(np.isnan(ppr)) and np.isfinite(corrected[0]) and corrected[0] != 0:
        ppr = corrected / corrected[0]
    tau_d = np.asarray(result.get("tau_d_s", []), float).ravel()
    taus = np.full(n_pulses, np.nan)
    taus[: min(n_pulses, tau_d.size)] = tau_d[: min(n_pulses, tau_d.size)] * 1000.0

    return pd.DataFrame({
        "pulse": np.arange(1, n_pulses + 1),
        "amp_corrected": corrected,
        "amp_uncorrected": uncorrected,
        "ppr": ppr,
        "tau_decay_ms": taus,
    })


def _metrics_table(result: Dict[str, Any], data: TraceData, config: TrainConfig,
                   pulses: pd.DataFrame) -> pd.DataFrame:
    threshold = np.asarray(result.get("threshold_amp1", []), float)
    threshold = threshold[np.isfinite(threshold)]
    model_name = result.get("event_model", {})
    if isinstance(model_name, dict):
        model_name = model_name.get("name", config.event_model)

    rows: List[Tuple[str, str]] = [
        ("file", Path(config.source).name),
        ("traces", str(data.n_trials)),
        ("samples", str(data.time_s.size)),
        ("duration (s)", f"{data.duration_s:.4f}"),
        ("train start (s)", f"{config.train_start:.5g}"),
        ("ISI (s)", f"{config.isi:.5g}"),
        ("pulses", str(config.n_pulses)),
        ("event model", str(model_name)),
        ("measurement", str(config.options.get("measurement", "NNLS"))),
        ("tau rise (ms)", f"{1000.0 * float(result.get('tau_r_s', np.nan)):.4g}"),
        ("median A1 threshold", f"{float(np.nanmedian(threshold)):.6g}" if threshold.size else "n/a"),
    ]
    for _, row in pulses.iterrows():
        rows.append((f"AMP{int(row['pulse'])}", f"{row['amp_corrected']:.6g}"))
    for _, row in pulses.iloc[1:].iterrows():
        rows.append((f"PPR{int(row['pulse'])}/1", f"{row['ppr']:.6g}"))
    return pd.DataFrame(rows, columns=["metric", "value"])


def run_train_fit(data: TraceData, config: TrainConfig, *,
                  progress: Optional[Callable[[str], None]] = None) -> TrainResult:
    """Run the full extraction. Safe to call from a worker thread."""
    options = build_options(config)
    if progress:
        progress(f"Fitting {config.n_pulses} pulses with '{config.event_model}'...")

    result = extract_metrics(
        time=data.time_s,
        trials=data.trials,
        train_start=float(config.train_start),
        isi=float(config.isi),
        n_pulses=int(config.n_pulses),
        options=copy.deepcopy(options),
        filename=Path(config.source).stem,
    )

    measurement = str(options.get("measurement", "NNLS"))
    pulses = _pulse_table(result, measurement, int(config.n_pulses))
    metrics = _metrics_table(result, data, config, pulses)
    return TrainResult(config=config, data=data, raw=result, pulses=pulses, metrics=metrics)


# --- figures ----------------------------------------------------------------

def _annotate(ax, text: str, pal: Dict[str, str], *, loc: str = "upper right") -> None:
    x, ha = (0.985, "right") if "right" in loc else (0.015, "left")
    y, va = (0.96, "top") if "upper" in loc else (0.04, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=8,
            color=pal["text_dim"], family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=pal["panel"],
                      edgecolor=pal["border"], alpha=0.9))


def build_event_figure(event: Optional[AverageEvent], fit: Optional[EventFit],
                       pal: Dict[str, str], *, show_snippets: bool = True) -> Figure:
    """Average event, the model overlay and the residual strip (inner loop)."""
    figure = Figure(figsize=(7.6, 5.4), dpi=100)
    figure.patch.set_facecolor(pal["bg"])
    grid = figure.add_gridspec(2, 1, height_ratios=[3.1, 1.0], hspace=0.09)
    ax = figure.add_subplot(grid[0])
    ax_res = figure.add_subplot(grid[1], sharex=ax)
    style_axes(ax, pal)
    style_axes(ax_res, pal)

    ax.set_ylabel("Event amplitude")
    ax_res.set_ylabel("Residual")
    ax_res.set_xlabel("Time from stimulus (ms)")
    ax.tick_params(labelbottom=False)

    if event is None or not event.ok:
        ax.text(0.5, 0.5, "Load a file and set the train geometry\nto see the average event.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color=pal["text_dim"])
        return figure

    if show_snippets and event.snippets.size:
        step = max(1, event.snippets.shape[0] // 120)
        for snippet in event.snippets[::step]:
            ax.plot(event.t_ms, snippet, color=pal["snippet"], lw=0.5, alpha=0.28, zorder=1)

    ax.plot(event.t_ms, event.average, color=pal["trace"], lw=2.1,
            label=f"{event.projection} of {event.n_used}", zorder=3)
    ax.axvline(0.0, color=pal["text_dim"], ls=":", lw=0.9, alpha=0.7)
    ax_res.axhline(0.0, color=pal["text_dim"], ls=":", lw=0.9, alpha=0.7)

    if fit is not None:
        ax.plot(fit.t_ms, fit.yhat, color=pal["model"], lw=2.0, ls="--",
                label=f"{fit.model} fit", zorder=4)
        ax_res.plot(fit.t_ms, fit.residual, color=pal["residual"], lw=1.0)
        ax_res.fill_between(fit.t_ms, fit.residual, 0, color=pal["residual"], alpha=0.22)

        lines = [f"R2   {fit.r2:.4f}", f"RMS  {fit.rms:.4g}"]
        if np.isfinite(fit.aic):
            lines.append(f"AIC  {fit.aic:.1f}")
        _annotate(ax, "\n".join(lines), pal)
        if fit.pinned:
            ax.text(0.985, 0.03, "at bounds: " + ", ".join(fit.pinned), transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color=pal["warn"],
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=pal["panel"],
                              edgecolor=pal["warn"], alpha=0.95))
        if not fit.converged:
            ax.text(0.5, 0.5, "fit did not converge", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color=pal["bad"], alpha=0.75)

    ax.legend(frameon=False, loc="upper left", fontsize=8, labelcolor=pal["text_dim"])
    ax.set_xlim(float(event.t_ms.min()), float(event.t_ms.max()))
    figure.subplots_adjust(left=0.11, right=0.98, top=0.97, bottom=0.11)
    return figure


def build_train_figure(result: Optional[TrainResult], pal: Dict[str, str]) -> Figure:
    """Average trace, fitted model and residual over the whole train."""
    figure = Figure(figsize=(11.0, 5.6), dpi=100)
    figure.patch.set_facecolor(pal["bg"])
    grid = figure.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.09)
    ax = figure.add_subplot(grid[0])
    ax_res = figure.add_subplot(grid[1], sharex=ax)
    style_axes(ax, pal)
    style_axes(ax_res, pal)
    ax.set_ylabel("Signal")
    ax_res.set_ylabel("Residual")
    ax_res.set_xlabel("Time (s)")
    ax.tick_params(labelbottom=False)

    if result is None:
        ax.text(0.5, 0.5, "Run the train fit to see the fitted trace.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color=pal["text_dim"])
        return figure

    avg = result.raw.get("average", {})
    time_s = np.asarray(result.raw.get("time_s", []), float)
    y = np.asarray(avg.get("y_avg", []), float)
    yhat = np.asarray(avg.get("yhat_avg", []), float)
    model_t = np.asarray(avg.get("model_time_s", time_s), float)
    model_y = np.asarray(avg.get("yhat_avg_oversampled", yhat), float)
    stims = np.asarray(result.raw.get("stim_times_s", []), float)

    ax.plot(time_s, y, color=pal["trace"], lw=1.7, label="Average trace", zorder=3)
    if model_t.size == model_y.size and model_y.size:
        ax.plot(model_t, model_y, color=pal["model"], lw=1.7, label="Model fit", zorder=4)

    if y.size == yhat.size and y.size:
        ax_res.plot(time_s, y - yhat, color=pal["residual"], lw=0.9)
        ax_res.fill_between(time_s, y - yhat, 0, color=pal["residual"], alpha=0.22)
    ax_res.axhline(0.0, color=pal["text_dim"], ls=":", lw=0.9, alpha=0.7)

    top = ax.get_ylim()[1]
    for index, stim in enumerate(stims, start=1):
        ax.axvline(float(stim), color=pal["text_dim"], ls="--", lw=0.7, alpha=0.55, zorder=1)
        ax.text(float(stim), top, str(index), color=pal["text_dim"], fontsize=7,
                ha="center", va="bottom")
    if stims.size:
        pre = float(result.config.options.get("pre_zoom_s", 0.15) or 0.15)
        post = float(result.config.options.get("post_zoom_s", 0.6) or 0.6)
        ax.set_xlim(float(stims[0]) - pre, float(stims[-1]) + post)

    r2 = result.train_r2
    if np.isfinite(r2):
        _annotate(ax, f"train R2  {r2:.4f}", pal)
    ax.legend(frameon=False, loc="upper left", fontsize=8, labelcolor=pal["text_dim"])
    figure.subplots_adjust(left=0.07, right=0.985, top=0.96, bottom=0.11)
    return figure


def build_amplitude_figure(result: Optional[TrainResult], pal: Dict[str, str]) -> Figure:
    """Per-pulse amplitude, PPR and decay constant, on separate axes.

    Amplitude and PPR are proportional by construction, so drawing them on twin
    axes makes one curve hide the other exactly; they get their own panels.
    """
    figure = Figure(figsize=(11.0, 5.6), dpi=100)
    figure.patch.set_facecolor(pal["bg"])
    grid = figure.add_gridspec(1, 3, wspace=0.32)
    ax_amp = figure.add_subplot(grid[0])
    ax_ppr = figure.add_subplot(grid[1])
    ax_tau = figure.add_subplot(grid[2])
    for ax in (ax_amp, ax_ppr, ax_tau):
        style_axes(ax, pal)

    ax_amp.set_xlabel("Pulse")
    ax_amp.set_ylabel("Amplitude")
    ax_ppr.set_xlabel("Pulse")
    ax_ppr.set_ylabel("Ratio to pulse 1")
    ax_tau.set_xlabel("Pulse")
    ax_tau.set_ylabel("Decay tau (ms)")

    if result is None or result.pulses.empty:
        for ax, title in ((ax_amp, "Amplitude"), (ax_ppr, "PPR"), (ax_tau, "Decay")):
            ax.text(0.5, 0.5, "No fit yet", transform=ax.transAxes, ha="center",
                    va="center", fontsize=9, color=pal["text_dim"])
            ax.set_title(title, fontsize=10, loc="left", color=pal["text"])
        return figure

    pulses = result.pulses
    x = pulses["pulse"].to_numpy(float)

    ax_amp.bar(x, pulses["amp_uncorrected"].to_numpy(float), width=0.62,
               color=pal["snippet"], alpha=0.45, label="uncorrected", zorder=2)
    ax_amp.plot(x, pulses["amp_corrected"].to_numpy(float), color=pal["good"],
                marker="o", ms=4.5, lw=1.8, label="corrected", zorder=3)
    ax_amp.legend(frameon=False, fontsize=8, labelcolor=pal["text_dim"])

    ax_ppr.axhline(1.0, color=pal["text_dim"], ls=":", lw=0.9, alpha=0.7)
    ax_ppr.plot(x, pulses["ppr"].to_numpy(float), color=pal["model"],
                marker="s", ms=4.0, lw=1.7)

    taus = pulses["tau_decay_ms"].to_numpy(float)
    if np.isfinite(taus).any():
        ax_tau.plot(x, taus, color=pal["trace"], marker="^", ms=4.0, lw=1.7)
        # An almost-constant tau otherwise renders as an unreadable offset axis.
        ax_tau.ticklabel_format(axis="y", useOffset=False, style="plain")
        lo, hi = float(np.nanmin(taus)), float(np.nanmax(taus))
        if hi - lo < max(1e-3, 0.01 * abs(hi)):
            pad = max(0.05 * abs(hi), 0.5)
            ax_tau.set_ylim(lo - pad, hi + pad)
    else:
        ax_tau.text(0.5, 0.5, "not reported", transform=ax_tau.transAxes, ha="center",
                    va="center", fontsize=9, color=pal["text_dim"])

    for ax in (ax_amp, ax_ppr, ax_tau):
        ax.set_xticks(x)
        ax.tick_params(labelsize=8)

    figure.subplots_adjust(left=0.06, right=0.985, top=0.95, bottom=0.13)
    return figure


# --- export -----------------------------------------------------------------

def export_results(result: TrainResult, directory: str, *, stem: Optional[str] = None) -> List[Path]:
    """Write the pulse table, the metric list and the exact configuration."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = stem or Path(result.config.source).stem
    written: List[Path] = []

    pulses_path = out_dir / f"{base}_pulses.csv"
    result.pulses.to_csv(pulses_path, index=False)
    written.append(pulses_path)

    metrics_path = out_dir / f"{base}_metrics.csv"
    result.metrics.to_csv(metrics_path, index=False)
    written.append(metrics_path)

    config_path = out_dir / f"{base}_config.json"
    config_path.write_text(result.config.to_json(), encoding="utf-8")
    written.append(config_path)

    return written
