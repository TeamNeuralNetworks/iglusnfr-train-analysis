"""Residual diagnostics demo across multiple event models.

This script reuses ``extract_metrics`` to fit the same recording with a
family of event models and compares the residual noise statistics. It is a
companion to ``demo_single_file.py`` but focuses on evaluating how well each
model captures fast kinetics and whether the residuals resemble Gaussian
noise without systematic peak-time bias.

Steps performed:
    1. Load the time vector (last column) and trial traces (all other columns)
       from an Excel sheet.
    2. For each event model listed in ``MODEL_CANDIDATES`` run
       ``extract_metrics`` with identical preprocessing options.
    3. Compute residual diagnostics on the average trace:
         • Jarque–Bera statistic, skewness, and excess kurtosis
           (lower values indicate a distribution closer to Gaussian).
         • Mean residual around each stimulus; positive values highlight
           models that leave a systematic positive bias near the peak time.
    4. Summarize the metrics in a table sorted by Gaussianity (Jarque–Bera).

Update ``TARGET_LEGACY_ID`` and ``OUT_DIR`` to match your data. The recording is
loaded from the well-organized raw stage (release/raw/<uid>.csv +
release/boutons_manifest.csv), resolved by legacy_id/uid via
dataset_tools.raw_loader - never from the private per-condition .xlsx tree. The
output CSV makes it easy to inspect or plot the metrics externally.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure the repository root is importable when running the script directly
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Feature_extraction.extract_metrics import extract_metrics
from dataset_tools import raw_loader


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = os.path.abspath(os.environ.get(
    "GLUSNFR_DATA_ROOT", os.path.join(os.path.expanduser("~"), "Desktop", "FINAL PUBLICATION TEST PROCESS")
))
TARGET_LEGACY_ID = "20211125_linescan1_20Hz_10pulses_4mMCa_bouton1_traces_converted"
TARGET_CONDITION = "Theo_4Ca"  # only used to disambiguate; set to None otherwise
OUT_DIR = os.path.join(DATA_ROOT, "Testout")

# Event models to compare. Add or remove entries as needed. Each tuple contains
# the event model name accepted by ``extract_metrics`` and optional keyword
# overrides (e.g. cooperative exponent).
MODEL_CANDIDATES: List[Tuple[str, Dict]] = [
    ("double_exp", {}),
    ("single_exp", {}),
    ("cooperative", {"event_model_settings": {"n_coop": 2.0}}),
    ("coop_plus_linear", {}),
    ("binding_kinetics", {}),
    ("two_component", {}),
]

# Shared options passed to ``extract_metrics`` for every model.
BASE_OPTIONS: Dict = {
    "normalize_dff": True,
    "bleach": True,
    "fit_source": "global",
    "decay_progression_mode": "free_monotonic",
    "plot": {"enabled": False},
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _finite(values: np.ndarray) -> np.ndarray:
    """Return only finite entries of ``values`` as a 1-D array."""

    arr = np.asarray(values, float).ravel()
    return arr[np.isfinite(arr)]


def jarque_bera(residuals: np.ndarray) -> Tuple[float, float, float]:
    """Compute Jarque–Bera statistic, skewness, and excess kurtosis.

    Returns ``(jb_stat, skewness, excess_kurtosis)``. ``jb_stat`` approaches
    zero when the residual distribution is close to Gaussian.
    """

    vals = _finite(residuals)
    if vals.size < 8:
        return float("nan"), float("nan"), float("nan")

    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1))
    if std <= 0:
        return float("nan"), float("nan"), float("nan")

    centered = vals - mean
    skew = float(np.mean(centered ** 3) / (std ** 3))
    kurtosis = float(np.mean(centered ** 4) / (std ** 4)) - 3.0
    jb = float((vals.size / 6.0) * (skew ** 2 + (kurtosis ** 2) / 4.0))
    return jb, skew, kurtosis


def peak_bias(
    time_s: np.ndarray,
    residuals: np.ndarray,
    stim_times_s: Iterable[float],
    window_ms: float = 4.0,
) -> Tuple[float, float, np.ndarray]:
    """Estimate residual bias around each stimulus.

    ``window_ms`` defines the symmetric window around ``stim_time`` that is
    averaged. Returns the overall mean residual, the maximum mean bias across
    pulses, and the per-pulse residual means.
    """

    time = np.asarray(time_s, float)
    resid = np.asarray(residuals, float)
    stim_times = np.asarray(list(stim_times_s), float)
    if time.size == 0 or resid.size != time.size or stim_times.size == 0:
        empty = np.full(stim_times.size if stim_times.size else 1, np.nan)
        return float("nan"), float("nan"), empty

    window_s = window_ms / 1000.0
    pulse_means: List[float] = []
    for stim in stim_times:
        mask = (time >= (stim - window_s)) & (time <= (stim + window_s))
        if not np.any(mask):
            pulse_means.append(float("nan"))
            continue
        pulse_means.append(float(np.mean(resid[mask])))

    pulse_arr = np.asarray(pulse_means, float)
    overall_mean = float(np.nanmean(pulse_arr)) if np.isfinite(pulse_arr).any() else float("nan")
    max_bias = float(np.nanmax(pulse_arr)) if np.isfinite(pulse_arr).any() else float("nan")
    return overall_mean, max_bias, pulse_arr


@dataclass
class ResidualDiagnostics:
    model: str
    jb_stat: float
    skewness: float
    excess_kurtosis: float
    mean_bias: float
    max_bias: float
    positive_bias_fraction: float
    pulse_bias: np.ndarray
    time_s: np.ndarray
    stim_times_s: np.ndarray
    y_avg: np.ndarray
    yhat_avg: np.ndarray
    residual: np.ndarray


def summarize_model(
    model_name: str,
    model_kwargs: Dict,
    *,
    time_s: np.ndarray,
    trials: np.ndarray,
    base_options: Dict,
    train_start: float,
    isi: float,
    n_pulses: int,
) -> ResidualDiagnostics:
    """Run ``extract_metrics`` for ``model_name`` and compute diagnostics."""

    options = dict(base_options)
    options.update(model_kwargs)
    options["event_model"] = model_name

    result = extract_metrics(
        time_s,
        trials,
        train_start=train_start,
        isi=isi,
        n_pulses=n_pulses,
        options=options,
    )

    y_avg = np.asarray(result["average"]["y_avg"], float)
    yhat_avg = np.asarray(result["average"]["yhat_avg"], float)
    residual = y_avg - yhat_avg

    jb_stat, skew, kurt = jarque_bera(residual)
    mean_bias, max_bias, pulse_bias = peak_bias(
        result["time_s"], residual, result["stim_times_s"]
    )
    finite_mask = np.isfinite(pulse_bias)
    pos_fraction = (
        float(np.mean(pulse_bias[finite_mask] > 0)) if finite_mask.any() else float("nan")
    )

    return ResidualDiagnostics(
        model=model_name,
        jb_stat=jb_stat,
        skewness=skew,
        excess_kurtosis=kurt,
        mean_bias=mean_bias,
        max_bias=max_bias,
        positive_bias_fraction=pos_fraction,
        pulse_bias=pulse_bias,
        time_s=np.asarray(result["time_s"], float),
        stim_times_s=np.asarray(result["stim_times_s"], float),
        y_avg=y_avg,
        yhat_avg=yhat_avg,
        residual=residual,
    )


def plot_model_diagnostics(diag: ResidualDiagnostics, out_dir: str) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(f"Residual diagnostics – {diag.model}")

    ax_trace = axes[0, 0]
    ax_trace.plot(diag.time_s, diag.y_avg, label="Average trace", color="tab:blue")
    ax_trace.plot(diag.time_s, diag.yhat_avg, label="Fit", color="tab:orange")
    for stim in diag.stim_times_s:
        ax_trace.axvline(stim, color="0.7", linestyle="--", linewidth=0.8)
    ax_trace.set_xlabel("Time (s)")
    ax_trace.set_ylabel("ΔF/F")
    ax_trace.legend(loc="upper right")
    ax_trace.set_title("Average trace vs model fit")

    ax_resid = axes[0, 1]
    ax_resid.plot(diag.time_s, diag.residual, color="tab:red")
    ax_resid.axhline(0.0, color="0.5", linestyle=":")
    for stim in diag.stim_times_s:
        ax_resid.axvline(stim, color="0.8", linestyle="--", linewidth=0.8)
    ax_resid.set_xlabel("Time (s)")
    ax_resid.set_ylabel("Residual ΔF/F")
    ax_resid.set_title("Residual trace")

    ax_hist = axes[1, 0]
    finite_resid = diag.residual[np.isfinite(diag.residual)]
    if finite_resid.size:
        ax_hist.hist(finite_resid, bins=50, density=True, color="tab:green", alpha=0.6)
        mean = float(np.mean(finite_resid))
        std = float(np.std(finite_resid, ddof=1))
        if std > 0:
            x_vals = np.linspace(mean - 4 * std, mean + 4 * std, 200)
            gauss = (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_vals - mean) / std) ** 2)
            ax_hist.plot(x_vals, gauss, "k--", label="Gaussian")
        ax_hist.legend(loc="upper right")
    ax_hist.set_xlabel("Residual ΔF/F")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Residual distribution")

    ax_bias = axes[1, 1]
    if diag.pulse_bias.size:
        ax_bias.bar(
            np.arange(1, diag.pulse_bias.size + 1),
            diag.pulse_bias,
            color="tab:purple",
        )
    ax_bias.axhline(0.0, color="0.5", linestyle=":")
    ax_bias.set_xlabel("Pulse index")
    ax_bias.set_ylabel("Mean residual ΔF/F")
    ax_bias.set_title("Residual bias per pulse")

    stats_text = (
        f"Jarque–Bera: {diag.jb_stat:.3g}\n"
        f"Skewness: {diag.skewness:.3g}\n"
        f"Excess kurtosis: {diag.excess_kurtosis:.3g}\n"
        f"Mean peak bias: {diag.mean_bias:.3g}\n"
        f"Max peak bias: {diag.max_bias:.3g}\n"
        f"Positive bias fraction: {diag.positive_bias_fraction:.2f}"
    )
    ax_bias.text(
        1.02,
        0.5,
        stats_text,
        transform=ax_bias.transAxes,
        va="center",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    filename = f"residual_diagnostics_{diag.model}.png"
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_comparison(diagnostics: List[ResidualDiagnostics], out_dir: str) -> str:
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True, constrained_layout=True)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for idx, diag in enumerate(diagnostics):
        color = color_cycle[idx % len(color_cycle)] if color_cycle else None
        axes[0].plot(diag.time_s, diag.y_avg, label=f"{diag.model} avg", linestyle=":", color=color)
        axes[0].plot(diag.time_s, diag.yhat_avg, label=f"{diag.model} fit", color=color)
        axes[1].plot(diag.time_s, diag.residual, label=diag.model, color=color)

        finite_resid = diag.residual[np.isfinite(diag.residual)]
        if finite_resid.size:
            axes[2].hist(
                finite_resid,
                bins=60,
                histtype="step",
                density=True,
                label=diag.model,
                color=color,
                alpha=0.9,
            )

    for stim in diagnostics[0].stim_times_s:
        for ax in axes[:2]:
            ax.axvline(stim, color="0.7", linestyle="--", linewidth=0.8)

    axes[0].set_ylabel("ΔF/F")
    axes[0].set_title("Average traces and fitted models")
    axes[0].legend(loc="upper right", ncol=2)

    axes[1].axhline(0.0, color="0.5", linestyle=":")
    axes[1].set_ylabel("Residual ΔF/F")
    axes[1].set_title("Residual traces")
    axes[1].legend(loc="upper right")

    axes[2].set_xlabel("Residual ΔF/F")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Residual distributions")
    axes[2].legend(loc="upper right")

    filename = "residual_diagnostics_comparison.png"
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    manifest = raw_loader.load_manifest(DATA_ROOT)
    row = raw_loader.resolve_recording(
        manifest, legacy_id=TARGET_LEGACY_ID, condition=TARGET_CONDITION,
    )
    raw = raw_loader.load_raw_csv(DATA_ROOT, row["uid"])
    time_raw = raw.time_s
    trials_raw = raw.values  # no average-column dropping here (matches prior behavior)

    valid = np.isfinite(time_raw)
    time_s = time_raw[valid]
    trials = trials_raw[valid, :]

    # Train configuration (adjust to your protocol)
    train_start = 0.5 - 0.001  # seconds
    isi = 0.05  # seconds
    n_pulses = 10

    diagnostics: List[ResidualDiagnostics] = []
    for model, kwargs in MODEL_CANDIDATES:
        print(f"Fitting model: {model}")
        diag = summarize_model(
            model,
            kwargs,
            time_s=time_s,
            trials=trials,
            base_options=BASE_OPTIONS,
            train_start=train_start,
            isi=isi,
            n_pulses=n_pulses,
        )
        diagnostics.append(diag)

    table = pd.DataFrame(
        [
            {
                "model": d.model,
                "jarque_bera": d.jb_stat,
                "skewness": d.skewness,
                "excess_kurtosis": d.excess_kurtosis,
                "mean_peak_bias": d.mean_bias,
                "max_peak_bias": d.max_bias,
                "positive_bias_fraction": d.positive_bias_fraction,
            }
            for d in diagnostics
        ]
    )
    table = table.sort_values(by="jarque_bera", key=lambda col: np.abs(col))

    out_path = os.path.join(OUT_DIR, "residual_diagnostics.csv")
    table.to_csv(out_path, index=False)

    figure_paths = [plot_model_diagnostics(diag, OUT_DIR) for diag in diagnostics]
    comparison_path = plot_comparison(diagnostics, OUT_DIR) if diagnostics else None

    print("\nResidual diagnostics (sorted by |Jarque–Bera|):")
    print(table.to_string(index=False, float_format=lambda x: f"{x:0.4g}"))
    print(f"\nSaved summary to: {out_path}")
    if figure_paths:
        print("Saved per-model figures:")
        for path in figure_paths:
            print(f" - {path}")
    if comparison_path:
        print(f"Saved comparison figure: {comparison_path}")


if __name__ == "__main__":
    main()

