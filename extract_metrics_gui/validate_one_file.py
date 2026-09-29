"""Headless check of the GUI's fitting layers on one synthetic file.

Exercises everything the interface depends on without opening a window:
loading, train detection, the inner-loop event fit for every registered model,
the outer-loop train fit, and the effect of parameter bounds.

    python -m extract_metrics_gui.validate_one_file
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "extract_metrics_gui"

import numpy as np
import pandas as pd

from . import core, loaders, registry
from .theme import palette


HERE = Path(__file__).resolve().parent
SAMPLE_DIR = HERE / "sample_data"
OUTPUT_DIR = HERE / "validation_output"
SAMPLE_CSV = SAMPLE_DIR / "validation_trace.csv"

TRAIN_START, ISI, N_PULSES = 0.5, 0.05, 10


def generate_validation_csv(path: Path) -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12)
    time_s = np.linspace(0.0, 1.35, 1351)
    stim_times = TRAIN_START + np.arange(N_PULSES) * ISI
    base_amps = np.array([1.00, 0.91, 0.86, 0.80, 0.77, 0.73, 0.69, 0.66, 0.63, 0.61], float)

    def kernel(dt: np.ndarray, tau_rise: float, tau_decay: float) -> np.ndarray:
        resp = np.exp(-dt / tau_decay) - np.exp(-dt / tau_rise)
        resp[dt < 0.0] = 0.0
        return resp

    traces = {}
    for trial_idx in range(5):
        y = np.full_like(time_s, 1.0)
        amp_scale = 0.18 + 0.012 * trial_idx
        decay_scale = 0.017 + 0.0015 * trial_idx
        for pulse_idx, stim_time in enumerate(stim_times):
            dt = time_s - stim_time
            amp = amp_scale * base_amps[pulse_idx] * (1.0 + rng.normal(0.0, 0.04))
            y += amp * kernel(dt, tau_rise=0.0028, tau_decay=decay_scale)
        y += rng.normal(0.0, 0.0025, size=time_s.size)
        traces[f"trace_{trial_idx + 1}"] = y

    frame = pd.DataFrame(traces)
    frame["tim"] = time_s
    frame.to_csv(path, index=False)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    generate_validation_csv(SAMPLE_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pal = palette("light")

    # 1. loading -------------------------------------------------------------
    data = loaders.load_traces(str(SAMPLE_CSV))
    _check(data.n_trials == 5, f"expected 5 traces, got {data.n_trials}")
    print(f"[load]   {loaders.describe(data)}")

    # 2. train detection -----------------------------------------------------
    guess = loaders.detect_train(data.time_s, data.trials)
    _check(guess is not None, "train detection returned nothing")
    _check(guess.n_pulses == N_PULSES, f"detected {guess.n_pulses} pulses, expected {N_PULSES}")
    _check(abs(guess.isi - ISI) < 0.1 * ISI, f"detected ISI {guess.isi:.4f}, expected {ISI}")
    print(f"[train]  {guess.message} (confidence {guess.confidence:.2f})")

    # 3. inner loop: every registered model ---------------------------------
    event = core.build_average_event(data, train_start=TRAIN_START, isi=ISI,
                                     n_pulses=N_PULSES, oversample=8)
    _check(event.ok, "average event could not be built")
    print(f"[event]  {event.n_used} snippets on a {event.t_ms.size}-sample grid")

    names = registry.model_names()
    _check(len(names) >= 15, f"only {len(names)} models resolved from the registry")
    rows = []
    for name in names:
        fit = core.fit_event(event, name)
        _check(np.isfinite(fit.r2), f"model '{name}' produced a non-finite R2")
        rows.append((name, fit.r2, len(fit.pinned)))
    best = max(rows, key=lambda r: r[1])
    print(f"[models] {len(names)} models fitted; best {best[0]} (R2={best[1]:.5f})")

    # 4. outer loop, and the effect of bounds -------------------------------
    base = {"bleach": False, "normalize_dff": False, "measurement": "NNLS",
            "recut_oversample": 4, "decay_progression_mode": "linear"}
    results = {}
    for label, bounds in (("narrow", (0.003, 0.010)), ("wide", (0.003, 0.030))):
        config = core.TrainConfig(
            source=str(SAMPLE_CSV), train_start=TRAIN_START, isi=ISI, n_pulses=N_PULSES,
            event_model="double_exp", options=dict(base),
            parameter_bounds={"tau_decay_fast": bounds},
        )
        result = core.run_train_fit(data, config)
        _check(len(result.pulses) == N_PULSES,
               f"expected {N_PULSES} pulse rows, got {len(result.pulses)}")
        _check(np.isfinite(result.pulses["amp_corrected"].to_numpy(float)[:3]).all(),
               "first pulses have non-finite amplitudes")
        results[label] = result
        print(f"[train]  tau_decay_fast={bounds}: AMP1={result.amp1:.6g} "
              f"R2={result.train_r2:.4f} tau_d1={result.pulses.loc[0, 'tau_decay_ms']:.2f} ms")

    _check(abs(results["narrow"].amp1 - results["wide"].amp1) > 1e-6,
           "parameter_bounds had no effect on the train fit - the GUI's main control is dead")
    _check(results["wide"].train_r2 > results["narrow"].train_r2,
           "widening the decay bound did not improve the fit")

    # 5. figures and export --------------------------------------------------
    result = results["wide"]
    fit = core.fit_event(event, "double_exp")
    for name, figure in (
        ("validation_event.png", core.build_event_figure(event, fit, pal)),
        ("validation_train.png", core.build_train_figure(result, pal)),
        ("validation_amplitudes.png", core.build_amplitude_figure(result, pal)),
    ):
        figure.savefig(OUTPUT_DIR / name, dpi=140, facecolor=figure.get_facecolor())
    core.build_event_figure(None, None, pal)  # empty states must not raise
    core.build_train_figure(None, pal)
    core.build_amplitude_figure(None, pal)

    written = core.export_results(result, str(OUTPUT_DIR), stem="validation")
    print(f"[export] {', '.join(p.name for p in written)}")
    print(f"[figures] written to {OUTPUT_DIR}")
    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
