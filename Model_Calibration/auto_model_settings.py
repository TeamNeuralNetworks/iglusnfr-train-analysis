"""
Auto-calibration of event-model settings from multi-trial responses.

Given time, trials and train info, this selects the best event model from the
library (cooperative | double_exp) by fitting the average response in a short
window after the first stimulus and returns recommended settings that can guide
downstream pipelines (e.g., extract_metrics options).

This avoids redundancy with demos by reusing the event_models registry.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.optimize import curve_fit


def _resolve_models():
    try:
        from Model_Calibration.event_models import get_models
    except Exception:
        from event_models import get_models  # type: ignore
    return get_models


def _baseline_subtract(time: np.ndarray, trials: np.ndarray, train_start: float) -> np.ndarray:
    t = np.asarray(time, float).reshape(-1)
    X = np.asarray(trials, float)
    if X.ndim == 1:
        X = X[:, None]
    pre = t < float(train_start)
    if not np.any(pre):
        return X - np.nanmedian(X, axis=0, keepdims=True)
    F0 = np.nanmedian(X[pre, :], axis=0)
    return X - F0[None, :]


def auto_select_event_model_settings(
    time: np.ndarray,
    trials: np.ndarray,
    *,
    train_start: float,
    isi: float,
    n_pulses: int,
    candidates: Tuple[str, ...] = ("double_exp", "cooperative"),
    window_ms: Tuple[float, float] = (0.0, 30.0),
) -> Dict:
    """Return recommended settings from multi-trial data.

    Parameters
    ----------
    time : (N,) array
        Time in seconds.
    trials : (N,M) array
        Trials as columns.
    train_start : float
        Train start (s).
    isi : float
        Inter-stimulus interval (s). Not used directly here, reserved for future.
    n_pulses : int
        Number of pulses. Not used directly here, reserved for future.
    candidates : tuple[str]
        Candidate model names to consider from the library.
    window_ms : tuple[float, float]
        Fit window in milliseconds relative to train_start.

    Returns
    -------
    dict with keys:
      - event_model: 'cooperative' | 'double_exp'
      - event_model_settings: dict of parameter overrides (e.g. {'n_coop': 2.0})
      - metrics: per-model metrics dict
      - fit_params: per-model parameter arrays
      - options: suggested options dict for extract_metrics
    """
    t = np.asarray(time, float).reshape(-1)
    X = _baseline_subtract(t, trials, float(train_start))
    y_avg = np.nanmean(np.asarray(X, float), axis=1)

    # Time in ms relative to train start
    t_ms = (t - float(train_start)) * 1000.0
    lo, hi = float(window_ms[0]), float(window_ms[1])
    mfit = (t_ms >= lo) & (t_ms <= hi)
    if not np.any(mfit):
        raise ValueError("auto_select_event_model_settings: empty fit window")
    t_fit = t_ms[mfit]
    y_fit = y_avg[mfit]

    get_models = _resolve_models()
    models = get_models(list(candidates))

    metrics: Dict[str, Dict] = {}
    params_map: Dict[str, np.ndarray] = {}
    best_name, best_aic = None, np.inf

    for name, spec in models.items():
        try:
            p0 = spec['p0_func'](y_fit, t_fit)
            popt, pcov = curve_fit(
                spec['func'], t_fit, y_fit,
                p0=p0, bounds=spec['bounds'], maxfev=3000
            )
            y_pred = spec['func'](t_fit, *popt)
            resid = y_fit - y_pred
            n = len(y_fit); k = len(popt)
            mse = float(np.mean(resid**2)) if n else np.inf
            if mse > 0 and np.isfinite(mse):
                logL = -0.5 * n * np.log(2 * np.pi * mse) - 0.5 * np.sum(resid**2) / mse
                aic = 2 * k - 2 * logL
            else:
                aic = np.inf
            ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
            r2 = 1 - np.sum(resid**2) / ss_tot if ss_tot > 0 else 0.0
            metrics[name] = {'aic': float(aic), 'r2': float(r2)}
            params_map[name] = popt
            if aic < best_aic:
                best_aic = aic; best_name = name
        except Exception:
            metrics[name] = {'aic': float('inf'), 'r2': float('-inf')}
            continue

    if best_name is None:
        # Fallback to cooperative
        best_name = 'cooperative'
        metrics.setdefault('cooperative', {'aic': float('inf'), 'r2': float('-inf')})

    coop_n = None
    if best_name == 'cooperative':
        # Extract n_coop from fitted params if available
        try:
            spec = models[best_name]
            idx = spec['params'].index('n_coop')
            coop_n = float(params_map.get(best_name, [np.nan]*len(spec['params']))[idx])
        except Exception:
            coop_n = 2.0

    event_model_settings: Dict[str, float] = {}
    if best_name == 'cooperative':
        event_model_settings['n_coop'] = float(coop_n) if coop_n is not None else 2.0

    options = {
        'event_model': best_name,
        'event_model_settings': dict(event_model_settings),
        'measurement': 'NNLS',
        'fail_method': 'NNLS',
        'threshold_mode': 'auto',
        'allow_shift': True,
    }

    return {
        'event_model': best_name,
        'event_model_settings': dict(event_model_settings),
        'metrics': metrics,
        'fit_params': params_map,
        'options': options,
    }

