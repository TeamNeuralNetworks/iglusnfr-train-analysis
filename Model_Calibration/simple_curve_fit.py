"""Shared utilities for fitting an average iGluSnFR event with simple models.

Provides a single helper so both the demo script and the extract_metrics
pipeline use the exact same curve_fit logic when estimating global kinetics.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit


def fit_average_event(
    time,
    data,
    model_name: str,
    stim_times_s: Optional[Sequence[float]] = None,
    *,
    window_ms: Tuple[float, float] = (0.0, 50.0),
    maxfev: int = 3000,
    pre_ms: float = 5.0,
    post_ms: float = 50.0,
    oversample: int = 1,
    projection: str = 'mean',
    peak_recenter=0,
    return_snippets: bool = False,
    onset_method: str = 'inflection',
    onset_baseline_threshold: float = 0.15,
    early_events_only: int = 0,
    fixed_tau_slow: Optional[float] = None,
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Optional[Tuple[Dict[str, float], np.ndarray, np.ndarray]]:
    """Recut trials, average, and fit an event model.

    Parameters
    ----------
    time : array-like
        Time vector (seconds if `stim_times_s` provided, otherwise milliseconds).
    data : array-like
        If `stim_times_s` is provided, shape should be `(T, N)` for `T` time
        points and `N` trials. Otherwise, `data` is the already averaged
        waveform aligned to `time`.
    model_name : str
        Name of the event model defined in module event_models.
    stim_times_s : sequence, optional
        Stimulus times in seconds. When given, `time` is interpreted in seconds
        and trials are recut around each stimulus before averaging (mean across
        all recut snippets).
    window_ms : tuple, optional
        (start, end) window in milliseconds used for fitting, by default
        (0, 50).
    maxfev : int, optional
        Maximum function evaluations passed to `curve_fit`.
    pre_ms, post_ms : float, optional
        Window around each stimulus used when recutting trials.
    peak_recenter : int | tuple | None, optional
        Maximum number of samples permitted when shifting snippet peaks prior to
        averaging; pass 0/None to keep stimulus-aligned windows.
    onset_method : str, optional
        Method for onset detection to exclude contaminated baseline:
        - 'inflection': Find inflection point (minimum derivative) and exclude points before it
        - 'baseline_threshold': Exclude all points below baseline + threshold * (peak - baseline)
        - 'none': No onset masking
        Default is 'inflection'.
    onset_baseline_threshold : float, optional
        Threshold factor for baseline_threshold method. Points below
        baseline + threshold * (peak - baseline) are excluded. Default is 0.15
        (15% of peak amplitude above baseline).
    early_events_only : int, optional
        If > 0, use only the first N events for computing the recut average.
        This helps capture fast kinetics uncontaminated by slow summation from
        later events. Default is 0 (use all events).
    fixed_tau_slow : float, optional
        If provided (in seconds), fixes tau_decay_slow to this value during
        curve_fit. This allows fitting tau_fast from early events while using
        a pre-estimated tau_slow (e.g., from post-train decay). Default is None.
    parameter_bounds : dict, optional
        Optional per-parameter bounds overrides. Keys should match model params
        (e.g., 'tau_decay_fast', 'tau_decay_slow'). Bounds use the same units
        as the model spec (seconds for taus, milliseconds for t_onset).

    Returns
    -------
    tuple | None
        `(params, t_ms, y_avg)` where `params` is a mapping of parameter
        names to fitted values and `t_ms`/`y_avg` are the averaged waveform.
        `None` if fitting fails.
    """
    try:
        if stim_times_s is not None:
            try:
                from smoothing import build_median_recut_waveform
            except Exception:  # pragma: no cover - allow running from subfolder
                import os, sys
                REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
                if REPO_ROOT not in sys.path:
                    sys.path.insert(0, REPO_ROOT)
                from smoothing import build_median_recut_waveform  # type: ignore
            res = build_median_recut_waveform(
                time,
                data,
                stim_times_s,
                pre_ms=pre_ms,
                post_ms=post_ms,
                peak_win_ms=25.0,
                peak_search_pre_ms=0.0,
                oversample=int(oversample),
                projection=str(projection).lower(),
                peak_recenter=peak_recenter,
                return_snippets=True,  # Always get snippets to support early_events_only
            )
            # Unpack - snippets always returned now
            t_rel_s, avg, snippets = res
            if t_rel_s is None or avg is None:
                return None
            
            # If early_events_only > 0, recompute average using only first N events
            # Snippets are ordered: [stim1_trial1, stim1_trial2, ..., stim2_trial1, ...]
            if early_events_only > 0 and snippets is not None and len(snippets) > 0:
                n_stims = len(stim_times_s)
                n_trials = len(snippets) // n_stims if n_stims > 0 else 0
                if n_trials > 0:
                    n_early = min(early_events_only, n_stims)
                    early_snip_indices = []
                    for ev_idx in range(n_early):
                        for tr_idx in range(n_trials):
                            early_snip_indices.append(ev_idx * n_trials + tr_idx)
                    early_snippets = [snippets[i] for i in early_snip_indices if i < len(snippets)]
                    if early_snippets:
                        S_early = np.array(early_snippets, dtype=float)
                        avg = np.nanmean(S_early, axis=0)
                        try:
                            from smoothing import progress_print
                            progress_print(f"[fit_average_event] Using only first {n_early} events ({len(early_snippets)} snippets) for fast kinetics fit")
                        except Exception:
                            pass
            
            # Store for return if caller wanted snippets
            if not return_snippets:
                snippets = None
            
            t_ms = np.asarray(t_rel_s, float) * 1000.0
            y_avg = np.asarray(avg, float)
        else:
            t_ms = np.asarray(time, float)
            y_avg = np.asarray(data, float)

        try:
            from .event_models import get_event_model
        except Exception:  # pragma: no cover - fallback when run from repo root
            from event_models import get_event_model  # type: ignore
        spec = get_event_model(model_name)
        if parameter_bounds:
            try:
                lb, ub = list(spec['bounds'][0]), list(spec['bounds'][1])
                for i, pname in enumerate(spec['params']):
                    if pname not in parameter_bounds:
                        continue
                    bound = parameter_bounds.get(pname)
                    if not isinstance(bound, (tuple, list)) or len(bound) != 2:
                        continue
                    lo, hi = float(bound[0]), float(bound[1])
                    if np.isfinite(lo):
                        lb[i] = lo
                    if np.isfinite(hi):
                        ub[i] = hi
                spec['bounds'] = (lb, ub)
            except Exception:
                pass
        mask = (t_ms >= window_ms[0]) & (t_ms <= window_ms[1])
        if not np.any(mask):
            return None
        tf = t_ms[mask]
        yf = y_avg[mask]

        # Onset detection to exclude contaminated baseline at high frequencies
        onset_idx = 0
        method = str(onset_method).lower()

        if method == 'inflection' and len(yf) > 5:
            # Find inflection point (minimum derivative) - where signal starts rising
            dy = np.gradient(yf, tf)
            # Smooth derivative to reduce noise
            from scipy.ndimage import uniform_filter1d
            dy_smooth = uniform_filter1d(dy, size=min(5, len(dy)))
            # Find first local minimum in derivative (where rise starts)
            # Look in first 30% of window where onset should occur
            search_len = max(5, int(len(dy_smooth) * 0.3))
            onset_idx = int(np.argmin(dy_smooth[:search_len]))
            # Validate: only exclude if there's actually a dip/notch (dy is negative)
            if dy_smooth[onset_idx] >= 0:
                onset_idx = 0  # No contamination detected, use full window

        elif method == 'baseline_threshold' and len(yf) > 5:
            # Exclude ALL points below baseline + threshold * (peak - baseline)
            # This is more aggressive than inflection detection
            # Estimate baseline from first few points (up to 20% of window or first 5 points)
            baseline_len = max(3, min(5, int(len(yf) * 0.2)))
            baseline = float(np.median(yf[:baseline_len]))
            # Find peak
            peak = float(np.nanmax(yf))
            # Compute threshold level as a fraction of the baseline-to-peak range
            frac = float(onset_baseline_threshold)
            if frac > 1.0:
                frac = frac / 100.0
            frac = max(0.0, min(1.0, frac))
            threshold_level = baseline + frac * (peak - baseline)
            # Ensure threshold stays within [min(baseline, peak), max(baseline, peak)]
            lo = min(baseline, peak)
            hi = max(baseline, peak)
            threshold_level = float(min(max(threshold_level, lo), hi))
            # Find first point that crosses threshold
            above_threshold = yf >= threshold_level
            if np.any(above_threshold):
                onset_idx = int(np.argmax(above_threshold))
            else:
                onset_idx = 0  # All points below threshold, keep everything
            # Onset detection debug messages (commented out - enable if needed for debugging)
            # try:
            #     from smoothing import progress_print
            #     progress_print(f"[fit_average_event] Baseline: {baseline:.3f}, Peak: {peak:.3f}, Threshold: {threshold_level:.3f}")
            # except Exception:
            #     pass

        # Apply onset masking
        if onset_idx > 0:
            tf = tf[onset_idx:]
            yf = yf[onset_idx:]

        # Fit error should ignore pre-stim (t<0) and pre-onset samples
        if tf.size:
            fit_start_ms = max(0.0, float(tf[0]))
        else:
            fit_start_ms = 0.0
        early_window_ms = 0.005
        early_boost = 2.0
        decay_boost = 2.0
        amp_floor = 0.2
        amp_scale = 0.8
        amp_weight = np.ones_like(yf)
        try:
            fit_mask_full = tf >= fit_start_ms
            if np.any(fit_mask_full):
                abs_y = np.abs(yf)
                max_abs = float(np.nanmax(abs_y[fit_mask_full]))
                if np.isfinite(max_abs) and max_abs > 0:
                    amp_norm = abs_y / max_abs
                    amp_weight = amp_floor + amp_scale * amp_norm
        except Exception:
            amp_weight = np.ones_like(yf)

        p0 = spec['p0_func'](yf, tf)
        # Robust seeding for two_step_binding: quick coarse grid search
        # to avoid local minima/stagnation at defaults.
        try:
            if spec.get('name', '').lower() == 'two_step_binding':
                # Unpack current guesses safely
                amp0, tb0, tc0, td0, tp0 = [float(x) for x in p0]
                # Build small grids around typical ranges; clip to bounds
                lb, ub = spec['bounds']
                def _clip(v, lo, hi):
                    return float(min(max(v, lo), hi))
                tb_grid = np.array([
                    _clip(x, lb[1], ub[1]) for x in (0.0006, 0.0010, 0.0016, 0.0025, 0.0040)
                ])
                tc_grid = np.array([
                    _clip(x, lb[2], ub[2]) for x in (0.004, 0.006, 0.008, 0.012, 0.016)
                ])
                td_grid = np.array([
                    _clip(x, lb[3], ub[3]) for x in (0.020, 0.030, 0.040, 0.060, 0.080)
                ])
                tp_grid = np.array([
                    _clip(x, lb[4], ub[4]) for x in (tp0 - 2.0, tp0 - 1.0, tp0, tp0 + 1.0, tp0 + 2.0)
                ])
                best = (np.inf, amp0, tb0, tc0, td0, tp0)
                for tb in tb_grid:
                    for tc in tc_grid:
                        # Enforce sequential regime: tc >= tb
                        if tc <= tb:
                            continue
                        for td in td_grid:
                            for tp in tp_grid:
                                pars = [1.0, tb, tc, td, tp]
                                yshape = spec['func'](tf, *pars)
                                denom = float(np.sum(yshape ** 2))
                                if denom <= 0 or not np.isfinite(denom):
                                    continue
                                amp = float(np.sum(yf * yshape)) / denom
                                r = yf - amp * yshape
                                sse = float(np.dot(r, r))
                                if sse < best[0]:
                                    best = (sse, amp, tb, tc, td, tp)
                _, amp_b, tb_b, tc_b, td_b, tp_b = best
                p0 = [float(amp_b), float(tb_b), float(tc_b), float(td_b), float(tp_b)]
        except Exception:
            # If anything goes wrong, keep original p0
            pass
        
        def _grid_from_bounds(lo, hi, default_vals, n=10):
            try:
                lo_f = float(lo)
                hi_f = float(hi)
            except Exception:
                lo_f, hi_f = np.nan, np.nan
            if np.isfinite(lo_f) and np.isfinite(hi_f):
                if abs(hi_f - lo_f) < 1e-12:
                    return np.array([lo_f], float)
                return np.linspace(lo_f, hi_f, int(n))
            return np.array([_clip(x, lo, hi) for x in default_vals], float)

        # Robust seeding for iGluSnFR: grid search over key parameters
        grid_search_best = None
        tau_rise_grid = None
        tau_fast_grid = None
        tau_slow_grid = None
        tau_superslow_grid = None
        frac_fast_grid = None
        frac_slow_grid = None
        tp_grid = None
        tp_data = None
        # Critical for high-frequency trains where the short fitting window makes
        # curve_fit sensitive to initial conditions
        try:
            if spec.get('name', '').lower() == 'iglusnfr':
                # For iGluSnFR: amp, tau_rise, tau_decay_fast, tau_decay_slow, frac_fast, t_onset
                amp0, tr0, tdf0, tds0, ff0, tp0 = [float(x) for x in p0]
                lb, ub = spec['bounds']
                
                def _clip(v, lo, hi):
                    return float(min(max(v, lo), hi))
                
                # Grids for key parameters with NON-OVERLAPPING ranges
                # tau_rise: 0.5-3ms, tau_fast: 1.5-7ms (tighter), tau_slow: 15-150ms
                tau_rise_grid = _grid_from_bounds(lb[1], ub[1], (0.0005, 0.001, 0.0015, 0.002, 0.003))
                tau_fast_grid = _grid_from_bounds(lb[2], ub[2], (0.0015, 0.002, 0.003, 0.004, 0.005, 0.007))
                tau_slow_grid = _grid_from_bounds(lb[3], ub[3], (0.015, 0.025, 0.040, 0.060, 0.100, 0.150))
                frac_fast_grid = np.array([_clip(x, lb[4], ub[4]) for x in (0.4, 0.55, 0.7, 0.85)])
                
                # Find peak location in data
                peak_idx = np.argmax(yf)
                tp_data = float(tf[peak_idx])
                tp_grid = np.array([_clip(x, lb[5], ub[5]) for x in (tp_data - 1.0, tp_data, tp_data + 1.0)])
                
                best = (np.inf, amp0, tr0, tdf0, tds0, ff0, tp0)
                for tr in tau_rise_grid:
                    for tdf in tau_fast_grid:
                        # Enforce tau_rise < tau_fast
                        if tr >= tdf:
                            continue
                        for tds in tau_slow_grid:
                            # Enforce tau_fast < tau_slow (with gap at 10ms boundary)
                            if tdf >= tds:
                                continue
                            for ff in frac_fast_grid:
                                for tp in tp_grid:
                                    pars = [1.0, tr, tdf, tds, ff, tp]
                                    yshape = spec['func'](tf, *pars)
                                    denom = float(np.sum(yshape ** 2))
                                    if denom <= 0 or not np.isfinite(denom):
                                        continue
                                    amp = float(np.sum(yf * yshape)) / denom
                                    if amp <= 0:
                                        continue
                                    r = yf - amp * yshape
                                    # Weight residuals: ignore pre-stim/onset; emphasize early and decay
                                    weights = np.zeros_like(r)
                                    fit_mask = tf >= fit_start_ms
                                    if np.any(fit_mask):
                                        weights[fit_mask] = 1.0
                                        early_mask = (tf >= fit_start_ms) & (tf <= fit_start_ms + early_window_ms)
                                        if np.any(early_mask):
                                            weights[early_mask] = np.maximum(weights[early_mask], early_boost)
                                        decay_mask = tf > tp
                                        if np.any(decay_mask):
                                            weights[decay_mask] = np.maximum(weights[decay_mask], decay_boost)
                                        weights[fit_mask] = weights[fit_mask] * amp_weight[fit_mask]
                                    else:
                                        weights = np.ones_like(r)
                                    sse = float(np.sum(weights * r ** 2))
                                    if sse < best[0]:
                                        best = (sse, amp, tr, tdf, tds, ff, tp)
                
                if best[0] < np.inf:
                    _, amp_b, tr_b, tdf_b, tds_b, ff_b, tp_b = best
                    p0 = [float(amp_b), float(tr_b), float(tdf_b), float(tds_b), float(ff_b), float(tp_b)]
                    grid_search_best = {
                        'sse': float(best[0]),
                        'params': p0.copy(),
                    }
                    
                    # Dynamically constrain bounds based on grid search results
                    # This prevents curve_fit from diverging to edge solutions
                    lb_list = list(spec['bounds'][0])
                    ub_list = list(spec['bounds'][1])
                    
                    # Constrain tau_rise upper bound to < tau_fast (prevents role swapping)
                    tau_rise_max = min(ub_list[1], tdf_b * 0.9)
                    ub_list[1] = float(tau_rise_max)
                    
                    # Keep tau_fast bounds as specified (don't constrain from grid search)
                    # This allows curve_fit to find the optimal tau_fast within the full range
                    
                    spec['bounds'] = (lb_list, ub_list)
                    
                    # Per-event grid search logging disabled for cleaner output
                    # from smoothing import progress_print
                    # progress_print(f"[fit_average_event] iGluSnFR grid search: tau_r={tr_b*1000:.1f}ms, tau_fast={tdf_b*1000:.1f}ms, tau_slow={tds_b*1000:.1f}ms, frac_fast={ff_b:.2f}")
                    pass
        except Exception:
            pass
        
        # Robust seeding for iGluSnFR TRI-EXPONENTIAL: grid search over key parameters
        try:
            if spec.get('name', '').lower() == 'iglusnfr_tri':
                # For iGluSnFR_tri: amp, tau_rise, tau_decay_fast, tau_decay_slow, tau_decay_superslow, frac_fast, frac_slow, t_onset
                amp0, tr0, tdf0, tds0, tdss0, ff0, fs0, tp0 = [float(x) for x in p0]
                lb, ub = spec['bounds']
                
                def _clip(v, lo, hi):
                    return float(min(max(v, lo), hi))
                
                # Grids for key parameters with NON-OVERLAPPING ranges
                # tau_rise: 0.3-3ms, tau_fast: 0.5-8ms, tau_slow: 8-35ms, tau_superslow: 25-100ms
                tau_rise_grid = _grid_from_bounds(lb[1], ub[1], (0.0003, 0.0005, 0.001, 0.002, 0.003))
                tau_fast_grid = _grid_from_bounds(lb[2], ub[2], (0.0005, 0.001, 0.002, 0.004, 0.006, 0.008))
                tau_slow_grid = _grid_from_bounds(lb[3], ub[3], (0.010, 0.015, 0.020, 0.030))
                # tau_superslow often fixed from post-train decay, use narrower grid unless bounded
                tau_superslow_grid = _grid_from_bounds(lb[4], ub[4], (0.030, 0.050, 0.080, 0.120))
                frac_fast_grid = np.array([_clip(x, lb[5], ub[5]) for x in (0.4, 0.55, 0.7, 0.85)])
                frac_slow_grid = np.array([_clip(x, lb[6], ub[6]) for x in (0.10, 0.20, 0.30)])  # Intermediate fraction
                
                # Find peak location in data
                peak_idx = np.argmax(yf)
                tp_data = float(tf[peak_idx])
                tp_grid = np.array([_clip(x, lb[7], ub[7]) for x in (tp_data - 1.0, tp_data, tp_data + 1.0)])
                
                best = (np.inf, amp0, tr0, tdf0, tds0, tdss0, ff0, fs0, tp0)
                for tr in tau_rise_grid:
                    for tdf in tau_fast_grid:
                        # Enforce tau_rise < tau_fast
                        if tr >= tdf:
                            continue
                        for tds in tau_slow_grid:
                            # Enforce tau_fast < tau_slow
                            if tdf >= tds:
                                continue
                            for tdss in tau_superslow_grid:
                                # Enforce tau_slow < tau_superslow
                                if tds >= tdss:
                                    continue
                                for ff in frac_fast_grid:
                                    for fs in frac_slow_grid:
                                        # Ensure frac_superslow = 1 - ff - fs > 0
                                        if ff + fs >= 0.95:
                                            continue
                                        for tp in tp_grid:
                                            pars = [1.0, tr, tdf, tds, tdss, ff, fs, tp]
                                            yshape = spec['func'](tf, *pars)
                                            denom = float(np.sum(yshape ** 2))
                                            if denom <= 0 or not np.isfinite(denom):
                                                continue
                                            amp = float(np.sum(yf * yshape)) / denom
                                            if amp <= 0:
                                                continue
                                            r = yf - amp * yshape
                                            # Weight residuals: ignore pre-stim/onset; emphasize early and decay
                                            weights = np.zeros_like(r)
                                            fit_mask = tf >= fit_start_ms
                                            if np.any(fit_mask):
                                                weights[fit_mask] = 1.0
                                                early_mask = (tf >= fit_start_ms) & (tf <= fit_start_ms + early_window_ms)
                                                if np.any(early_mask):
                                                    weights[early_mask] = np.maximum(weights[early_mask], early_boost)
                                                decay_mask = tf > tp
                                                if np.any(decay_mask):
                                                    weights[decay_mask] = np.maximum(weights[decay_mask], decay_boost)
                                                weights[fit_mask] = weights[fit_mask] * amp_weight[fit_mask]
                                            else:
                                                weights = np.ones_like(r)
                                            sse = float(np.sum(weights * r ** 2))
                                            if sse < best[0]:
                                                best = (sse, amp, tr, tdf, tds, tdss, ff, fs, tp)
                
                if best[0] < np.inf:
                    _, amp_b, tr_b, tdf_b, tds_b, tdss_b, ff_b, fs_b, tp_b = best
                    p0 = [float(amp_b), float(tr_b), float(tdf_b), float(tds_b), float(tdss_b), float(ff_b), float(fs_b), float(tp_b)]
                    grid_search_best = {
                        'sse': float(best[0]),
                        'params': p0.copy(),
                    }
                    
                    # Per-event grid search logging disabled for cleaner output
                    # from smoothing import progress_print
                    # progress_print(f\"[fit_average_event] iGluSnFR TRI grid search: ...\")
        except Exception:
            pass
        
        # Build sigma weights for curve_fit to emphasize the fast decay portion
        # The initial decay (0-5ms after peak) is where the fast component dominates
        # Lower sigma = higher weight in the least-squares objective
        sigma = None
        try:
            model_name = spec.get('name', '').lower()
            if model_name in ('iglusnfr', 'iglusnfr_tri'):
                peak_idx = int(np.argmax(yf))
                t_onset = float(tf[peak_idx])
                sigma = np.ones_like(yf)
                # High weight (low sigma) for initial decay: peak to peak+5ms
                fast_decay_mask = (tf >= t_onset) & (tf <= t_onset + 0.005)
                sigma[fast_decay_mask] = 0.3  # ~3x higher weight
                # Medium weight for mid decay: peak+5ms to peak+10ms
                mid_decay_mask = (tf > t_onset + 0.005) & (tf <= t_onset + 0.010)
                sigma[mid_decay_mask] = 0.6  # ~1.7x higher weight
                # Standard weight for late decay and rise
                # Boost early post-stim window to stabilize tau_rise
                early_mask = (tf >= fit_start_ms) & (tf <= fit_start_ms + early_window_ms)
                if np.any(early_mask):
                    sigma[early_mask] = np.minimum(sigma[early_mask], 0.5)
                # Use data amplitude to weight the fit (higher amplitude => higher weight)
                sigma = sigma / np.sqrt(np.maximum(amp_weight, 1e-6))
                
                # If fixed_tau_slow provided, lock tau_decay_slow bounds (or tau_decay_superslow for tri-exp)
                if fixed_tau_slow is not None:
                    lb_list = list(spec['bounds'][0])
                    ub_list = list(spec['bounds'][1])
                    tds_fixed = float(fixed_tau_slow)
                    
                    if model_name == 'iglusnfr_tri':
                        # For tri-exp: fixed_tau_slow -> tau_decay_superslow (index 4)
                        lb_list[4] = tds_fixed * 0.999
                        ub_list[4] = tds_fixed * 1.001
                        p0[4] = tds_fixed
                        try:
                            from smoothing import progress_print
                            progress_print(f"[fit_average_event] tau_superslow FIXED to {tds_fixed*1000:.1f}ms (from post-train decay)")
                        except Exception:
                            pass
                    else:
                        # For bi-exp: fixed_tau_slow -> tau_decay_slow (index 3)
                        lb_list[3] = tds_fixed * 0.999
                        ub_list[3] = tds_fixed * 1.001
                        p0[3] = tds_fixed
                        try:
                            from smoothing import progress_print
                            progress_print(f"[fit_average_event] tau_slow FIXED to {tds_fixed*1000:.1f}ms (from post-train decay)")
                        except Exception:
                            pass
                    
                    spec['bounds'] = (lb_list, ub_list)
        except Exception:
            sigma = None
        
        # Store grid search p0 for tri-exp fallback validation
        grid_search_p0 = p0.copy() if spec.get('name', '').lower() == 'iglusnfr_tri' else None

        def _weighted_sse(params):
            y_fit = spec['func'](tf, *params)
            r = yf - y_fit
            t_onset = float(params[-1])
            weights = np.zeros_like(r)
            fit_mask = tf >= fit_start_ms
            if np.any(fit_mask):
                weights[fit_mask] = 1.0
                early_mask = (tf >= fit_start_ms) & (tf <= fit_start_ms + early_window_ms)
                if np.any(early_mask):
                    weights[early_mask] = np.maximum(weights[early_mask], early_boost)
                decay_mask = tf > t_onset
                if np.any(decay_mask):
                    weights[decay_mask] = np.maximum(weights[decay_mask], decay_boost)
                weights[fit_mask] = weights[fit_mask] * amp_weight[fit_mask]
            else:
                weights = np.ones_like(r)
            return float(np.sum(weights * r ** 2))

        def _clip_p0(params):
            lb, ub = spec['bounds']
            return [float(min(max(v, lo), hi)) for v, lo, hi in zip(params, lb, ub)]

        def _seed_with_amp(base_params):
            params = [1.0] + list(base_params)
            yshape = spec['func'](tf, *params)
            denom = float(np.sum(yshape ** 2))
            amp = float(np.sum(yf * yshape)) / denom if denom > 0 else 1.0
            params[0] = max(amp, 0.0)
            return params

        maxfev_eff = int(maxfev)
        try:
            model_name = spec.get('name', '').lower()
            if model_name in ('iglusnfr', 'iglusnfr_tri'):
                maxfev_eff = int(maxfev_eff * 3)
        except Exception:
            maxfev_eff = int(maxfev)

        curve_fit_failed = False
        try:
            try:
                p0 = _clip_p0(p0)
            except Exception:
                pass
            popt, _ = curve_fit(
                spec['func'], tf, yf, p0=p0, bounds=spec['bounds'], maxfev=maxfev_eff,
                sigma=sigma, absolute_sigma=False
            )
        except Exception as e:
            curve_fit_failed = True
            if grid_search_best is not None:
                popt = np.array(grid_search_best['params'], float)
                try:
                    from smoothing import progress_print
                    progress_print(f"[fit_average_event] curve_fit failed ({type(e).__name__}); using grid search params")
                except Exception:
                    pass
            else:
                popt = np.array(p0, float)

        # Multi-start from paired grid seeds if curve_fit is clearly worse than grid search
        try:
            model_name = spec.get('name', '').lower()
            if (
                grid_search_best is not None
                and model_name in ('iglusnfr', 'iglusnfr_tri')
                and tp_grid is not None
                and tau_fast_grid is not None
                and tau_slow_grid is not None
                and tau_rise_grid is not None
            ):
                sse_curve = _weighted_sse(popt)
                if sse_curve > grid_search_best['sse'] * 1.05:
                    fast_n = len(tau_fast_grid)
                    slow_n = len(tau_slow_grid)
                    rise_n = len(tau_rise_grid)
                    fast_idx = np.unique([0, 1, fast_n // 2, max(fast_n // 2 - 1, 0), fast_n - 1])
                    slow_offsets = [0, 1, 2]
                    tp_seeds = list(tp_grid)
                    # Allow extra jitter when the fit is clearly off
                    if tp_data is not None and sse_curve > grid_search_best['sse'] * 1.25:
                        tp_seeds.extend([tp_data - 2.0, tp_data + 2.0])

                    ff_seed = None
                    fs_seed = None
                    if grid_search_best is not None:
                        if model_name == 'iglusnfr':
                            ff_seed = float(grid_search_best['params'][4])
                        else:
                            ff_seed = float(grid_search_best['params'][5])
                            fs_seed = float(grid_search_best['params'][6])
                    if ff_seed is None and frac_fast_grid is not None:
                        ff_seed = float(np.median(frac_fast_grid))
                    if fs_seed is None and frac_slow_grid is not None:
                        fs_seed = float(np.median(frac_slow_grid))

                    candidates = []
                    for i in fast_idx:
                        i = int(i)
                        tr = float(tau_rise_grid[min(i, rise_n - 1)])
                        tdf = float(tau_fast_grid[i])
                        for off in slow_offsets:
                            j = min(i + off, slow_n - 1)
                            tds = float(tau_slow_grid[j])
                            if tdf >= tds:
                                continue
                            for tp in tp_seeds:
                                if model_name == 'iglusnfr':
                                    base = [tr, tdf, tds, float(ff_seed), float(tp)]
                                    candidates.append(_seed_with_amp(base))
                                else:
                                    if tau_superslow_grid is None or frac_slow_grid is None:
                                        continue
                                    sup_n = len(tau_superslow_grid)
                                    k = min(j + 1, sup_n - 1)
                                    tdss = float(tau_superslow_grid[k])
                                    if tds >= tdss:
                                        continue
                                    ff = float(ff_seed) if ff_seed is not None else 0.5
                                    fs = float(fs_seed) if fs_seed is not None else 0.2
                                    if ff + fs >= 0.95:
                                        fs = max(0.05, 0.9 - ff)
                                    base = [tr, tdf, tds, tdss, ff, fs, float(tp)]
                                    candidates.append(_seed_with_amp(base))
                                if len(candidates) >= 12:
                                    break
                            if len(candidates) >= 12:
                                break
                        if len(candidates) >= 12:
                            break

                    best_params = popt
                    best_sse = sse_curve
                    for cand in candidates:
                        try:
                            p0_c = _clip_p0(cand)
                            popt_c, _ = curve_fit(
                                spec['func'], tf, yf, p0=p0_c, bounds=spec['bounds'], maxfev=maxfev,
                                sigma=sigma, absolute_sigma=False
                            )
                            sse_c = _weighted_sse(popt_c)
                            if sse_c < best_sse:
                                best_sse = sse_c
                                best_params = popt_c
                        except Exception:
                            continue
                    if best_sse < sse_curve * 0.98:
                        popt = np.array(best_params, float)
                        try:
                            from smoothing import progress_print
                            progress_print("[fit_average_event] Multi-start improved recut fit; using paired grid seed.")
                        except Exception:
                            pass
        except Exception:
            pass

        # Fallback if curve_fit underperforms the coarse grid search (local minimum)
        try:
            if grid_search_best is not None:
                y_fit = spec['func'](tf, *popt)
                r = yf - y_fit
                t_onset = float(popt[-1])
                weights = np.zeros_like(r)
                fit_mask = tf >= fit_start_ms
                if np.any(fit_mask):
                    weights[fit_mask] = 1.0
                    early_mask = (tf >= fit_start_ms) & (tf <= fit_start_ms + early_window_ms)
                    if np.any(early_mask):
                        weights[early_mask] = np.maximum(weights[early_mask], early_boost)
                    decay_mask = tf > t_onset
                    if np.any(decay_mask):
                        weights[decay_mask] = np.maximum(weights[decay_mask], decay_boost)
                    weights[fit_mask] = weights[fit_mask] * amp_weight[fit_mask]
                else:
                    weights = np.ones_like(r)
                sse_curve = float(np.sum(weights * r ** 2))
                if sse_curve > grid_search_best['sse'] * 1.10:
                    popt = np.array(grid_search_best['params'], float)
                    try:
                        from smoothing import progress_print
                        progress_print("[fit_average_event] curve_fit worse than grid search, using grid search params")
                    except Exception:
                        pass
        except Exception:
            pass
        
        # For tri-exponential: validate curve_fit results and fallback to grid search if needed
        if spec.get('name', '').lower() == 'iglusnfr_tri' and grid_search_p0 is not None:
            # Check if curve_fit produced extreme tau values that differ greatly from grid search
            # Indices: 2=tau_fast, 3=tau_slow, 4=tau_superslow
            tau_fast_cf = popt[2]
            tau_slow_cf = popt[3]
            tau_ss_cf = popt[4]
            tau_fast_gs = grid_search_p0[2]
            tau_slow_gs = grid_search_p0[3]
            tau_ss_gs = grid_search_p0[4]
            
            # Check for unreasonable deviations (>5x different from grid search)
            use_grid_search = False
            if tau_slow_cf > tau_slow_gs * 5 or tau_slow_cf < tau_slow_gs / 5:
                use_grid_search = True
            if tau_ss_cf > tau_ss_gs * 3 or tau_ss_cf < tau_ss_gs / 3:
                use_grid_search = True
                
            if use_grid_search:
                try:
                    from smoothing import progress_print
                    progress_print(f"[fit_average_event] TRI curve_fit produced extreme values, using grid search instead")
                    progress_print(f"[fit_average_event] curve_fit: tau_fast={tau_fast_cf*1000:.1f}ms, tau_slow={tau_slow_cf*1000:.1f}ms, tau_superslow={tau_ss_cf*1000:.1f}ms")
                except Exception:
                    pass
                popt = np.array(grid_search_p0)
        
        params = {name: float(val) for name, val in zip(spec['params'], popt)}
        # If caller requested recut snippets, attach them to the params dict
        # so callers that call this helper via the pipeline can access them
        # without changing the function's primary return signature.
        try:
            if return_snippets and 'snippets' in locals() and 't_rel_s' in locals() and 'avg' in locals():
                params['_recut'] = (t_rel_s, avg, snippets)
                # Verbose snippet attach logging disabled for cleaner output
                # try:
                #     from smoothing import progress_print
                #     progress_print(f"[fit_average_event] Attached {len(snippets)} snippets to params dict")
                # except Exception:
                #     pass
            elif return_snippets:
                # try:
                #     from smoothing import progress_print
                #     progress_print(f"[fit_average_event] return_snippets={return_snippets}, but snippets not in locals")
                # except Exception:
                #     pass
                pass
        except Exception as e:
            try:
                from smoothing import progress_print
                progress_print(f"[fit_average_event] Failed to attach snippets: {e}")
            except Exception:
                pass
        return params, t_ms, y_avg
    except Exception as e:
        try:
            from smoothing import progress_print
            import traceback
            progress_print(f"[fit_average_event] FAILED with error: {type(e).__name__}: {e}")
            progress_print(f"[fit_average_event] Traceback: {traceback.format_exc()}")
        except Exception:
            pass
        return None
