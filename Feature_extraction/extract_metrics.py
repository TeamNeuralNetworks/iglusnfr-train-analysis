"""Clear, streamlined iGluSnFR train analysis

This module provides a compact, publication‑friendly API that is
mathematically equivalent to the main pipeline. It extracts per‑pulse
amplitudes and PPRs from iGluSnFR trains with minimal moving parts and a
simple options dictionary for configuration and plotting. For tri‑exp
(`iglusnfr_tri`), recut fits use the bi‑exp model for fast/slow taus,
superslow tau is estimated from the final event decay, and NNLS templates
scan slow/superslow fractions with a monotonic superslow ramp.

Key steps performed (mirrors batch_measure_complex):
  1) Interpolate NaNs and (optionally) correct slow bleaching
  2) Baseline to ΔF/F0 using the median over the full pre‑train interval
  3) Estimate kinetics from the average trace (rise τr; per‑pulse decay τd)
  4) Robust NNLS per‑pulse amplitudes with micro‑shifts (forward, no overlap)
  5) Per‑pulse amplitudes via local averaged max around each stimulus
  6) Null amplitudes and a MAD‑based threshold/p‑value for A1

Only the pieces necessary for this workflow are implemented here, using the
same formulas and defaults as the original pipeline for equivalence.
"""

from typing import Optional, Dict, List, Tuple, Any

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls

try:
    from smoothing import (
            progress_print,
            fill_nans_timewise,
            sg_smooth,
            iglusnfr_kernel,
            time_zoom_mask,
            windowed_max,
            pick_peak_on_series,
            compute_no_signal_mask,
            build_median_recut_waveform,
    )
except Exception:
    # Fallback: allow importing when current working dir is this subfolder
    import os, sys
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from smoothing import (
        progress_print,
        fill_nans_timewise,
        sg_smooth,
        iglusnfr_kernel,
        time_zoom_mask,
        windowed_max,
        pick_peak_on_series,
        compute_no_signal_mask,
        build_median_recut_waveform,
    )

try:
    from Model_Calibration.simple_curve_fit import fit_average_event
except Exception:  # pragma: no cover - allow running from subfolder
    from simple_curve_fit import fit_average_event  # type: ignore


"""
Default parameters consolidated into a single dictionary for clarity.
These values match the original pipeline and are applied consistently.
"""
DEFAULTS = {
    # Smoothing
    'sg_window': 9,
    'sg_poly': 2,
    # Train fit/plot window
    'pre_zoom_s': 0.15,
    'post_zoom_s': 0.60,
    # Peak window (ms)
    'peak_window_ms': 25.0,
    'peak_avg_points': 5,
    'pre_peak_ms': 0.0,
    # Baseline and null sampling
    'f0_window_s': 0.4,
    'null_sim_max_points': 1000,
    'null_min_post_zoom_s': 0.05,
    'null_N': 3.0,  # MAD rule multiplier
    # Kinetics grids (ms)
    'kin_taur_grid_ms': [0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
    # Extend decay grids so τd can grow well beyond 10 ms when trains slow down
    'kin_taud0_grid_ms': [1.6, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 10.0, 12.5, 15.0, 18.0, 22.0, 28.0, 35.0, 45.0, 60.0],
    'kin_slope_grid_ms': [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0],
    # Robust NNLS + micro‑shift
    'huber_delta': 5.5,
    'irls_iters': 6,
    'delta_max_s': 2.0 / 1000.0,
    'delta_step_s': 0.25 / 1000.0,
    'shift_min_s': 0.05 / 1000.0,
    # ΔF/F0 baseline safety
    'f0_eps': 1e-9,
    # Bleach correction
    'bleach_huber_delta': 3.0,
    'bleach_tau_range_factor': (0.25, 4.0),
    'bleach_n_tau': 25,
    # Event model (kernel) used for per-pulse fitting
    # Supported (varying): 'double_exp' (rise+decay), 'cooperative', 'bilinear'
    # Default is a rise+decay kernel (difference of exponentials)
    'event_model': 'double_exp',
    # Measurement and thresholds
    # measurement: which amplitude series to use for p-values/classification
    #   'NNLS' | 'SAVGOL' | 'RAW'
    'measurement': 'NNLS',
    # threshold_mode: 'auto' selects 'MAD' for NNLS and 'SD' for SAVGOL;
    # can be forced to 'mad' or 'sd'
    'threshold_mode': 'auto',
    # failure classification method: 'NNLS' | 'SAVGOL' | 'RAW'
    # default None means "use measurement"
    'fail_method': None,
    # Toggle per-pulse micro-shifts during fitting and null sampling
    'allow_shift': True,
    # Kinetics source and progression controls
    #  - fit_source: 'global' | 'average' | 'individual'
    #    * global: fit a single event template from all trials (recut median)
    #    * average: fit kinetics on the multi-trial average trace
    #    * individual: fit kinetics per trial then aggregate (median)
    'fit_source': 'global',
    # Decay progression across train (applies to all fit_source modes)
    #  - 'fixed': apply one τd to the whole train (median of estimates)
    #  - 'free_monotonic': interpolate between first and last τd, non-decreasing
    #  - 'linear': non-negative slope linear regression across pulses
    'decay_progression_mode': 'linear',
    'anchor_first_tau': False,  # If True, anchor first event's tau as minimum (fastest decay)
    'anchor_final_tau': True,  # If True, anchor last event's tau as maximum (most reliable, no following events)
    # NNLS weight control
    'nnls_weight_mode': 'uniform',  # 'uniform', 'linear', 'exponential', 'savgol', 'peak'
    'nnls_weight_tau_s': None,  # Time constant for exponential or slope for linear (auto if None)
    'nnls_peak_window_s': 0.010,  # Peak-emphasis window length after each stimulus (seconds)
    'nnls_peak_weight': 3.0,  # Weight multiplier inside the peak window
    # Last-event tail downweighting to prevent overshoot (asymmetric noise handling)
    'nnls_last_event_tail_tau_s': None,  # Exponential downweight tau for last event tail (None=disabled, 'auto'=isi, or float)
    'fit_diagnostic_plot': False,  # Display weight and decay progression diagnostics
    # Template variants for residual-guided fitting (experimental)
    'use_template_variants': False,  # Enable multi-template NNLS (single pass, data-driven slow component)
    # Bi-exp: slow component fractions per event.
    # Tri-exp: slow fraction grid paired with template_variant_superslow_fracs.
    'template_variant_ratios': np.linspace(0.0, 1.0, 10),
    # NNLS fitting mode: 'simultaneous' (default, all events jointly) or 'sequential'
    # (greedy forward pass: each event fitted on residual after subtracting previous event tails;
    # resolves fast/superslow degeneracy in high-overlap trains; followed by two-pass
    # smoothing of kinetic parameters across events)
    'nnls_fit_mode': 'simultaneous',
}

# Recut options: oversample factor and projection ('mean'|'median'|'std')
DEFAULTS.update({
    'recut_oversample': 1,
    'recut_projection': 'median',
    'recut_peak_recenter': 0,
})

# Onset detection for high-frequency trains (exclude contaminated baseline)
DEFAULTS.update({
    'onset_method': 'inflection',  # 'inflection', 'baseline_threshold', or 'none'
    'onset_baseline_threshold': 0.15,  # Threshold for baseline_threshold method (0.1-0.3)
})

# PPR safety: floor amplitudes to noise threshold to prevent division by near-zero values
DEFAULTS.update({
    'amplitude_floor_to_noise': False,  # If True, floor per-trial amplitudes < threshold before PPR calc
    'average_amplitude_floor_to_noise': None,  # Separate control for average trace; None => follow amplitude_floor_to_noise
    'average_null_N': None,  # Separate null_N for average trace floor; None => follow null_N
})

# Tri-exp NNLS variants: sweep slow + superslow fractions (superslow ramps across the train)
DEFAULTS.update({
    'template_variant_superslow_fracs': np.linspace(0.0, 1.0, 11),  # Superslow fraction at final event
    'superslow_min_ratio': 1.0,  # Disable superslow variants if tau_superslow < tau_slow * ratio
    'allow_tau_slow_override': True,  # Allow last-event tau to replace recut tau_decay_slow when shorter
    'force_tau_slow_override': False,  # Force last-event tau as tau_decay_slow (supersedes allow_tau_slow_override)
    'template_variant_tau_slow_ms': None,  # Optional slow-tau grid (ms) for bi-exp template variants
    'template_variant_select': 'soft',  # 'soft'|'hard' selection across template variants
})

# Parameter bounds: classical upper/lower limits for all fitted parameters
# Format: 'parameter_bounds': {'param_name': (lower, upper), ...}
# - Use (value, value) to force a fixed value
# - Use (np.nan, np.nan), (np.inf, np.inf), or None for unconstrained
# - Applies to: tau_rise, tau_decay_fast, tau_decay_slow, frac_fast, n_coop, etc.
# Example: {'tau_decay_fast': (0.006, 0.010), 'tau_decay_slow': (0.025, 0.035)}
# Example (fixed): {'tau_decay_fast': (0.008, 0.008)}
DEFAULTS.update({
    'parameter_bounds': {},  # Dict of {param_name: (lower, upper)} constraints
})

# Default bounds used to auto-build grids when parameter_bounds are not provided.
DEFAULT_PARAM_BOUNDS = {
    'tau_rise': (0.0005, 0.003),            # 0.5–3 ms
    'tau_decay_fast': (0.003, 0.010),       # 3–10 ms
    'tau_decay_slow': (0.010, 0.040),       # 10–40 ms
    'tau_decay_superslow': (0.035, 0.120),  # 35–120 ms
}

# Selected kernel (set inside extract_metrics based on options; default is iglusnfr_kernel)
_KERNEL_FUN = iglusnfr_kernel
# Variant kernel builder for template variants feature (set if model supports it)
_VARIANT_KERNEL_BUILDER = None


# -------------------------
# Small utilities
# -------------------------

def apply_parameter_bounds(fitted_params: Dict[str, float], bounds: Dict[str, tuple]) -> Dict[str, float]:
    """Apply upper/lower bounds to fitted parameters.

    Args:
        fitted_params: Dict of fitted parameter values {param_name: value}
        bounds: Dict of bounds {param_name: (lower, upper)}
                - (value, value): forces fixed value
                - (np.nan, np.nan), (np.inf, np.inf), or None: unconstrained
                - (lower, upper): clips to range

    Returns:
        Dict with constrained parameter values
    """
    result = dict(fitted_params)

    for param_name, value in fitted_params.items():
        if param_name not in bounds:
            continue

        bound = bounds[param_name]
        if bound is None:
            continue

        if not isinstance(bound, (tuple, list)) or len(bound) != 2:
            continue

        lower, upper = float(bound[0]), float(bound[1])

        # Check if unconstrained (NaN or Inf)
        if (not np.isfinite(lower) and not np.isfinite(upper)):
            continue

        # Force fixed value when lower == upper
        if np.isfinite(lower) and np.isfinite(upper) and abs(lower - upper) < 1e-12:
            if abs(value - lower) > 1e-12:
                try:
                    from smoothing import progress_print
                    progress_print(f"[bounds] Forcing {param_name} = {lower:.6f} (was {value:.6f})")
                except Exception:
                    pass
            result[param_name] = lower
            continue

        # Apply bounds
        original_value = value
        if np.isfinite(lower):
            value = max(value, lower)
        if np.isfinite(upper):
            value = min(value, upper)

        if abs(value - original_value) > 1e-12:
            try:
                from smoothing import progress_print
                progress_print(f"[bounds] Clipping {param_name} from {original_value:.6f} to {value:.6f}")
            except Exception:
                pass
        result[param_name] = value

    return result


def estimate_tau_from_post_train_decay(
    t: np.ndarray,
    y: np.ndarray,
    stim_times: np.ndarray,
    isi: float,
    min_decay_points: int = 10,
) -> Optional[float]:
    """Estimate tau_slow from the post-train decay portion of the signal.
    
    After the last stimulus, the signal decays from the accumulated response.
    This decay is dominated by the slowest component and provides an upper
    bound estimate for tau_slow.
    
    Args:
        t: Time vector (seconds)
        y: Signal (averaged trace)
        stim_times: Stimulus times (seconds)
        isi: Inter-stimulus interval (seconds)
        min_decay_points: Minimum data points required for fitting
        
    Returns:
        Estimated tau_slow in seconds, or None if fitting fails
    """
    try:
        from scipy.optimize import curve_fit
        
        # Post-train region: start 0.5 ISI after last stim (let peak settle)
        post_train_start = float(stim_times[-1]) + 0.5 * isi
        # Use up to 5 ISIs of decay (or end of trace)
        post_train_end = min(float(stim_times[-1]) + 5 * isi, t[-1])
        
        mask = (t >= post_train_start) & (t <= post_train_end)
        if np.sum(mask) < min_decay_points:
            return None
            
        t_decay = t[mask] - post_train_start  # Time relative to decay start
        y_decay = y[mask]
        
        # Estimate baseline from end of decay region
        baseline = np.median(y_decay[-max(3, len(y_decay)//5):])
        y_decay_zeroed = y_decay - baseline
        
        # Skip if signal already at baseline
        peak_decay = np.max(y_decay_zeroed)
        if peak_decay < 0.1 * np.std(y_decay_zeroed):
            return None
        
        # Single exponential fit: y = A * exp(-t/tau)
        def exp_decay(t, A, tau):
            return A * np.exp(-t / max(tau, 1e-6))
        
        # Initial guess: amplitude from first point, tau from half-life
        A0 = float(y_decay_zeroed[0])
        half_idx = np.argmin(np.abs(y_decay_zeroed - 0.5 * A0))
        tau0 = float(t_decay[half_idx]) / np.log(2) if half_idx > 0 else 0.050
        tau0 = np.clip(tau0, 0.010, 0.500)
        
        try:
            popt, _ = curve_fit(
                exp_decay, t_decay, y_decay_zeroed,
                p0=[A0, tau0],
                bounds=([0, 0.005], [np.inf, 1.0]),  # tau: 5ms to 1000ms
                maxfev=1000
            )
            tau_fitted = float(popt[1])
            amp_fitted = float(popt[0])
            
            # Log the result
            from smoothing import progress_print
            progress_print(f"[post-train] Estimated tau_slow from decay: {tau_fitted*1000:.1f} ms")
            
            # Store fit info globally for plotting (post_train_start is absolute time)
            global _POST_TRAIN_DECAY_FIT
            _POST_TRAIN_DECAY_FIT = {
                'tau': tau_fitted,
                'amp': amp_fitted,
                'baseline': baseline,
                't_start': post_train_start,
                't_end': post_train_end,
            }
            
            return tau_fitted
            
        except Exception:
            return None
            
    except Exception:
        return None

# Global storage for post-train decay fit (for plotting)
_POST_TRAIN_DECAY_FIT = None
# Global storage for last-event decay fit (for plotting tri-exp superslow)
_LAST_EVENT_DECAY_FIT = None


def estimate_tau_superslow_from_last_event_decay(
    t: np.ndarray,
    y: np.ndarray,
    stim_times: np.ndarray,
    isi: float,
    *,
    event_t0_s: float = 0.0,
    tau_fast_s: Optional[float] = None,
    min_decay_points: int = 12,
) -> Tuple[Optional[float], Optional[float]]:
    """Estimate tau_superslow from a robust single-exponential fit of last-event decay.

    Fits only the decay after the last event peak using a single exponential.
    Weights use amplitude (savgol-smoothed if available, raw otherwise).
    Returns (tau_superslow, None).
    """
    global _LAST_EVENT_DECAY_FIT
    try:
        if stim_times is None or len(stim_times) == 0:
            return None, None

        last_st = float(stim_times[-1]) + float(event_t0_s)
        # Find the local peak of the last event within a short window
        peak_search_end = min(last_st + max(0.5 * isi, 0.010), t[-1])
        peak_mask = (t >= last_st) & (t <= peak_search_end)
        if not np.any(peak_mask):
            return None, None

        peak_idx_rel = int(np.argmax(y[peak_mask]))
        peak_idx = np.where(peak_mask)[0][0] + peak_idx_rel
        peak_t = float(t[peak_idx])
        peak_y = float(y[peak_idx])

        # Baseline from pre-train segment when available
        baseline = None
        try:
            pre_mask = t < float(stim_times[0])
            if np.sum(pre_mask) >= 5:
                baseline = float(np.nanmedian(y[pre_mask]))
        except Exception:
            baseline = None

        # Decay window: start 5ms after peak to avoid immediate transient
        decay_start = peak_t + 0.005
        if decay_start >= t[-1]:
            return None, None

        # Determine end of decay: when signal returns to baseline, else end of trace
        decay_end = float(t[-1])
        if baseline is None:
            baseline = float(np.nanmedian(y[max(0, peak_idx - 10):peak_idx + 1]))
        # Cap decay window to avoid fitting the accumulated slow baseline
        # which inflates the tau estimate.  Use a short window so the fit
        # stays dominated by the intermediate/slow component, not by the
        # very-slow accumulated tail from all preceding events.
        decay_end = min(decay_end, decay_start + 0.050)
        try:
            if np.isfinite(baseline):
                amp = peak_y - baseline
                if not np.isfinite(amp) or amp <= 0:
                    return None, None
                pre_mask = t < float(stim_times[0])
                if np.sum(pre_mask) >= 5:
                    noise = float(np.nanmedian(np.abs(y[pre_mask] - baseline)))
                else:
                    noise = float(np.nanmedian(np.abs(y - baseline)))
                threshold = baseline + max(0.05 * amp, 2.0 * noise)
                after_mask = t >= decay_start
                if np.any(after_mask):
                    y_after = y[after_mask]
                    t_after = t[after_mask]
                    below = y_after <= threshold
                    if np.any(below):
                        first_idx = int(np.argmax(below))
                        candidate_end = float(t_after[first_idx])
                        if candidate_end > decay_start:
                            candidate_mask = (t >= decay_start) & (t <= candidate_end)
                            if np.sum(candidate_mask) >= min_decay_points:
                                decay_end = candidate_end
        except Exception:
            decay_end = float(t[-1])

        decay_mask = (t >= decay_start) & (t <= decay_end)
        if np.sum(decay_mask) < min_decay_points:
            return None, None

        t_decay = t[decay_mask] - decay_start
        y_decay = y[decay_mask]

        # Baseline estimate from pre-train segment or decay tail as fallback
        if baseline is None or not np.isfinite(baseline):
            tail_len = max(3, len(y_decay) // 5)
            baseline = float(np.median(y_decay[-tail_len:]))
        y_decay_zeroed = y_decay - baseline

        # Skip if no clear decay signal
        peak_decay = float(np.nanmax(y_decay_zeroed))
        if not np.isfinite(peak_decay) or peak_decay <= 0:
            return None, None

        # Choose weights from savgol-smoothed trace if available, raw otherwise
        y_weight_src = None
        try:
            sg_window = int(DEFAULTS.get('sg_window', 9))
            sg_poly = int(DEFAULTS.get('sg_poly', 2))
            if sg_window % 2 == 0:
                sg_window += 1
            if sg_window >= 3 and sg_window <= y.size:
                y_weight_src = sg_smooth(fill_nans_timewise(y, t), sg_window, sg_poly)
        except Exception:
            y_weight_src = None
        if y_weight_src is None:
            y_weight_src = y

        y_weight = y_weight_src[decay_mask]
        y_weight_zeroed = y_weight - baseline

        # Robust single-exp fit in log space with amplitude weights
        pos_mask = (y_decay_zeroed > 0) & np.isfinite(y_decay_zeroed) & np.isfinite(t_decay)
        if np.sum(pos_mask) < min_decay_points:
            return None, None

        t_fit = t_decay[pos_mask]
        y_fit = y_decay_zeroed[pos_mask]
        w_amp = np.maximum(y_weight_zeroed[pos_mask], 0.0)
        if not np.isfinite(w_amp).any() or np.nanmax(w_amp) <= 0:
            w_amp = np.ones_like(y_fit)
        else:
            w_amp = w_amp / (np.nanmax(w_amp) + 1e-12)

        logy = np.log(y_fit)
        # Iteratively reweighted least squares with Huber weights
        weights = w_amp.copy()
        tau_slow = None
        a_slow = None
        for _ in range(5):
            wsum = float(np.sum(weights))
            if wsum <= 0:
                break
            t_bar = float(np.sum(weights * t_fit) / wsum)
            y_bar = float(np.sum(weights * logy) / wsum)
            t_demean = t_fit - t_bar
            denom = float(np.sum(weights * t_demean * t_demean))
            if denom <= 1e-12:
                break
            m = float(np.sum(weights * t_demean * (logy - y_bar)) / denom)
            b = float(y_bar - m * t_bar)
            if not np.isfinite(m) or m >= -1e-9:
                break
            tau_slow = float(-1.0 / m)
            a_slow = float(np.exp(b))
            # Huber reweight on log residuals
            resid = logy - (b + m * t_fit)
            delta = 1.5
            abs_r = np.abs(resid)
            w_huber = np.ones_like(resid)
            mask = abs_r > delta
            w_huber[mask] = delta / (abs_r[mask] + 1e-12)
            weights = w_amp * w_huber

        if tau_slow is None or not np.isfinite(tau_slow):
            return None, None

        # If fit hit the upper ceiling it means the exponential never converged —
        # treat as a failed estimate so superslow variants stay disabled.
        _TAU_SUPERSLOW_MAX = 0.350
        if tau_slow >= _TAU_SUPERSLOW_MAX:
            return None, None
        tau_slow = float(np.clip(tau_slow, 0.010, _TAU_SUPERSLOW_MAX))
        if a_slow is None or not np.isfinite(a_slow):
            a_slow = float(peak_decay)

        try:
            from smoothing import progress_print
            progress_print(f"[last-event] Estimated tau_superslow from robust single-exp decay: {tau_slow*1000:.1f} ms")
        except Exception:
            pass
        try:
            _LAST_EVENT_DECAY_FIT = {
                't_start': float(decay_start),
                't_end': float(decay_end),
                'baseline': float(baseline),
                'a_fast': 0.0,
                'tau_fast': np.nan,
                'a_slow': float(a_slow),
                'tau_slow': float(tau_slow),
            }
        except Exception:
            pass
        return tau_slow, None
    except Exception:
        return None, None


def _validate_kernel_monotonic_decay(kernel: np.ndarray, tolerance: float = 0.02) -> bool:
    """Check that kernel has monotonic decay after its peak.
    
    Multi-exponential kernels with certain parameter combinations can exhibit
    non-physical behavior (secondary bumps, curve rising after peak). This
    function validates that the decay portion is monotonically decreasing.
    
    Args:
        kernel: 1D kernel array (normalized or not)
        tolerance: Fraction of peak below which we stop checking (to ignore noise at tail)
        
    Returns:
        True if decay is monotonic (valid kernel), False if non-monotonic (reject)
    """
    if len(kernel) < 3:
        return True
    
    peak_idx = int(np.argmax(kernel))
    peak_val = kernel[peak_idx]
    
    if peak_val <= 0 or peak_idx >= len(kernel) - 2:
        return True  # Edge cases: no valid peak or peak at end
    
    # Check decay portion: from peak to where signal drops below tolerance
    threshold = peak_val * tolerance
    decay_portion = kernel[peak_idx:]
    
    # Find where we drop below threshold (stop checking there)
    below_thresh = np.where(decay_portion < threshold)[0]
    end_check = below_thresh[0] if len(below_thresh) > 0 else len(decay_portion)
    decay_to_check = decay_portion[:max(2, end_check)]
    
    # Check monotonicity: each sample should be <= previous
    # Allow tiny numerical noise (1e-6 relative)
    diffs = np.diff(decay_to_check)
    max_rise = np.max(diffs) if len(diffs) > 0 else 0
    
    # Non-monotonic if any sample rises more than 1% of peak
    return max_rise < peak_val * 0.01


def _calculate_nnls_weights(
    t: np.ndarray,
    stim_times: np.ndarray,
    isi: float,
    weight_mode: str = 'uniform',
    weight_tau_s: Optional[float] = None,
    y_ref: Optional[np.ndarray] = None,
    peak_window_s: Optional[float] = None,
    peak_weight: Optional[float] = None,
) -> np.ndarray:
    """
    Calculate weights for NNLS fitting based on weight_mode.
    
    Args:
        t: Time array
        stim_times: Stimulus times array  
        isi: Inter-stimulus interval in seconds
        weight_mode: 'uniform', 'linear', 'exponential', 'savgol', or 'peak'
        weight_tau_s: Time constant for exponential decay or linear slope (in seconds)
        y_ref: Optional reference trace used for Savitzky-Golay weights (must
            match ``t`` in shape when ``weight_mode`` is ``'savgol'``)
        peak_window_s: Post-stimulus window length for peak emphasis (seconds)
        peak_weight: Weight multiplier inside the peak window
        
    Returns:
        weights: Array same length as t with weights for each timepoint
    """
    weights = np.ones_like(t)
    
    if weight_mode == 'uniform':
        return weights
    
    elif weight_mode == 'linear':
        tau = weight_tau_s if weight_tau_s is not None else isi
        for i, stim_t in enumerate(stim_times):
            # Find next stimulus time (or end of trace)
            if i < len(stim_times) - 1:
                next_stim_t = stim_times[i + 1]
            else:
                # For last stimulus, use same interval as previous
                if len(stim_times) > 1:
                    next_stim_t = stim_t + (stim_times[-1] - stim_times[-2])
                else:
                    next_stim_t = stim_t + isi
            
            # Create linear decay from 1 to 0 over the interval
            mask = (t >= stim_t) & (t < next_stim_t)
            if np.any(mask):
                t_rel = t[mask] - stim_t
                duration = next_stim_t - stim_t
                # Linear decay with controllable slope
                slope_factor = tau / duration  # tau controls how steep the decay is
                decay = 1.0 - (t_rel / duration) * slope_factor
                weights[mask] = np.maximum(decay, 0.0)  # Clip at 0
                
    elif weight_mode == 'exponential':
        tau = weight_tau_s if weight_tau_s is not None else 0.010  # Default 10ms
        for i, stim_t in enumerate(stim_times):
            # Find next stimulus time (or end of trace) 
            if i < len(stim_times) - 1:
                next_stim_t = stim_times[i + 1]
            else:
                if len(stim_times) > 1:
                    next_stim_t = stim_t + (stim_times[-1] - stim_times[-2])
                else:
                    next_stim_t = stim_t + isi
            
            # Create exponential decay
            mask = (t >= stim_t) & (t < next_stim_t)
            if np.any(mask):
                t_rel = t[mask] - stim_t
                weights[mask] = np.exp(-t_rel / tau)
    
    elif weight_mode == 'savgol':
        if y_ref is None:
            raise ValueError("Savgol weight_mode requires a reference trace (y_ref)")
        ref = np.asarray(y_ref, float)
        if ref.shape != t.shape:
            raise ValueError("Reference trace for savgol weights must match time array shape")
        ref = np.where(np.isfinite(ref), ref, 0.0)
        ref = np.abs(ref)
        ref = ref - np.nanmin(ref)
        max_ref = np.nanmax(ref)
        if not np.isfinite(max_ref) or max_ref <= 0:
            return weights
        norm = ref / max_ref
        weights = norm
        return weights
    
    elif weight_mode == 'peak':
        window_s = 0.010 if peak_window_s is None else float(peak_window_s)
        if not np.isfinite(window_s) or window_s <= 0:
            window_s = 0.010
        mult = 3.0 if peak_weight is None else float(peak_weight)
        if not np.isfinite(mult) or mult <= 1.0:
            mult = 1.0
        for stim_t in stim_times:
            mask = (t >= stim_t) & (t < (stim_t + window_s))
            if np.any(mask):
                weights[mask] = np.maximum(weights[mask], mult)
        return weights

    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")

    return weights


def _apply_last_event_tail_weights(
    weights: np.ndarray,
    t: np.ndarray,
    stim_times: np.ndarray,
    isi: float,
    tail_tau_s: Optional[float],
    peak_window_s: float = 0.010,
) -> np.ndarray:
    """Apply exponential downweighting to the last event's decay tail.

    This prevents NNLS from overshooting the last event amplitude by reducing
    the influence of the long post-event tail, where noise can bias the fit.

    Args:
        weights: Existing weight array (will be modified)
        t: Time array
        stim_times: Stimulus times
        isi: Inter-stimulus interval
        tail_tau_s: Exponential time constant for tail decay (None=disabled,
            'auto' or float). If float, weights decay as exp(-t_rel/tau).
        peak_window_s: Keep full weight for this duration after last stim

    Returns:
        Modified weights array
    """
    if tail_tau_s is None or len(stim_times) == 0:
        return weights

    # Handle string modes
    if isinstance(tail_tau_s, str):
        mode = tail_tau_s.lower()
        if mode == 'auto':
            # Use ISI as tau
            tail_tau_s = isi
        elif mode == 'best':
            # 'best' mode requires search - caller should resolve before calling
            # Fall back to 'auto' behavior here
            tail_tau_s = isi
        else:
            try:
                tail_tau_s = float(tail_tau_s)
            except ValueError:
                return weights

    tail_tau_s = float(tail_tau_s)
    if not np.isfinite(tail_tau_s) or tail_tau_s <= 0:
        return weights

    last_stim = float(stim_times[-1])
    # Start downweighting after peak window
    tail_start = last_stim + peak_window_s

    # Apply exponential decay to tail region
    tail_mask = t >= tail_start
    n_affected = np.sum(tail_mask)
    if n_affected > 0:
        t_rel = t[tail_mask] - tail_start
        decay_factor = np.exp(-t_rel / tail_tau_s)
        # Beyond 3*tau, set weight to near-zero (effectively exclude from fit)
        # This prevents the long tail from biasing the amplitude
        decay_factor = np.where(t_rel > 3 * tail_tau_s, 0.001, decay_factor)
        decay_factor = np.maximum(decay_factor, 0.001)
        weights[tail_mask] *= decay_factor
        
        # Log effect
        try:
            from smoothing import progress_print
            n_excluded = np.sum(t_rel > 3 * tail_tau_s)
            min_w = float(np.min(weights[tail_mask]))
            progress_print(f"[tail-weight] τ={tail_tau_s*1000:.1f}ms | {n_affected} tail pts | {n_excluded} excluded (w<0.01)")
        except Exception:
            pass

    return weights


def _compute_residual_whiteness(residual: np.ndarray) -> float:
    """Compute a whiteness score for residuals (higher = more white noise-like).

    Uses multiple metrics combined:
    - Lag-1 autocorrelation (should be ~0 for white noise)
    - Sign-change frequency (should be ~50% for white noise)
    - Running mean envelope (should be small for white noise)

    Returns:
        Whiteness score (0-1, higher is better/whiter)
    """
    if len(residual) < 10:
        return 0.0

    r = residual - np.mean(residual)
    var_r = np.var(r)
    if var_r < 1e-12:
        return 1.0  # Zero residual is perfect

    # Lag-1 autocorrelation: the definitive test for white noise.
    # For a truly white (iid) residual, lag-1 autocorr ~ 0.
    # Score = 1 - |r1|, ranging from 1 (white) to 0 (fully correlated).
    autocorr_1 = np.corrcoef(r[:-1], r[1:])[0, 1] if len(r) > 1 else 0.0
    if not np.isfinite(autocorr_1):
        autocorr_1 = 0.0
    whiteness = 1.0 - abs(float(autocorr_1))
    return float(whiteness)


def _find_best_tail_tau(
    t: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    stim_times: np.ndarray,
    isi: float,
    base_weights: np.ndarray,
    peak_window_s: float = 0.010,
    tau_candidates: Optional[list] = None,
    verbose: bool = False,
) -> float:
    """Search for the tail tau that produces the whitest residuals.

    Uses the pre-built design matrix X for fast evaluation.

    Args:
        t: Time array
        y: Data array
        X: Pre-built design matrix (from variant selection pass)
        stim_times: Stimulus times
        isi: Inter-stimulus interval
        base_weights: Base weights before tail adjustment
        peak_window_s: Peak window duration
        tau_candidates: List of tau values to try (in seconds)
        verbose: Print search progress

    Returns:
        Best tau value in seconds
    """
    if tau_candidates is None:
        # Try a range from 0.3*ISI to 3*ISI
        tau_candidates = [isi * m for m in [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]]

    best_tau = isi  # Default fallback
    best_score = -1.0

    try:
        from smoothing import progress_print
    except ImportError:
        progress_print = print

    scores = []
    for tau in tau_candidates:
        # Apply tail weighting with this tau
        w_test = _apply_last_event_tail_weights(
            base_weights.copy(), t, stim_times, isi, tau, peak_window_s
        )
        # Quick NNLS with pre-built design matrix
        try:
            a_test = _nnls_weighted(X, y, w_test)
            residual = y - X @ a_test
        except Exception:
            continue

        # Score the residuals
        score = _compute_residual_whiteness(residual)
        scores.append((tau, score))

        if score > best_score:
            best_score = score
            best_tau = tau

    if verbose and scores:
        scores_str = ", ".join([f"{tau*1000:.0f}ms:{s:.3f}" for tau, s in scores])
        progress_print(f"[tail-best] τ search: {scores_str} → best={best_tau*1000:.1f}ms (score={best_score:.3f})")

    return best_tau


def _trim_spines(ax):
    """Hide top/right spines for a cleaner look (safe no-op on failure)."""
    try:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    except Exception:
        pass


def _nnls_irls_singlecol(y: np.ndarray, k: np.ndarray, *, robust: bool, huber_delta: float, iters: int) -> float:
    """Single‑column NNLS with optional Huber IRLS (amplitude ≥ 0)."""
    # Validate inputs to catch NaN/Inf before calling nnls
    if not np.all(np.isfinite(y)):
        n_bad = np.sum(~np.isfinite(y))
        raise ValueError(f"Input y contains {n_bad} NaN/Inf values (size={y.size})")
    if not np.all(np.isfinite(k)):
        n_bad = np.sum(~np.isfinite(k))
        raise ValueError(f"Input k (kernel) contains {n_bad} NaN/Inf values (size={k.size})")
    if k.size == 0 or y.size == 0:
        return 0.0

    a = max(0.0, nnls(k[:, None], y)[0][0])
    if not robust:
        return a
    for _ in range(max(1, int(iters))):
        r = y - a * k
        absr = np.abs(r)
        w = np.where(absr <= huber_delta, 1.0, huber_delta / np.maximum(absr, 1e-12))
        Wsqrt = np.sqrt(w)
        kw = k * Wsqrt
        yw = y * Wsqrt
        a = max(0.0, nnls(kw[:, None], yw)[0][0])
    return float(a)


def _nnls_weighted(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Weighted NNLS solver.
    
    Args:
        X: Design matrix (n_timepoints, n_pulses)
        y: Target vector (n_timepoints,)
        weights: Weight vector (n_timepoints,)
        
    Returns:
        amplitudes: Non-negative amplitudes (n_pulses,)
    """
    if X.size == 0:
        return np.zeros(X.shape[1] if X.ndim > 1 else 0)
    
    # Apply weights by scaling both X and y
    Wsqrt = np.sqrt(weights)
    X_weighted = X * Wsqrt[:, None]
    y_weighted = y * Wsqrt
    
    return np.maximum(0.0, nnls(X_weighted, y_weighted)[0])


def _fit_single_pulse_amp(
    y: np.ndarray,
    t: np.ndarray,
    stim_time: float,
    tau_r_s: float,
    tau_d_s: float,
    *,
    pre_zoom_s: float,
    post_zoom_s: float,
    robust: bool = True,
    huber_delta: float,
    irls_iters: int,
    allow_shift: bool = True,
    delta_max_s: float,
    delta_step_s: float,
    shift_min_s: float,
    event_t0_s: float = 0.0,
) -> Tuple[float, float]:
    """Estimate amplitude at ``stim_time`` with optional micro-shift.

    Assumes baseline has been corrected; residual from earlier events should be
    subtracted before calling. Returns ``(amplitude, best_shift_s)``.
    """
    anchor = float(stim_time) + float(event_t0_s)
    local_mask = (t >= (anchor - pre_zoom_s)) & (t <= (anchor + post_zoom_s))
    if not np.any(local_mask):
        return 0.0, 0.0
    y_seg = y[local_mask]
    best_a, best_d = 0.0, 0.0
    if allow_shift:
        start = max(float(shift_min_s), 0.0)
        if delta_max_s > 0 and start <= delta_max_s + 1e-12:
            pos_shifts = np.arange(start, delta_max_s + 1e-12, delta_step_s)
            shifts = np.concatenate(([0.0], pos_shifts)) if pos_shifts.size else np.array([0.0])
        else:
            shifts = np.array([0.0])
    else:
        shifts = np.array([0.0])
    shifts = np.unique(shifts.astype(float))
    for d in shifts:
        k_full = _KERNEL_FUN(t - (anchor + d), tau_r_s, tau_d_s)
        k_loc = k_full[local_mask]
        if k_loc.size < 3 or np.all(k_loc == 0):
            continue
        a_loc = _nnls_irls_singlecol(y_seg, k_loc, robust=robust, huber_delta=huber_delta, iters=irls_iters)
        if a_loc > best_a:
            best_a, best_d = a_loc, d
    return float(best_a), float(best_d)


def fit_amplitudes_with_template_variants(
    y: np.ndarray,
    t: np.ndarray,
    stim_times: np.ndarray,
    tau_r_s: float,
    tau_d_vec_s: np.ndarray,
    *,
    variant_ratios: List[Any],
    weight_mode: str,
    weight_tau_s: Optional[float],
    isi: float,
    event_t0_s: float = 0.0,
    jitter_variant_ms: Optional[np.ndarray] = None,
    peak_window_s: Optional[float] = None,
    peak_weight: Optional[float] = None,
    last_event_tail_tau_s: Optional[float] = None,
    residual_reweight_iters: int = 5,
    residual_reweight_tau: float = 2.0,
    hard_select: bool = False,
):
    """Single-pass NNLS with multiple template variants per event.

    Instead of one template per event, creates N variants with different slow/fast
    component ratios and optionally temporal jitters. NNLS automatically selects
    the best combination.

    Uses residual diagnostics to detect and correct template mismatch:
    - Correlation of residual with cumulative model → decay rate error
    - Correlation of residual with model derivative → timing error

    Args:
        y: Signal to fit
        t: Time vector
        stim_times: Stimulus times
        tau_r_s: Rise time constant
        tau_d_vec_s: Decay time constants per event
        variant_ratios: List of slow component fractions (bi-exp) or
            tri-exp tuples [(frac_slow, frac_superslow_max), ...] where
            superslow ramps 0→max across the train.
        weight_mode: Weighting mode for NNLS
        weight_tau_s: Weighting time constant
        isi: Inter-stimulus interval
        event_t0_s: Event onset offset
        jitter_variant_ms: Optional array of temporal shifts in milliseconds
        residual_reweight_iters: Number of iterative refinement passes (default 3)
        residual_reweight_tau: Downweighting aggressiveness (default 2.0)
        hard_select: If True, choose a single dominant variant per event and
            re-fit amplitudes with a reduced design matrix.

    Returns:
        (amplitudes, shifts, design, reconstruction, components_list, variant_info)
    """
    n_events = len(stim_times)
    n_template_variants = len(variant_ratios)

    # Setup jitter variants (in seconds)
    if jitter_variant_ms is None:
        jitter_variant_s = np.array([0.0])
    else:
        jitter_variant_s = np.asarray(jitter_variant_ms, float) / 1000.0
    # Clamp jitter so anchor never falls before the stimulus (t0+jitter >= 0)
    # or beyond the next stimulus (t0+jitter < ISI).
    jitter_variant_s = np.clip(jitter_variant_s, -event_t0_s, isi - event_t0_s - 0.001)
    jitter_variant_s = np.unique(jitter_variant_s)  # drop duplicates from clipping
    n_jitter_variants = len(jitter_variant_s)

    event_t0_s = float(event_t0_s)
    
    # Track kinetics across iterations (diagnostic only; no adjustment)
    current_tau_d = tau_d_vec_s.copy()
    current_t0 = event_t0_s
    kinetics_adjusted = False
    # Compute base weights once
    base_weights = _calculate_nnls_weights(
        t,
        stim_times,
        isi,
        weight_mode,
        weight_tau_s,
        y_ref=y,
        peak_window_s=peak_window_s,
        peak_weight=peak_weight,
    )

    # Handle 'best' mode flag - actual search happens after X is built
    pw_s = peak_window_s if peak_window_s is not None else 0.010
    do_best_search = isinstance(last_event_tail_tau_s, str) and last_event_tail_tau_s.lower() == 'best'
    resolved_tail_tau = last_event_tail_tau_s if not do_best_search else isi  # Temporary default

    # Compute tail-weighted version (may be updated after best search)
    tail_weighted = _apply_last_event_tail_weights(
        base_weights.copy(), t, stim_times, isi, resolved_tail_tau, pw_s
    )
    use_tail_weighting = last_event_tail_tau_s is not None

    # === Main iterative loop with residual-driven refinement ===
    for iteration in range(max(1, residual_reweight_iters)):
        # Build design matrix with current kinetics
        kernel_columns = []
        for i_event in range(n_events):
            st = float(stim_times[i_event])
            tau_d = float(current_tau_d[i_event])

            for frac_slow in variant_ratios:
                for jitter_s in jitter_variant_s:
                    anchor = st + current_t0 + jitter_s
                    k = _build_variant_kernel(
                        t - anchor, tau_r_s, tau_d, frac_slow,
                        event_idx=i_event, n_events=n_events,
                    )
                    kernel_columns.append(k)

        X = np.column_stack(kernel_columns) if kernel_columns else np.zeros((t.size, 0))

        # Use base weights for variant selection (all iterations except last)
        # Use tail-weighted for final amplitude (last iteration only)
        is_last_iter = (iteration == residual_reweight_iters - 1)
        if iteration == 0:
            weights = base_weights.copy()
        elif is_last_iter and use_tail_weighting and not do_best_search:
            weights = tail_weighted.copy()
        
        # NNLS solve
        a_variants = _nnls_weighted(X, y, weights)
        
        # Compute reconstruction
        yhat_iter = X @ a_variants
        residual = y - yhat_iter
        
        # === Residual-based kinetics diagnostics (principled approach) ===
        if iteration < residual_reweight_iters - 1:
            # === Diagnostic 0: Jitter saturation check ===
            # If dominant jitters consistently hit limits, template timing is wrong
            # This is a more reliable diagnostic than residual correlations
            temp_a3d = a_variants.reshape(n_events, n_template_variants, n_jitter_variants)
            temp_flat = temp_a3d.reshape(n_events, -1)
            dom_flat_idx = np.argmax(temp_flat, axis=1)
            dom_jitter_idx = dom_flat_idx % n_jitter_variants
            dom_jitters = np.array([jitter_variant_s[j] for j in dom_jitter_idx])
            
            jitter_min, jitter_max = jitter_variant_s.min(), jitter_variant_s.max()
            # Check how many events hit the jitter limits (within 0.5ms of boundary)
            n_at_min = np.sum(dom_jitters <= jitter_min + 0.0005)
            n_at_max = np.sum(dom_jitters >= jitter_max - 0.0005)
            frac_saturated = (n_at_min + n_at_max) / n_events
            
            # Jitter saturation diagnostics are suppressed to reduce log noise.
            
            # === Diagnostic 1: Post-train decay check (diagnostic only, no adjustment) ===
            # Kinetics adjustment via tau_d scaling is ineffective due to area normalization
            # in the template builder. The templates are normalized by area, so changing
            # tau_d has minimal effect on the NNLS solution.
            post_start = float(stim_times[-1]) + isi
            post_end = float(stim_times[-1]) + 3 * isi
            post_mask = (t >= post_start) & (t <= min(post_end, t[-1]))
            
            if np.sum(post_mask) > 5:
                residual_post = residual[post_mask]
                mean_resid = np.mean(residual_post)
                mean_yhat = np.mean(yhat_iter[post_mask]) + 1e-9
                relative_bias = mean_resid / mean_yhat
                
                # Post-train bias diagnostics are suppressed to reduce log noise.
            
            # Define train window for analysis
            train_start = float(stim_times[0])
            train_end = float(stim_times[-1]) + 2 * isi
            train_mask = (t >= train_start) & (t <= train_end)
            
            if np.sum(train_mask) > 10:
                t_train = t[train_mask]
                y_train = y[train_mask]
                yhat_train = yhat_iter[train_mask]
                residual_train = residual[train_mask]
                
                # === Diagnostic 1: Cumulative model correlation ===
                # If residual correlates with cumulative(yhat), decay is too slow
                # Mathematically: E[r * cumsum(yhat)] should be 0 for correct model
                cumsum_yhat = np.cumsum(yhat_train)
                cumsum_yhat_centered = cumsum_yhat - np.mean(cumsum_yhat)
                residual_centered = residual_train - np.mean(residual_train)
                
                # Correlation coefficient
                var_cumsum = np.var(cumsum_yhat_centered)
                var_resid = np.var(residual_centered)
                if var_cumsum > 1e-12 and var_resid > 1e-12:
                    corr_cumsum = np.mean(cumsum_yhat_centered * residual_centered) / np.sqrt(var_cumsum * var_resid)
                else:
                    corr_cumsum = 0.0
                
                # === Diagnostic 2: Model derivative correlation ===
                # If residual correlates with d(yhat)/dt, timing is wrong
                dyhat_dt = np.gradient(yhat_train, t_train)
                dyhat_centered = dyhat_dt - np.mean(dyhat_dt)
                var_dyhat = np.var(dyhat_centered)
                if var_dyhat > 1e-12 and var_resid > 1e-12:
                    corr_derivative = np.mean(dyhat_centered * residual_centered) / np.sqrt(var_dyhat * var_resid)
                else:
                    corr_derivative = 0.0
                
                # Correlations are diagnostic only; kinetics are not adjusted here.
                
                # === Update weights based on residual structure ===
                # Downweight regions with systematic residual bias
                window_size = max(3, int(isi / (t[1] - t[0]) / 2)) if len(t) > 1 else 3
                cumsum = np.cumsum(np.insert(residual, 0, 0))
                running_mean = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
                pad_left = window_size // 2
                pad_right = len(residual) - len(running_mean) - pad_left
                running_mean = np.concatenate([
                    np.full(pad_left, running_mean[0] if len(running_mean) > 0 else 0),
                    running_mean,
                    np.full(max(0, pad_right), running_mean[-1] if len(running_mean) > 0 else 0)
                ])[:len(residual)]
                
                abs_residual = np.abs(residual)
                mad_residual = np.median(abs_residual)
                if mad_residual > 1e-9:
                    norm_bias = np.abs(running_mean) / (1.4826 * mad_residual + 1e-9)
                    norm_residual = abs_residual / (1.4826 * mad_residual)
                    combined_penalty = np.maximum(norm_bias * 2, norm_residual)
                    residual_weight = 1.0 / (1.0 + (combined_penalty / residual_reweight_tau) ** 2)
                    weights = base_weights * residual_weight
                    weights = np.maximum(weights, 0.01 * np.max(base_weights))

    # === Final pass with tail weighting for amplitude refinement ===
    # Now that we have selected the best variants, re-fit with tail-weighted
    # This prevents last-event overshoot without affecting variant selection
    if use_tail_weighting:
        # If 'best' mode, search for optimal tau using the built design matrix X
        if do_best_search:
            resolved_tail_tau = _find_best_tail_tau(
                t, y, X, stim_times, isi, base_weights,
                peak_window_s=pw_s, verbose=True
            )
            tail_weighted = _apply_last_event_tail_weights(
                base_weights.copy(), t, stim_times, isi, resolved_tail_tau, pw_s
            )
        a_variants = _nnls_weighted(X, y, tail_weighted)
        yhat_iter = X @ a_variants

    weights_final = weights

    # Reshape to 3D: [n_events, n_template_variants, n_jitter_variants]
    a_variants_3d = a_variants.reshape(n_events, n_template_variants, n_jitter_variants)

    # === Post-train residual constraint ===
    # The post-train decay should match the sum of all event tails
    # If systematically biased, the model tails are wrong
    # We use this to validate but NOT to adjust amplitudes (that would be ad-hoc)
    post_train_start = float(stim_times[-1]) + isi  # Start after last event's peak window
    post_train_end = float(stim_times[-1]) + 3 * isi  # ~3 ISIs for decay
    post_train_mask = (t >= post_train_start) & (t <= min(post_train_end, t[-1]))
    
    if np.sum(post_train_mask) > 5:
        residual_post = (y - yhat_iter)[post_train_mask]
        y_post = y[post_train_mask]
        
        # Mean residual as fraction of signal
        mean_resid_post = np.mean(residual_post)
        signal_rms = np.sqrt(np.mean(y_post**2)) + 1e-9
        relative_to_signal = mean_resid_post / signal_rms
        
        if abs(relative_to_signal) > 0.1:
            direction = "under" if mean_resid_post > 0 else "over"
            # progress_print(f"[NNLS] Post-train residual: {direction}prediction by {abs(mean_resid_post):.3f} ({abs(relative_to_signal)*100:.1f}% of signal)")

    # Find dominant template variant and jitter for each event
    # Flatten last two dimensions to find overall max
    a_variants_2d_flat = a_variants_3d.reshape(n_events, -1)
    dominant_flat_idx = np.argmax(a_variants_2d_flat, axis=1)
    dominant_template_idx = dominant_flat_idx // n_jitter_variants
    dominant_jitter_idx = dominant_flat_idx % n_jitter_variants

    # Extract dominant jitter values
    d_events = np.array([jitter_variant_s[j_idx] for j_idx in dominant_jitter_idx], dtype=float)

    if hard_select:
        # Hard selection: choose the dominant variant per event, then refit amplitudes
        X_hard = np.zeros((t.size, n_events), float)
        for i_event in range(n_events):
            st = float(stim_times[i_event])
            tau_d = float(current_tau_d[i_event])
            frac_slow = variant_ratios[dominant_template_idx[i_event]]
            jitter_s = jitter_variant_s[dominant_jitter_idx[i_event]]
            anchor = st + current_t0 + jitter_s
            X_hard[:, i_event] = _build_variant_kernel(
                t - anchor, tau_r_s, tau_d, frac_slow,
                event_idx=i_event, n_events=n_events,
            )
        a_events = _nnls_weighted(X_hard, y, weights_final)
        yhat = X_hard @ a_events
        components = [a_events[i] * X_hard[:, i] for i in range(n_events)]
        X_used = X_hard
    else:
        # Soft selection: sum across all template and jitter variants per event
        a_events = np.sum(a_variants_3d, axis=(1, 2))
        yhat = X @ a_variants
        components = []
        for i_event in range(n_events):
            st = float(stim_times[i_event])
            tau_d = float(current_tau_d[i_event])  # Use adjusted tau_d

            # Sum contributions from all template and jitter variants for this event
            comp_event = np.zeros_like(y)
            for i_template, frac_slow in enumerate(variant_ratios):
                for i_jitter, jitter_s in enumerate(jitter_variant_s):
                    idx = i_event * (n_template_variants * n_jitter_variants) + i_template * n_jitter_variants + i_jitter
                    amp_var = a_variants[idx]
                    if amp_var > 0:
                        anchor = st + current_t0 + jitter_s  # Use adjusted t0
                        k = _build_variant_kernel(
                            t - anchor, tau_r_s, tau_d, frac_slow,
                            event_idx=i_event, n_events=n_events,
                        )
                        comp_event += amp_var * k
            components.append(comp_event)
        X_used = X

    # Diagnostic info: which variants were selected for each event
    # Sum over jitter dimension to get template variant distribution
    template_variant_sums = np.sum(a_variants_3d, axis=2)  # [n_events, n_template_variants]
    dominant_template_ratio = None
    is_scalar_variants = all(np.isscalar(v) for v in variant_ratios)
    if is_scalar_variants:
        dominant_template_ratio = np.array([variant_ratios[idx] for idx in dominant_template_idx], dtype=float)
    else:
        dominant_template_ratio = [variant_ratios[idx] for idx in dominant_template_idx]
    dominant_jitter_ms = d_events * 1000.0

    variant_info = {
        'n_template_variants': n_template_variants,
        'n_jitter_variants': n_jitter_variants,
        'variant_ratios': variant_ratios,
        'jitter_variant_ms': jitter_variant_s * 1000.0,  # Convert back to ms for display
        'amplitudes_3d': a_variants_3d,  # [n_events, n_template_variants, n_jitter_variants]
        'template_variant_sums': template_variant_sums,  # [n_events, n_template_variants]
        'dominant_template_idx': dominant_template_idx,
        'dominant_template_ratio': dominant_template_ratio,
        'dominant_jitter_idx': dominant_jitter_idx,
        'dominant_jitter_ms': dominant_jitter_ms,
        'kinetics_adjusted': kinetics_adjusted,
        'final_tau_d': current_tau_d,
        'final_t0': current_t0,
        'hard_select': bool(hard_select),
    }

    return a_events, d_events, X_used, yhat, components, variant_info


def fit_amplitudes_sequential(
    y: np.ndarray,
    t: np.ndarray,
    stim_times: np.ndarray,
    tau_r_s: float,
    tau_d_vec_s: np.ndarray,
    *,
    variant_ratios: List[Any],
    isi: float,
    event_t0_s: float = 0.0,
    jitter_variant_ms: Optional[np.ndarray] = None,
    peak_window_s: Optional[float] = None,
    peak_weight: float = 1.0,
    weight_mode: str = 'uniform',
    weight_tau_s: Optional[float] = None,
    last_event_post_s: float = 0.200,
    last_event_tail_tau_s: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], Dict]:
    """Greedy sequential NNLS: fit each event on the residual after subtracting previous event tails.

    Resolves the fast/superslow degeneracy that arises in simultaneous fitting when the
    accumulated superslow tail from early events looks like a new release at later events.

    For each event i:
      1. residual_i = y - sum(all previously fitted components)
      2. For each (frac_slow, jitter) candidate, solve for non-negative amplitude by
         projection (dot-product least-squares) in a window [t_i, t_i + ISI - margin]
      3. Pick the candidate minimising RMS in that window
      4. Subtract the fitted component from the running accumulated signal

    Returns the same tuple format as fit_amplitudes_with_template_variants so that the
    existing two-pass smoothing logic (which smooths frac_slow / frac_superslow across
    events and re-fits with fixed kinetics) applies unchanged.
    """
    n_events = len(stim_times)
    jitter_s_arr = (
        np.asarray(jitter_variant_ms, float) / 1000.0
        if jitter_variant_ms is not None
        else np.array([0.0])
    )
    # Clamp jitter so anchor never falls before the stimulus (t0+jitter >= 0)
    # or beyond the next stimulus (t0+jitter < ISI).
    jitter_s_arr = np.clip(jitter_s_arr, -event_t0_s, isi - event_t0_s - 0.001)
    jitter_s_arr = np.unique(jitter_s_arr)  # drop duplicates from clipping
    pw_s = peak_window_s if peak_window_s is not None else 0.010
    margin_s = 0.002  # 2 ms clearance before next stimulus

    # Compute base weights using the same scheme as the simultaneous fit
    base_weights = _calculate_nnls_weights(
        t, stim_times, isi, weight_mode, weight_tau_s,
        y_ref=y,
        peak_window_s=peak_window_s,
        peak_weight=peak_weight,
    )
    # Apply last-event tail downweighting to base weights
    base_weights = _apply_last_event_tail_weights(
        base_weights, t, stim_times, isi, last_event_tail_tau_s, pw_s
    )

    a_events = np.zeros(n_events)
    d_events = np.zeros(n_events)
    dominant_ratios: List[Any] = []
    dominant_jitter_idx_list: List[int] = []
    kernels_raw: List[np.ndarray] = []   # best kernel shape per event (unit: kernel units)
    accumulated = np.zeros_like(y)

    for i_event in range(n_events):
        st = float(stim_times[i_event])
        tau_d = float(tau_d_vec_s[i_event])
        residual = y - accumulated

        t_start = st + event_t0_s
        if i_event < n_events - 1:
            t_end = float(stim_times[i_event + 1]) - margin_s
        else:
            t_end = min(t[-1], t_start + last_event_post_s)
        t_end = min(t_end, t[-1])
        win_mask = (t >= t_start) & (t <= t_end)

        if not np.any(win_mask):
            kernels_raw.append(np.zeros_like(y))
            dominant_ratios.append(variant_ratios[0])
            dominant_jitter_idx_list.append(0)
            continue

        best_rms = float('inf')
        best_amp = 0.0
        best_k_raw = np.zeros_like(y)
        best_jitter_s = 0.0
        best_ratio = variant_ratios[0]
        best_jitter_idx = 0

        r_win = residual[win_mask]

        # Use the pre-computed base weights (savgol / exponential / peak etc.)
        w_win = base_weights[win_mask].copy()

        for i_ratio, frac_slow in enumerate(variant_ratios):
            for i_jitter, jitter_s in enumerate(jitter_s_arr):
                anchor = st + event_t0_s + jitter_s
                k = _build_variant_kernel(
                    t - anchor, tau_r_s, tau_d, frac_slow,
                    event_idx=i_event, n_events=n_events,
                )
                k_win = k[win_mask]
                wk = w_win * k_win
                wr = w_win * r_win
                denom = np.dot(wk, wk)
                if denom < 1e-12:
                    continue
                amp = max(0.0, float(np.dot(wr, wk) / denom))
                rms = float(np.sqrt(np.mean((wr - amp * wk) ** 2)))
                if rms < best_rms:
                    best_rms = rms
                    best_amp = amp
                    best_k_raw = k
                    best_jitter_s = jitter_s
                    best_ratio = frac_slow
                    best_jitter_idx = i_jitter

        kernels_raw.append(best_k_raw)
        a_events[i_event] = best_amp
        d_events[i_event] = best_jitter_s
        dominant_ratios.append(best_ratio)
        dominant_jitter_idx_list.append(best_jitter_idx)
        accumulated += best_amp * best_k_raw

    # Forward-pass amplitudes are the correct solution: each event was fitted on
    # the residual after subtracting all previous event tails, so overlap is already
    # accounted for sequentially. A joint NNLS re-fit against the full signal would
    # reintroduce the fast/superslow degeneracy we are trying to avoid.
    components = [a_events[i] * kernels_raw[i] for i in range(n_events)]

    yhat = sum(components) if components else np.zeros_like(y)

    # Normalised design matrix (unit-peak per event) for downstream compatibility
    X_used = np.column_stack([
        k / max(float(np.max(np.abs(k))), 1e-12) for k in kernels_raw
    ]) if n_events > 0 else np.zeros((t.size, 0))

    dominant_jitter_ms = np.array(
        [jitter_s_arr[j] for j in dominant_jitter_idx_list], dtype=float
    ) * 1000.0

    variant_info: Dict[str, Any] = {
        'n_template_variants': len(variant_ratios),
        'n_jitter_variants': len(jitter_s_arr),
        'variant_ratios': variant_ratios,
        'dominant_template_ratio': dominant_ratios,
        'dominant_jitter_ms': dominant_jitter_ms,
        'dominant_jitter_idx': dominant_jitter_idx_list,
        'hard_select': True,   # sequential is implicitly hard-select per event
        'sequential': True,
        'final_tau_d': tau_d_vec_s.copy(),
        'final_t0': event_t0_s,
        'kinetics_adjusted': False,
    }

    return a_events, d_events, X_used, yhat, components, variant_info


def _build_variant_kernel(
    dt: np.ndarray,
    tau_r: float,
    tau_d: float,
    frac_slow: float,
    event_idx: Optional[int] = None,
    n_events: Optional[int] = None,
) -> np.ndarray:
    """Build kernel with specific slow component fraction.

    For iGluSnFR model with bi-exponential decay, this adjusts frac_fast parameter.
    For other models, returns standard kernel (ignoring frac_slow).

    NOTE: This function relies on module-level _VARIANT_KERNEL_BUILDER being set up.
    """
    global _VARIANT_KERNEL_BUILDER
    if _VARIANT_KERNEL_BUILDER is not None:
        try:
            return _VARIANT_KERNEL_BUILDER(dt, tau_r, tau_d, frac_slow, event_idx, n_events)
        except TypeError:
            return _VARIANT_KERNEL_BUILDER(dt, tau_r, tau_d, frac_slow)
    else:
        # Fallback: use standard kernel (no variant support)
        return _KERNEL_FUN(dt, tau_r, tau_d)


def fit_amplitudes_no_overlap_forward(
    y: np.ndarray,
    t: np.ndarray,
    stim_times: np.ndarray,
    tau_r_s: float,
    tau_d_vec_s: np.ndarray,
    *,
    pre_zoom_s: float,
    post_zoom_s: float,
    robust: bool = True,
    huber_delta: float,
    irls_iters: int,
    allow_shift: bool = True,
    delta_max_s: float,
    delta_step_s: float,
    shift_min_s: float,
    event_t0_s: float = 0.0,
):
    """Forward, non‑overlap per‑pulse fitting with micro‑shifts.

    Returns (amplitudes, shifts, design, reconstruction, components_list).
    """
    n = len(stim_times)
    a = np.zeros(n, float)
    d = np.zeros(n, float)
    residual = y.copy()
    
    event_t0_s = float(event_t0_s)

    for p in range(n):
        st = float(stim_times[p])
        anchor = st + event_t0_s
        if p < n - 1:
            next_anchor = float(stim_times[p + 1]) + event_t0_s
        else:
            next_anchor = None
        max_end = min(anchor + post_zoom_s, next_anchor) if next_anchor is not None else (anchor + post_zoom_s)
        avail_post = max(0.0, max_end - anchor)
        
        if avail_post < 1e-6:
            continue
            
        a_p, d_p = _fit_single_pulse_amp(
            residual, t, st, tau_r_s, float(tau_d_vec_s[p]),
            pre_zoom_s=pre_zoom_s, post_zoom_s=avail_post,
            robust=robust, huber_delta=huber_delta, irls_iters=irls_iters,
            allow_shift=allow_shift, delta_max_s=delta_max_s, delta_step_s=delta_step_s,
            shift_min_s=shift_min_s, event_t0_s=event_t0_s,
        )
        
        a[p] = a_p
        d[p] = d_p
        
        # Subtract this component from residual for next iteration
        k = _KERNEL_FUN(t - (anchor + d_p), tau_r_s, float(tau_d_vec_s[p]))
        comp = a_p * k
        residual = residual - comp
    
    # NOW reconstruct all components using the fitted parameters
    # These are the actual components as they appear in the final signal
    components = []
    for p in range(n):
        if a[p] > 0:  # Only create component if amplitude is non-zero
            anchor_p = float(stim_times[p]) + event_t0_s
            k = _KERNEL_FUN(t - (anchor_p + d[p]), tau_r_s, float(tau_d_vec_s[p]))
            components.append(a[p] * k)
        else:
            components.append(np.zeros_like(y))
    
    # The reconstruction is the sum of all components
    yhat = np.sum(components, axis=0) if components else np.zeros_like(y)
    
    # Build design matrix for reference
    X = (
        np.column_stack([
            _KERNEL_FUN(
                t - (float(stim_times[p]) + event_t0_s + d[p]),
                tau_r_s,
                float(tau_d_vec_s[p]),
            )
            for p in range(n)
        ])
        if n
        else np.zeros((t.size, 0))
    )
    
    return a, d, X, yhat, components


def fit_amplitudes_with_fixed_variants(
    y: np.ndarray,
    t: np.ndarray,
    stim_times: np.ndarray,
    tau_r_s: float,
    tau_d_vec_s: np.ndarray,
    *,
    dominant_template_ratios,
    pre_zoom_s: float,
    post_zoom_s: float,
    robust: bool = True,
    huber_delta: float,
    irls_iters: int,
    allow_shift: bool = True,
    delta_max_s: float,
    delta_step_s: float,
    shift_min_s: float,
    event_t0_s: float = 0.0,
):
    """Forward, non-overlap per-pulse fitting using variant kernels with fixed ratios.

    Similar to fit_amplitudes_no_overlap_forward but uses _build_variant_kernel
    with the dominant template ratios from the average fit. This ensures per-trial
    decay curves match the corresponding event tau from the average.

    Args:
        dominant_template_ratios: Per-event slow fraction ratios from average fit.
            Can be scalar array for bi-exp or tuple array for tri-exp.

    Returns (amplitudes, shifts, design, reconstruction, components_list).
    """
    n = len(stim_times)
    a = np.zeros(n, float)
    d = np.zeros(n, float)
    residual = y.copy()
    
    event_t0_s = float(event_t0_s)
    n_events = n

    for p in range(n):
        st = float(stim_times[p])
        anchor = st + event_t0_s
        if p < n - 1:
            next_anchor = float(stim_times[p + 1]) + event_t0_s
        else:
            next_anchor = None
        max_end = min(anchor + post_zoom_s, next_anchor) if next_anchor is not None else (anchor + post_zoom_s)
        avail_post = max(0.0, max_end - anchor)
        
        if avail_post < 1e-6:
            continue
        
        # Get the dominant ratio for this pulse
        frac_slow = dominant_template_ratios[p] if p < len(dominant_template_ratios) else 0.5
            
        a_p, d_p = _fit_single_pulse_amp_variant(
            residual, t, st, tau_r_s, float(tau_d_vec_s[p]), frac_slow,
            pre_zoom_s=pre_zoom_s, post_zoom_s=avail_post,
            robust=robust, huber_delta=huber_delta, irls_iters=irls_iters,
            allow_shift=allow_shift, delta_max_s=delta_max_s, delta_step_s=delta_step_s,
            shift_min_s=shift_min_s, event_t0_s=event_t0_s,
            event_idx=p, n_events=n_events,
        )
        
        a[p] = a_p
        d[p] = d_p
        
        # Subtract this component from residual for next iteration
        k = _build_variant_kernel(t - (anchor + d_p), tau_r_s, float(tau_d_vec_s[p]), frac_slow,
                                  event_idx=p, n_events=n_events)
        comp = a_p * k
        residual = residual - comp
    
    # Reconstruct all components using the fitted parameters
    components = []
    for p in range(n):
        frac_slow = dominant_template_ratios[p] if p < len(dominant_template_ratios) else 0.5
        if a[p] > 0:
            anchor_p = float(stim_times[p]) + event_t0_s
            k = _build_variant_kernel(t - (anchor_p + d[p]), tau_r_s, float(tau_d_vec_s[p]), frac_slow,
                                      event_idx=p, n_events=n_events)
            components.append(a[p] * k)
        else:
            components.append(np.zeros_like(y))
    
    yhat = np.sum(components, axis=0) if components else np.zeros_like(y)
    
    # Build design matrix for reference
    X = (
        np.column_stack([
            _build_variant_kernel(
                t - (float(stim_times[p]) + event_t0_s + d[p]),
                tau_r_s,
                float(tau_d_vec_s[p]),
                dominant_template_ratios[p] if p < len(dominant_template_ratios) else 0.5,
                event_idx=p, n_events=n_events,
            )
            for p in range(n)
        ])
        if n
        else np.zeros((t.size, 0))
    )
    
    return a, d, X, yhat, components


def _fit_single_pulse_amp_variant(
    y: np.ndarray,
    t: np.ndarray,
    stim_time: float,
    tau_r_s: float,
    tau_d_s: float,
    frac_slow: float,
    *,
    pre_zoom_s: float,
    post_zoom_s: float,
    robust: bool,
    huber_delta: float,
    irls_iters: int,
    allow_shift: bool,
    delta_max_s: float,
    delta_step_s: float,
    shift_min_s: float,
    event_t0_s: float,
    event_idx: int,
    n_events: int,
):
    """Fit single pulse amplitude using variant kernel with fixed slow fraction.
    
    Similar to _fit_single_pulse_amp but uses _build_variant_kernel.
    """
    anchor = float(stim_time) + float(event_t0_s)
    local_mask = (t >= (anchor - pre_zoom_s)) & (t <= (anchor + post_zoom_s))
    if not np.any(local_mask):
        return 0.0, 0.0
    y_seg = y[local_mask]
    best_a, best_d = 0.0, 0.0
    
    if allow_shift:
        start = max(float(shift_min_s), 0.0)
        if delta_max_s > 0 and start <= delta_max_s + 1e-12:
            pos_shifts = np.arange(start, delta_max_s + 1e-12, delta_step_s)
            shifts = np.concatenate(([0.0], pos_shifts)) if pos_shifts.size else np.array([0.0])
        else:
            shifts = np.array([0.0])
    else:
        shifts = np.array([0.0])
    shifts = np.unique(shifts.astype(float))
    
    for d in shifts:
        k_full = _build_variant_kernel(t - (anchor + d), tau_r_s, tau_d_s, frac_slow,
                                       event_idx=event_idx, n_events=n_events)
        k_loc = k_full[local_mask]
        if k_loc.size < 3 or np.all(k_loc == 0):
            continue
        a_loc = _nnls_irls_singlecol(y_seg, k_loc, robust=robust, huber_delta=huber_delta, iters=irls_iters)
        if a_loc > best_a:
            best_a, best_d = a_loc, d
    return float(best_a), float(best_d)


def compute_localmax_corrected_amps(
    t: np.ndarray,
    y: np.ndarray,
    stim_times: np.ndarray,
    win_ms: float,
    n_avg: int,
    pre_ms: float,
    d_vec: np.ndarray,
    tau_r_s: float,
    tau_d_vec_s: np.ndarray,
    *,
    event_t0_s: float = 0.0,
    guide_y: Optional[np.ndarray] = None,
):
    """Local averaged max around each stimulus, removing earlier events.

    Events are processed sequentially from start to finish. After measuring the
    peak for pulse ``p`` the corresponding kernel scaled by that peak is
    subtracted from ``y`` so that later pulses are unaffected by the earlier
    ones.
    """
    if y is None or t.size == 0 or np.size(y) == 0:
        return np.zeros(len(stim_times), float)
    y_resid = y.copy()
    guide = None if guide_y is None or np.size(guide_y) == 0 else np.asarray(guide_y, float)
    amps = []
    event_t0_s = float(event_t0_s)
    for p, st in enumerate(stim_times):
        center = float(st) + event_t0_s
        if guide is None:
            v = windowed_max(t, y_resid, [center], win_ms, int(n_avg), pre_ms)
            amp_p = float(v[0]) if np.size(v) else 0.0
        else:
            tp, _ = pick_peak_on_series(t, guide, float(st), win_ms, pre_ms)
            idx = int(np.argmin(np.abs(t - tp))) if t.size else 0
            n_use = max(1, int(n_avg))
            half = n_use // 2
            i0 = max(0, idx - half)
            i1 = min(t.size, i0 + n_use)
            i0 = max(0, i1 - n_use)
            amp_p = float(np.nanmean(y_resid[i0:i1])) if i1 > i0 else 0.0
        amps.append(amp_p)
        if (
            d_vec is not None
            and tau_d_vec_s is not None
            and len(d_vec) > p
            and len(tau_d_vec_s) > p
        ):
            k = _KERNEL_FUN(
                t - (float(st) + event_t0_s + float(d_vec[p])),
                tau_r_s,
                float(tau_d_vec_s[p]),
            )
            # Normalize kernel by its peak for proper subtraction
            k_peak = np.max(k)
            if k_peak > 1e-12:
                k = k / k_peak
            y_resid = y_resid - amp_p * k
    return np.asarray(amps, float)


def compute_peak_corrected_from_components(
    t: np.ndarray,
    y_meas: np.ndarray,
    stim_times: np.ndarray,
    components: Optional[List[np.ndarray]],
    *,
    win_ms: float,
    pre_ms: float,
) -> np.ndarray:
    """Return per‑pulse amplitudes corrected for prior‑pulse residuals.

    For each pulse ``p`` we locate the peak time on the provided measurement
    series ``y_meas`` and subtract the sum of NNLS component contributions from
    all earlier pulses evaluated at that peak time. This matches the visual
    interpretation “red dot minus white triangle”.

    Parameters
    ----------
    t : 1D np.ndarray
        Time vector
    y_meas : 1D np.ndarray
        Measurement series used to locate peaks (e.g. yhat, SavGol, or raw)
    stim_times : 1D np.ndarray
        Stimulus times in seconds
    components : list[np.ndarray] | None
        NNLS per‑pulse components (length = n_pulses). May be None (returns
        uncorrected peaks in that case).
    win_ms, pre_ms : float
        Peak search window and pre‑search lead, in milliseconds
    """
    t = np.asarray(t, float)
    y = np.asarray(y_meas, float)
    n = int(len(stim_times))
    if t.size == 0 or y.size == 0 or n == 0:
        return np.zeros(n, float)

    # Build cumulative baseline of previous pulses only: B_p(t) = sum_{k<p} comp[k](t)
    baselines = None
    if isinstance(components, list) and components and all(hasattr(c, "__len__") for c in components):
        cum = np.zeros_like(y)
        baselines = []
        for p in range(n):
            baselines.append(cum.copy())
            if p < len(components):
                cum = cum + np.asarray(components[p], float)

    amps_corr = np.zeros(n, float)
    for p, st in enumerate(stim_times):
        tp, vp = pick_peak_on_series(t, y, float(st), win_ms, pre_ms)
        idx = int(np.argmin(np.abs(t - tp))) if t.size else 0
        base_prev = float(baselines[p][idx]) if baselines is not None else 0.0
        amps_corr[p] = float(vp) - base_prev
    return amps_corr

def sample_null_amplitudes_consistent(
    y: np.ndarray,
    t: np.ndarray,
    baseline_mask: np.ndarray,
    tau_r_s: float,
    tau_d_s: float,
    *,
    train_start: float,
    f0_window_s: float,
    pre_zoom_s: float,
    post_zoom_s: float,
    robust: bool = True,
    huber_delta: float,
    irls_iters: int,
    allow_shift: bool = True,
    delta_max_s: float,
    delta_step_s: float,
    null_min_post_zoom_s: float,
    null_sim_max_points: int,
    peak_window_ms: float,
    peak_avg_points: int,
    pre_peak_ms: float,
    shift_min_s: float,
    n_samples: int = 1000,
    seed: int = 42,
    event_t0_s: float = 0.0,
):
    """Null distribution for A1 using the same single‑pulse estimator.

    Simulated start times lie in the final f0_window_s before train_start and
    are thinned to at most null_sim_max_points.
    """
    idx = np.flatnonzero(baseline_mask)
    if idx.size < 10:
        return np.array([])
    baseline_start = t[idx[0]]
    baseline_end = t[idx[-1]]
    null_start = max(baseline_start, train_start - f0_window_s)
    null_end = min(baseline_end, train_start)
    # Cover the entire f0 window; exclude starts whose peak window would
    # overlap the train onset.
    st_min = null_start
    st_max = null_end - (peak_window_ms / 1000.0)
    if st_max <= st_min:
        return np.array([])
    cand_mask = (t >= st_min) & (t <= st_max)
    starts_full = t[cand_mask]
    if starts_full.size == 0:
        return np.array([])
    limit = int(min(int(null_sim_max_points), int(n_samples)))
    rng = np.random.default_rng(seed)
    if starts_full.size > limit:
        idx_sel = rng.choice(starts_full.size, size=limit, replace=False)
        idx_sel.sort()
        starts = starts_full[idx_sel]
    else:
        starts = starts_full

    amps = []
    event_t0_s = float(event_t0_s)
    for st in starts:
        anchor = float(st) + event_t0_s
        avail_post = min(post_zoom_s, train_start - anchor - 1e-6, null_end - st)
        if avail_post < null_min_post_zoom_s:
            continue
        a_hat, d_hat = _fit_single_pulse_amp(
            y, t, float(st), tau_r_s, tau_d_s,
            pre_zoom_s=pre_zoom_s, post_zoom_s=avail_post,
            robust=robust, huber_delta=huber_delta, irls_iters=irls_iters,
            allow_shift=allow_shift, delta_max_s=delta_max_s, delta_step_s=delta_step_s,
            shift_min_s=shift_min_s, event_t0_s=event_t0_s,
        )
        t_fit = t[(t >= anchor - pre_zoom_s) & (t <= anchor + avail_post)]
        if t_fit.size:
            k_fit = _KERNEL_FUN(t_fit - (anchor + d_hat), tau_r_s, tau_d_s)
            y_evt = a_hat * k_fit
            val = windowed_max(
                t_fit,
                y_evt,
                [float(st) + event_t0_s],
                peak_window_ms,
                peak_avg_points,
                pre_peak_ms,
            )
            amps.append(float(val[0]) if np.size(val) else 0.0)
    return np.asarray(amps, float)


def baseline_threshold_and_pval(null_amps: np.ndarray, N: float, mode: str = "mad"):
    """Return (threshold, pval_fn) under a single rule.

    mode:
      - 'sd'  => mean(null)   + N * std(null)
      - 'mad' => median(null) + N * (1.4826 * MAD(null))
    pval_fn(x) = Pr(null >= x) with +1 smoothing.
    """
    if null_amps is None or np.size(null_amps) == 0 or not np.isfinite(N):
        return np.nan, (lambda x: np.nan)
    a = np.asarray(null_amps, float)
    mode_u = (mode or '').strip().lower()
    if mode_u == 'sd':
        mu = float(np.nanmean(a))
        sd = float(np.nanstd(a))
        thr = float(mu + float(N) * sd) if np.isfinite(sd) else np.nan
    else:
        med = float(np.nanmedian(a))
        mad = float(np.nanmedian(np.abs(a - med)))
        sigma_hat = 1.4826 * mad
        thr = float(med + float(N) * sigma_hat) if np.isfinite(sigma_hat) else np.nan
    def pval(x):
        return float((np.sum(a >= x) + 1) / (a.size + 1))
    return thr, pval


# -------------------------
# Bleach correction (robust mono‑exp)
# -------------------------

def _fit_monoexp_robust(
    t: np.ndarray,
    y: np.ndarray,
    mask_fit: np.ndarray,
    *,
    n_iter: int = 4,
    huber_delta: float,
    tau_range_factor,
    n_tau: int,
):
    """Robust fit of A + B exp(−t/τ) on masked samples; returns (trend, tau_best)."""
    tt = np.asarray(t, float)
    yy = np.asarray(y, float)
    m = mask_fit & np.isfinite(yy)
    if m.sum() < 10:
        return np.zeros_like(yy), np.nan
    t_fit = tt[m]
    y_fit = yy[m]
    dur = tt[-1] - tt[0]
    tmin = max(1e-6, tau_range_factor[0] * dur)
    tmax = max(tmin * 1.01, tau_range_factor[1] * dur)
    tau_grid = np.geomspace(tmin, tmax, int(n_tau))
    best = (np.inf, None, None)
    for tau in tau_grid:
        e = np.exp(-t_fit / tau)
        X = np.column_stack([np.ones_like(e), e])
        w = np.ones_like(e)
        for _ in range(max(1, int(n_iter))):
            w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
            Wsqrt = np.sqrt(w)
            Xw = X * Wsqrt[:, None]
            yw = y_fit * Wsqrt
            coef, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
            resid = y_fit - (coef[0] + coef[1] * e)
            absr = np.abs(resid)
            w = np.where(absr <= huber_delta, 1.0, huber_delta / np.maximum(absr, 1e-12))
        r2 = float(np.sum(resid**2))
        if r2 < best[0]:
            best = (r2, coef, tau)
    if best[1] is None:
        return np.zeros_like(yy), np.nan
    A, B = best[1]
    tau_best = float(best[2])
    trend = A + B * np.exp(-tt / tau_best)
    return trend, tau_best


def apply_bleach_correction(
    t: np.ndarray,
    y: np.ndarray,
    train_start_s: float,
    stim_times: np.ndarray,
    *,
    post_zoom_s: float,
    peak_window_ms: float,
    pre_peak_ms: float,
    huber_delta: float,
    tau_range_factor,
    n_tau: int,
) -> np.ndarray:
    """Correct slow bleaching by subtracting a robust mono‑exp trend.

    Fit uses pre‑train baseline and (if available) post‑train quiet region
    detected via compute_no_signal_mask to avoid stimulus‑evoked epochs.
    """
    try:
        # Fit on regions without evoked signal
        mask_quiet = compute_no_signal_mask(
            t, y, stim_times, post_zoom_s, peak_win_ms=peak_window_ms, peak_search_pre_ms=pre_peak_ms
        )
        trend, tau = _fit_monoexp_robust(
            t, y, mask_quiet, n_iter=4,
            huber_delta=huber_delta, tau_range_factor=tau_range_factor, n_tau=n_tau
        )
        baseline_sel = y[(t < train_start_s) & np.isfinite(y)]
        if baseline_sel.size == 0:
            baseline_sel = y[np.isfinite(y)]
        baseline_ref = float(np.nanmedian(baseline_sel)) if baseline_sel.size else 0.0
        corrected = y - trend + baseline_ref
        return corrected
    except Exception:
        return y.copy()


# -------------------------
# Kinetics (fast grid on average)
# -------------------------

def estimate_kinetics_from_average(
    t: np.ndarray,
    y_avg: np.ndarray,
    stim_times: np.ndarray,
    *,
    taur_grid_ms,
    taud0_grid_ms,
    slope_grid_ms,
    pre_zoom_s: float,
    post_zoom_s: float,
    isi: float = 0.05,
    weight_mode: str = 'uniform',
    weight_tau_s: Optional[float] = None,
    sg_window: int = DEFAULTS['sg_window'],
    sg_poly: int = DEFAULTS['sg_poly'],
    peak_window_s: Optional[float] = None,
    peak_weight: Optional[float] = None,
    last_event_tail_tau_s: Optional[float] = None,
    event_t0_s: float = 0.0,
) -> Tuple[float, float, float, np.ndarray]:
    """Grid search τr, τd0, slope on the average trace (zoomed window)."""
    tau_r_grid = np.array(taur_grid_ms, float) / 1000.0
    tau_d0_grid = np.array(taud0_grid_ms, float) / 1000.0
    slope_grid = np.array(slope_grid_ms, float) / 1000.0
    isi_guess = float(stim_times[1] - stim_times[0]) if len(stim_times) > 1 else isi
    anchor0 = float(stim_times[0]) + float(event_t0_s)
    zmask, _, _ = time_zoom_mask(t, anchor0, isi_guess, len(stim_times), pre_zoom_s, post_zoom_s)

    # Calculate weights for NNLS fitting
    if weight_mode == 'savgol':
        y_weight_ref = sg_smooth(fill_nans_timewise(y_avg, t), int(sg_window), int(sg_poly))
    else:
        y_weight_ref = None
    weights = _calculate_nnls_weights(
        t,
        stim_times,
        isi_guess,
        weight_mode,
        weight_tau_s,
        y_ref=y_weight_ref,
        peak_window_s=peak_window_s,
        peak_weight=peak_weight,
    )
    # Apply last-event tail downweighting
    pw_s = peak_window_s if peak_window_s is not None else 0.010
    weights = _apply_last_event_tail_weights(
        weights, t, stim_times, isi_guess, last_event_tail_tau_s, pw_s
    )

    def obj_for(tau_r, tau_d_vec):
        X = np.column_stack([
            _KERNEL_FUN(t - (float(st) + float(event_t0_s)), tau_r, td)
            for st, td in zip(stim_times, tau_d_vec)
        ])
        if X.size == 0:
            a = np.zeros(len(stim_times))
        else:
            # Use weighted NNLS
            a = _nnls_weighted(X[zmask, :], y_avg[zmask], weights[zmask])
        r = y_avg - X @ a
        return float(np.dot(r[zmask], r[zmask]) / max(1, zmask.sum()))

    best = (np.inf, 0.002, 0.006, 0.0)
    for tau_r in tau_r_grid:
        for tau_d0 in tau_d0_grid:
            for slope in slope_grid:
                tau_d_vec = tau_d0 + slope * np.arange(len(stim_times))
                if np.any(tau_d_vec <= tau_r + 0.0002):
                    val = 1e9
                else:
                    val = obj_for(tau_r, tau_d_vec)
                if val < best[0]:
                    best = (val, tau_r, tau_d0, slope)
    _, tau_r_fit, tau_d0_fit, slope_fit = best
    tau_d_vec = tau_d0_fit + slope_fit * np.arange(len(stim_times))
    # Always enforce non‑decreasing τd across pulses (simpler and more stable)
    tau_d_vec = np.maximum.accumulate(tau_d_vec)
    return float(tau_r_fit), float(tau_d0_fit), float(slope_fit), np.asarray(tau_d_vec, float)


# -------------------------
# Public API
# -------------------------

def extract_metrics(
    time: np.ndarray,
    trials: np.ndarray,
    train_start: float,
    isi: float,
    n_pulses: int,
    options: Optional[Dict] = None,
    filename: Optional[str] = None,
) -> Dict:
    """Extract per‑pulse metrics with a minimal, explicit interface.

    Options dictionary keys (all optional):
      - bleach: bool (default True) — apply mono‑exp bleaching correction
      - normalize_dff: bool (default True) — work in ΔF/F0
      - sg_window: int (default 9)
      - sg_poly: int (default 2)
      - peak_window_ms: float (default 25.0)
      - peak_avg_points: int (default 5)
      - pre_peak_ms: float (default 0.0)
      - null_N: float (default 3.0) — threshold multiplier
      - measurement: {'NNLS'|'SAVGOL'|'RAW'} (default 'NNLS') — which amplitudes are
        used for p‑values/classification
      - threshold_mode: {'auto'|'mad'|'sd'} (default 'auto') — auto = MAD for NNLS,
        SD for SAVGOL. Failure rates compare pulses 1–3 against a single baseline
        threshold computed from pulse 1; pulses 2/3 amplitudes subtract residual
        pre‑stim currents before comparison
      - allow_shift: bool (default True) — enable per‑pulse micro‑shifts
      - recut_peak_recenter: int | tuple | None (default 0) — number of samples
        permitted for peak realignment before averaging; 0/None keeps stimulus
        alignment
      - event_model: {'double_exp'|'cooperative'|'iglusnfr'|'iglusnfr_tri'} (default 'double_exp')
        — template used for NNLS fitting and residual subtraction. For 'iglusnfr_tri',
        recut fits use the bi‑exp model; superslow tau is estimated from the final event decay.
      - use_template_variants: bool (default False) — enable multi‑template NNLS
      - template_variant_ratios: list[float] — slow fraction grid (bi‑exp and tri‑exp)
      - template_variant_superslow_fracs: list[float] — tri‑exp superslow fractions
        at the final event (ramps monotonically across the train)
      - superslow_min_ratio: float (default 1.0) — disable superslow variants if
        tau_superslow < tau_slow * ratio
      - allow_tau_slow_override: bool (default True) — when superslow < slow, allow
        last‑event tau to replace recut tau_decay_slow (else clamp superslow only)
      - force_tau_slow_override: bool (default False) — always use last‑event tau
        as tau_decay_slow, regardless of recut fit (supersedes allow_tau_slow_override)
      - plot: dict with keys
          - enabled: bool (default False)
          - traces: list of {'raw','savgol','nnls'} (default ['nnls'])
          - show_decay: bool (default True)
          - trials: bool (default False) — also plot each trial with its fit
          - baseline: bool (default False) — for each trial, plot a two-panel
          figure with baseline fits + noise histogram and the train; forces
          trials=True when enabled
          - residuals: bool (default False) — add residual diagnostics panels
            comparing pre-train baseline noise vs. residuals after subtracting
            the selected measurement model (NNLS or SavGol)
    """
    # Parse options (merge into a single config dict)
    if options is not None and not isinstance(options, dict):
        raise TypeError(f"options must be a dict or None, got {type(options).__name__}")
    opts = options.copy() if isinstance(options, dict) else {}
    _plot_raw = opts.get('plot', {})
    if _plot_raw and not isinstance(_plot_raw, dict):
        raise TypeError(f"options['plot'] must be a dict, got {type(_plot_raw).__name__}")
    plot_opts = _plot_raw if isinstance(_plot_raw, dict) else {}
    _pbool = lambda key, default=False: bool(plot_opts.get(key, default))  # noqa: E731
    want_plot         = _pbool('enabled', False)
    traces            = list(plot_opts.get('traces', ['nnls']))
    show_decay        = _pbool('show_decay', True)
    plot_trials       = _pbool('trials')
    baseline_figs     = _pbool('baseline')
    plot_residuals    = _pbool('residuals')
    plot_peaks_details    = _pbool('plot_peaks_details')
    plot_residual_buildup = _pbool('residual_buildup')
    plot_nnls_residual    = _pbool('nnls_residual')
    plot_param_evolution  = _pbool('param_evolution')
    if baseline_figs:
        plot_trials = True  # baseline panel requires per-trial figures
    cfg = {**DEFAULTS, **{k: v for k, v in opts.items() if k != 'plot'}}
    if 'nnls_show_weights' in opts and 'fit_diagnostic_plot' not in opts:
        cfg['fit_diagnostic_plot'] = bool(opts.get('nnls_show_weights', False))
    # Detect legacy decay key names used before tau_decay_fast was standardised
    param_bounds = cfg.get('parameter_bounds', {})
    if param_bounds is None:
        param_bounds = {}
    elif not isinstance(param_bounds, dict):
        raise TypeError(
            f"parameter_bounds must be a dict or None, got {type(param_bounds).__name__}: {param_bounds!r}"
        )
    decay_bound_key = next(
        (_k for _k in ('tau_decay_fast', 'tau_decay', 'tau_decay_slow') if _k in param_bounds),
        None,
    )
    interpolated_settings: List[Dict[str, Any]] = []
    global_fit_params: Dict[str, float] = {}
    recut_slow_replaced = False
    cfg_em_settings = cfg.get('event_model_settings')
    if isinstance(cfg_em_settings, dict):
        # Use a shallow copy to avoid mutating the caller's dict in-place
        cfg['event_model_settings'] = dict(cfg_em_settings)
    else:
        cfg['event_model_settings'] = {}
    # Optional auto-calibration of event model from multi-trial data (run after preprocessing)
    do_bleach = bool(cfg.get('bleach', True))
    use_dff = bool(cfg.get('normalize_dff', True))
    sgW = int(cfg['sg_window'])
    sgP = int(cfg['sg_poly'])
    win_ms = float(cfg['peak_window_ms'])
    n_avg = int(cfg['peak_avg_points'])
    pre_ms = float(cfg['pre_peak_ms'])
    null_N = float(cfg['null_N'])
    meas = str(cfg.get('measurement', 'NNLS')).strip().upper()
    failm = str((cfg.get('fail_method') or meas)).strip().upper()
    thr_mode = str(cfg.get('threshold_mode', 'auto')).strip().lower()
    allow_shift = bool(cfg.get('allow_shift', True))
    peak_recenter = cfg.get('recut_peak_recenter', 0)

    # Shapes & schedule
    t = np.asarray(time, float).reshape(-1)
    Y = np.asarray(trials, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    if Y.shape[0] != t.size:
        raise ValueError("trials must have same number of samples as time")
    original_n_trials = int(Y.shape[1])
    # Drop empty trial columns (all NaN/non-finite in original input) so they
    # do not become artificial zero traces in downstream processing.
    valid_trial_cols = np.isfinite(Y).any(axis=0)
    if not np.all(valid_trial_cols):
        n_drop = int(np.size(valid_trial_cols) - np.sum(valid_trial_cols))
        progress_print(f"[preprocess] Dropping {n_drop} empty trial column(s) (all non-finite).")
        Y = Y[:, valid_trial_cols]
        kept_trial_cols_0based = np.flatnonzero(valid_trial_cols).astype(int)
        dropped_trial_cols_0based = np.flatnonzero(~valid_trial_cols).astype(int)
    else:
        kept_trial_cols_0based = np.arange(original_n_trials, dtype=int)
        dropped_trial_cols_0based = np.array([], dtype=int)
    if Y.shape[1] == 0:
        raise ValueError("No valid trial columns remain after dropping all-NaN columns.")
    # ---- ISI-aware guards and defaults ----
    isi = float(isi)
    isi_ms = isi * 1000.0
    stim_times = float(train_start) + isi * np.arange(int(n_pulses))

    # Enforce kinetics grids are within parameter bounds; generate logspace default if absent or filtered empty
    def _get_bounds(name, fallback):
        bound = param_bounds.get(name)
        if bound is None:
            return fallback
        if not isinstance(bound, (tuple, list)) or len(bound) != 2:
            raise TypeError(
                f"parameter_bounds['{name}'] must be a 2-tuple (lo, hi), got {bound!r}"
            )
        try:
            lo, hi = float(bound[0]), float(bound[1])
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"parameter_bounds['{name}'] = {bound!r}: cannot convert to float: {e}"
            ) from e
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError(
                f"parameter_bounds['{name}'] = ({lo}, {hi}): bounds must be finite"
            )
        if lo <= 0 or hi <= 0:
            raise ValueError(
                f"parameter_bounds['{name}'] = ({lo}, {hi}): bounds must be > 0"
            )
        if lo >= hi:
            raise ValueError(
                f"parameter_bounds['{name}'] = ({lo}, {hi}): lower bound must be < upper bound"
            )
        return lo, hi

    def _filter_grid_ms(grid, lo_s, hi_s):
        lo_ms, hi_ms = lo_s * 1000.0, hi_s * 1000.0
        result = []
        for v in grid:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv) and lo_ms <= fv <= hi_ms:
                result.append(fv)
        return result

    def _resolve_grid(cfg_key, lo_s, hi_s):
        """Filter existing grid to [lo_s, hi_s]; auto-generate logspace only when using the built-in default."""
        existing = cfg.get(cfg_key)
        if existing:
            filtered = _filter_grid_ms(existing, lo_s, hi_s)
            if filtered:
                return filtered
            if cfg_key in opts:
                raise ValueError(
                    f"{cfg_key}: all user-provided values are outside "
                    f"[{lo_s * 1000:.3f}, {hi_s * 1000:.3f}] ms. "
                    f"Provided: {list(existing)}. Remove the key to auto-generate."
                )
        return (np.logspace(np.log10(lo_s), np.log10(hi_s), 11) * 1000.0).tolist()

    lo_rise_s, hi_rise_s = _get_bounds('tau_rise', DEFAULT_PARAM_BOUNDS['tau_rise'])
    # Prefer tau_decay_fast; fall back to legacy key (tau_decay / tau_decay_slow) if only that is present
    effective_decay_key = 'tau_decay_fast' if ('tau_decay_fast' in param_bounds or not decay_bound_key) else decay_bound_key
    lo_decay_s, hi_decay_s = _get_bounds(effective_decay_key, DEFAULT_PARAM_BOUNDS['tau_decay_fast'])

    cfg['kin_taur_grid_ms'] = _resolve_grid('kin_taur_grid_ms', lo_rise_s, hi_rise_s)
    cfg['kin_taud0_grid_ms'] = _resolve_grid('kin_taud0_grid_ms', lo_decay_s, hi_decay_s)

    # Peak window: auto if not user-overridden or too wide for ISI
    if ('peak_window_ms' not in cfg) or (float(cfg['peak_window_ms']) >= isi_ms):
        cfg['peak_window_ms'] = max(6.0, min(12.0, 0.45 * isi_ms))
    # Hard cap: never exceed ISI
    cfg['peak_window_ms'] = min(float(cfg['peak_window_ms']), isi_ms)
    if ('peak_avg_points' not in cfg) or (int(cfg['peak_avg_points']) > 5):
        cfg['peak_avg_points'] = 3
    if 'pre_peak_ms' not in cfg:
        cfg['pre_peak_ms'] = 0.0

    # Micro-shift: cap to a fraction of ISI
    cfg['delta_max_s'] = min(float(cfg.get('delta_max_s', 0.002)), 0.25 * isi)

    # Null-fit single-pulse needs some post window before next stim
    cfg['null_min_post_zoom_s'] = min(float(cfg.get('null_min_post_zoom_s', 0.05)), 0.5 * isi)

    # Recut event: keep 3 ms guard before next stimulus, never exceed 20 ms
    _POST_EVT_MS = max(6.0, min(20.0, isi_ms - 3.0))

    # Preprocess trials: interpolate NaNs, optional bleach, then ΔF/F0 baseline
    baseline_mask = (t < float(train_start))
    if baseline_mask.sum() < 5:
        raise ValueError(
            f"Fewer than 5 baseline samples before train_start={float(train_start):.6f} s "
            f"(found {baseline_mask.sum()}). Time range: {t[0]:.6f}–{t[-1]:.6f} s. "
            f"Check that train_start and time are in the same units."
        )
    Yc = np.zeros_like(Y)
    for j in range(Y.shape[1]):
        yj = fill_nans_timewise(Y[:, j], t)
        if do_bleach:
            yj = apply_bleach_correction(
                t, yj, float(train_start), stim_times,
                post_zoom_s=cfg['post_zoom_s'], peak_window_ms=cfg['peak_window_ms'], pre_peak_ms=cfg['pre_peak_ms'],
                huber_delta=cfg['bleach_huber_delta'], tau_range_factor=cfg['bleach_tau_range_factor'], n_tau=cfg['bleach_n_tau']
            )
        Yc[:, j] = yj
    # F0 per trial over full pre-train baseline
    F0 = np.zeros(Yc.shape[1])
    for j in range(Yc.shape[1]):
        b = Yc[baseline_mask, j]
        b = b[np.isfinite(b)]
        if not b.size:
            raise ValueError(
                f"Trial column {kept_trial_cols_0based[j]}: no finite values in baseline window. "
                f"Check signal quality around train_start={float(train_start):.6f} s."
            )
        F0[j] = np.nanmedian(b)
    if use_dff:
        safe_F0 = np.where(np.abs(F0) < cfg['f0_eps'], np.nan, F0)
        Yd = (Yc - F0) / safe_F0
        bad_cols = ~np.isfinite(Yd).any(axis=0)
        if np.any(bad_cols):
            raise ValueError(
                f"ΔF/F₀ failed for {bad_cols.sum()} trial(s) "
                f"(columns {kept_trial_cols_0based[bad_cols].tolist()}): F₀ is zero or near-zero. "
                f"Set normalize_dff=False or check baseline quality."
            )
    else:
        Yd = Yc - F0

    # Sanitize: fill any remaining NaNs per trial to avoid failures in NNLS/curve_fit
    for j in range(Yd.shape[1]):
        if not np.all(np.isfinite(Yd[:, j])):
            Yd[:, j] = fill_nans_timewise(Yd[:, j], t)

    # Configure kernel function for the chosen event model (τ‑varying or fixed template)
    # Default must match DEFAULTS['event_model'] for consistency
    event_model = str(cfg.get('event_model', DEFAULTS.get('event_model', 'double_exp'))).strip().lower()
    em_settings = cfg.get('event_model_settings', {}) or {}
    if not isinstance(em_settings, dict):
        raise ValueError("event_model_settings must be a dict of parameter initial guesses")
    coop_n_default = float(em_settings.get('n_coop', 2.0))

    def _build_kernel_from_library(name: str):
        """Return (kernel_fun, spec) using Model_Calibration.event_models."""
        try:
            try:
                from Model_Calibration.event_models import get_event_model
            except Exception:
                from event_models import get_event_model  # type: ignore
            spec = get_event_model(name)
        except Exception as e:
            raise ValueError(f"Unknown library model '{name}': {e}")
        def _kernel_from_params(params_fixed: Dict[str, float]):
            # returns kernel(dt_s, tau_r, tau_d) ignoring tau_r/tau_d (fixed template)
            # Amplitude is peak-scaled as defined by the model spec (no area normalization).
            def kfun(dt_s: np.ndarray, tau_r: float, tau_d: float) -> np.ndarray:
                dt_s = np.asarray(dt_s, float)
                dt_ms = dt_s * 1000.0
                pars = []
                for p in spec['params']:
                    if p == 'amp':
                        pars.append(1.0)
                    elif p == 't_onset':
                        pars.append(0.0)
                    else:
                        pars.append(params_fixed[p])
                y = spec['func'](dt_ms, *pars)
                return y
            return kfun
        def _kernel_tau_varying(mapper):
            # mapper builds full param vector from tau_r,tau_d; no extra normalization
            def kfun(dt_s: np.ndarray, tau_r: float, tau_d: float) -> np.ndarray:
                dt_s = np.asarray(dt_s, float)
                dt_ms = dt_s * 1000.0
                pars = mapper(float(tau_r), float(tau_d))
                y = spec['func'](dt_ms, *pars)
                return y
            return kfun
        return spec, _kernel_from_params, _kernel_tau_varying

    global _KERNEL_FUN
    # Store model spec for progression rules
    model_spec = None
    # Helper to set kernel from current cfg and return effective (event_model, n_coop|None)
    def _apply_event_model_from_cfg(verbose: bool = True) -> Tuple[str, Optional[float]]:
        nonlocal event_model, coop_n_default, em_settings, model_spec
        # τ‑varying supported directly
        varying_supported = {'double_exp', 'cooperative', 'bilinear'}
        evm = str(cfg.get('event_model', event_model)).strip().lower()
        em_settings = cfg.get('event_model_settings', {}) or {}
        if not isinstance(em_settings, dict):
            raise ValueError("event_model_settings must be a dict of parameter initial guesses")
        if evm.startswith('library:'):
            _lib = evm.split(':', 1)[1].strip().lower()
            if _lib in varying_supported:
                evm = _lib
        if evm in varying_supported:
            spec, _, make_var = _build_kernel_from_library(evm if evm != 'double_exp' else 'double_exp')
            model_spec = spec  # Store for progression rules
            if evm == 'cooperative':
                extra_keys = set(em_settings.keys()) - {'n_coop'}
                if extra_keys:
                    raise ValueError(f"Unsupported event_model_settings for cooperative: {sorted(extra_keys)}")
                coop_n_default = float(em_settings.get('n_coop', coop_n_default))
                def _map(tau_r, tau_d):
                    return [1.0, float(tau_r), float(tau_d), float(coop_n_default), 0.0]
                _KERNEL_FUN = make_var(_map)
                if verbose:
                    progress_print(f"[model] Using event model 'cooperative' (τ‑varying), n_coop={coop_n_default}")
                event_model = evm
                return evm, coop_n_default
            if em_settings:
                raise ValueError(f"{evm} in τ‑varying mode does not accept event_model_settings")
            if evm == 'double_exp':
                def _map(tau_r, tau_d):
                    return [1.0, float(tau_r), float(tau_d), 0.0]
                _KERNEL_FUN = make_var(_map)
                if verbose:
                    progress_print("[model] Using event model 'double_exp' (τ‑varying)")
                event_model = evm
                return evm, None
            # bilinear
            def _map(tau_r, tau_d):
                return [1.0, float(tau_r) * 1000.0, float(tau_d) * 1000.0, 0.0]
            _KERNEL_FUN = make_var(_map)
            if verbose:
                progress_print("[model] Using event model 'bilinear' (τ‑varying)")
            event_model = evm
            return evm, None
        else:
            # Fixed template
            lib_name = evm
            if lib_name.startswith('library:'):
                lib_name = lib_name.split(':', 1)[1].strip().lower()
            if lib_name == 'iglusnfr_tri':
                # Base kernel uses bi-exponential iGluSnFR; superslow is reserved for NNLS variants.
                base_spec, make_fixed, _ = _build_kernel_from_library('iglusnfr')
                model_spec = base_spec  # Progression rules follow bi-exp kinetics
                allowed_base = set(base_spec['params']) - {'amp', 't_onset'}
                allowed_extra = {'tau_decay_superslow', 'frac_slow'}
                unknown = set(em_settings.keys()) - (allowed_base | allowed_extra)
                if unknown:
                    raise ValueError(f"Unknown event_model_settings for '{lib_name}': {sorted(unknown)}. Allowed keys: {sorted(allowed_base | allowed_extra)}")
                # Build complete base params (defaults -> bounds -> explicit overrides)
                param_bounds = cfg.get('parameter_bounds', {})
                base_defaults = [
                    ('tau_rise', 0.003),
                    ('tau_decay_fast', 0.008),
                    ('tau_decay_slow', 0.035),
                    ('frac_fast', 0.6),
                ]
                base_params = {}
                for param_name, default_value in base_defaults:
                    bound = param_bounds.get(param_name)
                    if bound and isinstance(bound, (tuple, list)) and len(bound) == 2:
                        lower, upper = float(bound[0]), float(bound[1])
                        if np.isfinite(lower) and np.isfinite(upper):
                            base_params[param_name] = (lower + upper) / 2.0
                        else:
                            base_params[param_name] = default_value
                    else:
                        base_params[param_name] = default_value
                for key in allowed_base:
                    if key in em_settings:
                        try:
                            base_params[key] = float(em_settings[key])
                        except Exception:
                            pass
                _KERNEL_FUN = make_fixed(base_params)
                if verbose:
                    progress_print("[model] Using 'iglusnfr_tri' base kernel (bi-exp); superslow added during NNLS variant screening.")
                event_model = lib_name
                return lib_name, None

            spec, make_fixed, _ = _build_kernel_from_library(lib_name)
            model_spec = spec  # Store for progression rules
            allowed = set(spec['params']) - {'amp', 't_onset'}
            unknown = set(em_settings.keys()) - allowed
            if unknown:
                raise ValueError(f"Unknown event_model_settings for '{lib_name}': {sorted(unknown)}. Allowed keys: {sorted(allowed)}")
            # Fit missing params on the average event (placeholder; uses y_avg later if needed)
            # For fixed-template, kernel ignores tau_r/tau_d and uses fitted params
            _KERNEL_FUN = make_fixed({k: float(v) for k, v in em_settings.items() if k in allowed})
            if verbose:
                progress_print(f"[model] Using fixed-template '{lib_name}' (amplitude-only per pulse).")
            event_model = lib_name
            return lib_name, None

    # First application with initial cfg
    ev_model_name, n_coop_effective = _apply_event_model_from_cfg()
    varying_supported_names = {"double_exp", "cooperative", "bilinear"}
    is_varying_model = ev_model_name in varying_supported_names
    event_model_fit = event_model
    if event_model == 'iglusnfr_tri':
        # Recut/per-event fits use bi-exponential kinetics; superslow is handled in NNLS variants.
        event_model_fit = 'iglusnfr'

    triexp_variant_grid = None

    def _build_triexp_variant_grid() -> List[Tuple[float, float]]:
        nonlocal triexp_variant_grid
        if triexp_variant_grid is not None:
            return triexp_variant_grid

        ratios = cfg.get('template_variant_ratios', [0.2, 0.4, 0.6, 0.8])
        ss_fracs = cfg.get('template_variant_superslow_fracs', [0.0, 0.1, 0.2])
        tau_slow_grid_ms = cfg.get('template_variant_tau_slow_ms', None)
        if not isinstance(ratios, (list, tuple, np.ndarray)):
            ratios = [ratios]
        if not isinstance(ss_fracs, (list, tuple, np.ndarray)):
            ss_fracs = [ss_fracs]
        tau_slow_grid_s = None
        if tau_slow_grid_ms is None:
            param_bounds = cfg.get('parameter_bounds', {}) or {}
            bound = param_bounds.get('tau_decay_slow')
            if bound and isinstance(bound, (tuple, list)) and len(bound) == 2:
                lower, upper = float(bound[0]), float(bound[1])
                if np.isfinite(lower) and np.isfinite(upper) and (upper - lower) > 0.005:
                    mid = 0.5 * (lower + upper)
                    candidates = [lower, mid, upper, lower * 1.25, lower * 1.5, lower * 2.0]
                    tau_slow_grid_s = []
                    for c in candidates:
                        if not np.isfinite(c):
                            continue
                        c = min(max(c, lower), upper)
                        if c > 0:
                            tau_slow_grid_s.append(c)
                    tau_slow_grid_s = sorted(set(tau_slow_grid_s)) if tau_slow_grid_s else None
        if tau_slow_grid_s is None and tau_slow_grid_ms is not None:
            tau_slow_grid_s = []
            for v in tau_slow_grid_ms:
                try:
                    val_s = float(v) / 1000.0
                except Exception:
                    continue
                if np.isfinite(val_s) and val_s > 0:
                    tau_slow_grid_s.append(val_s)
            tau_slow_grid_s = sorted(set(tau_slow_grid_s)) if tau_slow_grid_s else None
        grid: List[Tuple[float, float]] = []

        for r in ratios:
            try:
                r = float(r)
            except Exception:
                continue
            if not np.isfinite(r) or r < 0.0:
                continue
            for ss in ss_fracs:
                try:
                    ss = float(ss)
                except Exception:
                    continue
                if not np.isfinite(ss) or ss < 0.0:
                    continue
                if r + ss > 1.0 + 1e-9:
                    continue
                if tau_slow_grid_s:
                    for tau_slow_s in tau_slow_grid_s:
                        grid.append((r, ss, tau_slow_s))
                else:
                    grid.append((r, ss))

        if not grid:
            grid = [(0.3, 0.1)]
        triexp_variant_grid = grid
        return grid

    # Set up variant kernel builder if template variants are enabled
    global _VARIANT_KERNEL_BUILDER
    _VARIANT_KERNEL_BUILDER = None  # Reset to None by default

    if cfg.get('use_template_variants', False):
        # Template variants: build kernels with varying slow/fast component ratios
        if ev_model_name in ('iglusnfr', 'iglusnfr_tri'):
            # For iGluSnFR models, we can vary frac_fast/frac_slow parameters
            try:
                try:
                    from Model_Calibration.event_models import get_event_model
                except Exception:
                    from event_models import get_event_model  # type: ignore

                is_tri = (ev_model_name == 'iglusnfr_tri')
                spec_iglu = get_event_model(ev_model_name)
                model_func = spec_iglu['func']

                # Get base parameters: use middle of bounds if specified, otherwise defaults
                param_bounds = cfg.get('parameter_bounds', {})
                base_params = {}

                if is_tri:
                    param_defaults = [
                        ('tau_rise', 0.002),          # 2 ms
                        ('tau_decay_fast', 0.005),    # 5 ms
                        ('tau_decay_slow', 0.015),    # 15 ms
                        ('tau_decay_superslow', 0.040),  # 40 ms (will be updated from last-event decay)
                        ('frac_fast', 0.5),           # Default fast fraction
                        ('frac_slow', 0.3),           # Default intermediate fraction
                    ]
                else:
                    param_defaults = [
                        ('tau_rise', 0.003),          # 3 ms
                        ('tau_decay_fast', 0.008),    # 8 ms
                        ('tau_decay_slow', 0.035),    # 35 ms
                    ]
                
                for param_name, default_value in param_defaults:
                    bound = param_bounds.get(param_name)
                    if bound and isinstance(bound, (tuple, list)) and len(bound) == 2:
                        lower, upper = float(bound[0]), float(bound[1])
                        if np.isfinite(lower) and np.isfinite(upper):
                            base_params[param_name] = (lower + upper) / 2.0
                        else:
                            base_params[param_name] = default_value
                    else:
                        base_params[param_name] = default_value

                # Allow event_model_settings to seed initial kernel parameters
                if em_settings:
                    base_params.update(em_settings)

                if is_tri:
                    def _iglusnfr_variant_builder(
                        dt: np.ndarray,
                        tau_r: float,
                        tau_d: float,
                        frac_slow: float,
                        event_idx: Optional[int] = None,
                        n_events: Optional[int] = None,
                    ) -> np.ndarray:
                        """Build tri-exponential iGluSnFR kernel with slow fraction + ramped superslow."""
                        dt_ms = dt * 1000.0
                        # Defaults for fractions from settings (kept fixed across tau sweeps)
                        frac_fast_base = float(base_params.get('frac_fast', 0.5))
                        frac_slow_base = float(base_params.get('frac_slow', 0.3))
                        frac_fast_base = np.clip(frac_fast_base, 0.0, 1.0)
                        frac_slow_base = np.clip(frac_slow_base, 0.0, 1.0 - frac_fast_base)

                        # Interpret variant spec: tuple/list => (frac_slow, frac_superslow_max, tau_slow)
                        default_superslow = max(0.0, 1.0 - frac_fast_base - frac_slow_base)
                        frac_slow_val = None
                        frac_superslow_max = None
                        tau_slow_override = None
                        if isinstance(frac_slow, (list, tuple)) and len(frac_slow) >= 2:
                            try:
                                frac_slow_val = float(frac_slow[0])
                                frac_superslow_max = float(frac_slow[1])
                            except Exception:
                                frac_slow_val = None
                                frac_superslow_max = None
                            if len(frac_slow) >= 3:
                                try:
                                    tau_slow_override = float(frac_slow[2])
                                except Exception:
                                    tau_slow_override = None
                        elif np.isscalar(frac_slow):
                            try:
                                frac_slow_val = float(frac_slow)
                            except Exception:
                                frac_slow_val = None

                        if frac_slow_val is None or not np.isfinite(frac_slow_val):
                            frac_slow_val = frac_slow_base
                        if frac_superslow_max is None or not np.isfinite(frac_superslow_max):
                            frac_superslow_max = default_superslow

                        # Use the requested superslow fraction directly (no ramp).
                        # The variant grid already includes ss_frac=0 entries so the
                        # NNLS / sequential fit can select no superslow where appropriate.
                        frac_superslow = max(0.0, min(1.0, frac_superslow_max))
                        frac_slow_use = max(0.0, min(1.0, frac_slow_val))
                        if frac_slow_use + frac_superslow > 1.0:
                            frac_slow_use = max(0.0, 1.0 - frac_superslow)
                        frac_fast = max(0.0, 1.0 - frac_slow_use - frac_superslow)
                        frac_intermediate = frac_slow_use

                        # Use recut fit for fast/slow taus (slow scales from fast ratio)
                        tau_superslow_s = float(base_params.get('tau_decay_superslow', 0.040))
                        tau_fast_s = float(tau_d) if np.isfinite(tau_d) and tau_d > 0 else base_params.get('tau_decay_fast', 0.003)
                        tau_fast_base = float(base_params.get('tau_decay_fast', 0.003))
                        tau_slow_base = float(base_params.get('tau_decay_slow', 0.015))
                        ratio_slow = tau_slow_base / max(tau_fast_base, 1e-6)
                        if not np.isfinite(ratio_slow) or ratio_slow <= 1.0:
                            ratio_slow = 1.5
                        tau_slow_s = tau_fast_s * ratio_slow
                        if tau_slow_override is not None and np.isfinite(tau_slow_override) and tau_slow_override > 0:
                            tau_slow_s = float(tau_slow_override)
                        if tau_slow_s <= tau_fast_s or tau_slow_s >= tau_superslow_s:
                            return np.zeros_like(dt)  # hierarchy violated — zero kernel, NNLS assigns zero amplitude
                        params = [
                            1.0,  # amp
                            tau_r,  # tau_rise from global fit
                            tau_fast_s,  # tau_decay_fast in SECONDS
                            tau_slow_s,  # tau_decay_slow (seconds)
                            tau_superslow_s,  # tau_decay_superslow (seconds)
                            frac_fast,  # frac_fast varies
                            frac_intermediate,  # frac_slow (intermediate)
                            0.0,  # t_onset
                        ]
                        y = model_func(dt_ms, *params)
                        peak_val = np.max(y) if np.any(y > 0) else 1.0
                        y_norm = y / max(peak_val, 1e-12)
                        
                        # Validate monotonic decay - reject non-physical kernels
                        if not _validate_kernel_monotonic_decay(y_norm):
                            # Fallback: use pure bi-exponential (frac_superslow=0)
                            params_fallback = [
                                1.0, tau_r, tau_fast_s, tau_slow_s, tau_superslow_s,
                                frac_fast + frac_superslow, frac_intermediate, 0.0,
                            ]
                            y_fallback = model_func(dt_ms, *params_fallback)
                            peak_fb = np.max(y_fallback) if np.any(y_fallback > 0) else 1.0
                            return y_fallback / max(peak_fb, 1e-12)
                        return y_norm
                else:
                    def _iglusnfr_variant_builder(
                        dt: np.ndarray,
                        tau_r: float,
                        tau_d: float,
                        frac_slow: float,
                        event_idx: Optional[int] = None,
                        n_events: Optional[int] = None,
                    ) -> np.ndarray:
                        """Build bi-exponential iGluSnFR kernel with specific slow component fraction.
                        
                        Parameters are in seconds; model time axis is milliseconds.
                        """
                        dt_ms = dt * 1000.0
                        frac_slow_val = None
                        tau_slow_override = None
                        if isinstance(frac_slow, (list, tuple)) and len(frac_slow) >= 2:
                            try:
                                frac_slow_val = float(frac_slow[0])
                                tau_slow_override = float(frac_slow[1])
                            except Exception:
                                frac_slow_val = None
                                tau_slow_override = None
                        else:
                            try:
                                frac_slow_val = float(frac_slow)
                            except Exception:
                                frac_slow_val = None
                        if frac_slow_val is None or not np.isfinite(frac_slow_val):
                            frac_slow_val = 0.5
                        frac_fast = 1.0 - frac_slow_val
                        tau_slow_use = base_params['tau_decay_slow']
                        if tau_slow_override is not None and np.isfinite(tau_slow_override) and tau_slow_override > 0:
                            tau_slow_use = float(tau_slow_override)
                        params = [
                            1.0,  # amp
                            tau_r,  # tau_rise from global fit
                            float(tau_d) if tau_d > 0 else base_params['tau_decay_fast'],  # tau_decay_fast (seconds)
                            tau_slow_use,  # tau_decay_slow (seconds)
                            frac_fast,  # frac_fast varies across templates
                            0.0,  # t_onset
                        ]
                        y = model_func(dt_ms, *params)
                        peak_val = np.max(y) if np.any(y > 0) else 1.0
                        y_norm = y / max(peak_val, 1e-12)
                        
                        # Validate monotonic decay - reject non-physical kernels
                        if not _validate_kernel_monotonic_decay(y_norm):
                            # Fallback: use single exponential (all fast)
                            params_fallback = [1.0, tau_r, float(tau_d), tau_slow_use, 1.0, 0.0]
                            y_fallback = model_func(dt_ms, *params_fallback)
                            peak_fb = np.max(y_fallback) if np.any(y_fallback > 0) else 1.0
                            return y_fallback / max(peak_fb, 1e-12)
                        return y_norm

                _VARIANT_KERNEL_BUILDER = _iglusnfr_variant_builder
                model_type = "tri-exponential" if is_tri else "bi-exponential"
                if is_tri:
                    variant_preview = _build_triexp_variant_grid()
                    # Count bi-exp-equivalent variants (superslow_frac = 0)
                    biexp_equiv = [v for v in variant_preview if len(v) >= 2 and v[1] == 0.0]
                    progress_print(f"[model] Template variants enabled for {model_type} iGluSnFR with {len(variant_preview)} pairs ({len(biexp_equiv)} pure bi-exp)")
                else:
                    progress_print(f"[model] Template variants enabled for {model_type} iGluSnFR with ratios: {cfg.get('template_variant_ratios')}")
            except Exception as e:
                progress_print(f"[warning] Could not set up template variants for iGluSnFR: {e}")
                progress_print("[warning] Falling back to standard single-template fitting")
        else:
            progress_print(f"[warning] Template variants requested but not supported for model '{ev_model_name}'")
            progress_print("[warning] Currently only 'iglusnfr' and 'iglusnfr_tri' models support template variants")

    # No auto model selection: honor explicit event_model; otherwise use default.

    # Process NNLS weight configuration
    weight_mode = str(cfg.get('nnls_weight_mode', 'uniform')).strip().lower()
    weight_tau_s = cfg.get('nnls_weight_tau_s', None)
    if weight_tau_s is not None:
        weight_tau_s = float(weight_tau_s)
    elif weight_mode == 'linear':
        weight_tau_s = float(isi)  # Default linear decay uses ISI
    elif weight_mode == 'exponential':
        if cfg.get('fit_source', 'global') == 'global':
            # Will be set later from tau_d when available
            weight_tau_s = 0.010  # Default 10ms until tau_d is estimated
        else:
            weight_tau_s = 0.010  # Default 10ms for other modes
    elif weight_mode == 'savgol':
        weight_tau_s = None

    peak_window_s = cfg.get('nnls_peak_window_s', 0.010)
    if peak_window_s is not None:
        peak_window_s = float(peak_window_s)
    peak_weight = cfg.get('nnls_peak_weight', 3.0)
    if peak_weight is not None:
        peak_weight = float(peak_weight)

    # Average trace and kinetics
    y_avg = np.nanmean(Yd, axis=1)
    # Optional: store recut snippets for plotting/diagnostics
    recut_snippets = None
    recut_t_rel = None
    recut_avg = None

    # Helper: estimate base kinetics from recut average of all trials/events
    def _estimate_from_recut_average():
        try:
            # Use universal window: cap at 50ms, otherwise 6ms for very fast ISI
            isi_ms = isi * 1000.0
            if isi_ms > 6.0:
                post_ms_fit = min(50.0, isi_ms - 5.0)
            else:
                post_ms_fit = 6.0

            # Honor explicit top-level option 'recut_snippets'.
            need_snips = bool(
                cfg.get('plot', {}).get('enabled', False)
                or cfg.get('recut_snippets', False)
            )
            if need_snips:
                t_rel, avg, snippets = build_median_recut_waveform(
                    t, Yd, stim_times, pre_ms=5.0, post_ms=post_ms_fit,
                    peak_win_ms=cfg['peak_window_ms'], peak_search_pre_ms=cfg['pre_peak_ms'],
                    oversample=int(cfg.get('recut_oversample', 1)),
                    projection=str(cfg.get('recut_projection', 'median')).lower(),
                    peak_recenter=peak_recenter,
                    return_snippets=True,
                )

            else:
                t_rel, avg = build_median_recut_waveform(
                    t, Yd, stim_times, pre_ms=5.0, post_ms=post_ms_fit,
                    peak_win_ms=cfg['peak_window_ms'], peak_search_pre_ms=cfg['pre_peak_ms'],
                    oversample=int(cfg.get('recut_oversample', 1)),
                    projection=str(cfg.get('recut_projection', 'median')).lower(),
                    peak_recenter=peak_recenter,
                )
                snippets = None
            if t_rel is None or avg is None:
                raise ValueError('recut_average unavailable')
            # Capture snippets and recut outputs for outer scope plotting if returned
            nonlocal recut_snippets, recut_t_rel, recut_avg
            recut_t_rel = t_rel
            recut_avg  = avg
            recut_snippets = snippets
            if 'snippets' in locals():
                recut_snippets = snippets
            # Debug: print recut information
            try:
                if snippets is not None:
                    progress_print(f"[recut] Extracted {len(snippets)} snippets, time range: {t_rel[0]*1000:.1f} to {t_rel[-1]*1000:.1f} ms")
                    # Count valid data points per snippet
                    n_valid = [np.sum(np.isfinite(s)) for s in snippets]
                    progress_print(f"[recut] Valid points per snippet: min={min(n_valid)}, max={max(n_valid)}, mean={np.mean(n_valid):.1f}")
            except Exception:
                pass
            # Grid search on (tau_r, tau_d0) using current kernel
            tau_r_grid = np.array(cfg['kin_taur_grid_ms'], float) / 1000.0
            tau_d0_grid = np.array(cfg['kin_taud0_grid_ms'], float) / 1000.0
            best = (np.inf, 0.002, 0.015)
            dt = float(np.median(np.diff(t_rel)))
            dt_s = dt
            # Use _KERNEL_FUN for current model (cooperative uses n_coop)
            for tr in tau_r_grid:
                for td in tau_d0_grid:
                    k = _KERNEL_FUN(t_rel, tr, td)
                    denom = float(np.sum(k**2))
                    amp = float(np.sum(avg * k)) / denom if denom > 0 else 1.0
                    fit = amp * k
                    err = float(np.nanmean((avg - fit) ** 2))
                    if err < best[0]:
                        best = (err, tr, td)
            _, tau_r_b, tau_d0_b = best
            # default slope 0, tau_d_vec constant; progression handled below
            return float(tau_r_b), float(tau_d0_b)
        except Exception:
            return None, None

    # Choose kinetics according to fit_source
    fit_source = str(cfg.get('fit_source', 'global')).lower()
    # Normalize decay mode and accept a few synonyms/typos
    raw_mode = str(cfg.get('decay_progression_mode', 'linear')).strip().lower()
    _mode_map = {
        'linear_anchored': 'linear',
        'monotonic': 'free_monotonic',
        'free': 'free_monotonic',
        'freed_monotonic': 'free_monotonic',
    }
    dec_mode = _mode_map.get(raw_mode, raw_mode)
    if dec_mode not in {'fixed', 'free_monotonic', 'linear', 'none'}:
        try:
            progress_print(f"[warn] Unknown decay_progression_mode='{raw_mode}', falling back to 'linear'. Allowed: fixed|free_monotonic|linear|none")
        except Exception:
            pass
        dec_mode = 'linear'

    # Determine if two-pass smoothing will be used (smoothing deferred to after NNLS)
    two_pass_will_run = (
        dec_mode in ('linear', 'free_monotonic')
        and cfg.get('use_template_variants', False)
    )

    # Offset (seconds) between stimulus time and actual event onset
    event_t0_s = 0.0

    def _estimate_single_event_tau(event_idx, tau_r_local, tau_d0_fallback):
        """Fit tau_d for a single event on the average trace.

        Fits only the decay after the event peak, focusing on the post-event window
        to minimize contamination from subsequent events in the train.
        Returns the fast decay estimate, amplitude, and optional extra parameter
        estimates when available.
        """
        try:
            st = stim_times[event_idx] + event_t0_s
            # Use a focused window: from event onset to next event (or post_zoom_s if last event)
            if event_idx < len(stim_times) - 1:
                # Not the last event: fit up to the next event
                next_st = stim_times[event_idx + 1] + event_t0_s
                zmask_evt = (t >= st) & (t < next_st)
            else:
                # Last event: use full post_zoom window
                zmask_evt = (t >= st) & (t <= (st + cfg['post_zoom_s']))

            tf = t[zmask_evt]; yf = y_avg[zmask_evt]

            if tf.size < 3:
                # Not enough points to fit
                return float(tau_d0_fallback), 1.0, None

            tau_d_grid_ms = np.array(cfg['kin_taud0_grid_ms'], float)
            tau_d_grid = tau_d_grid_ms / 1000.0
            best = (np.inf, tau_d0_fallback, 1.0)

            for td in tau_d_grid:
                k = _KERNEL_FUN(tf - st, tau_r_local, td)
                denom = float(np.sum(k**2))
                amp = float(np.sum(yf * k)) / denom if denom > 0 else 1.0
                fit = amp * k
                err = float(np.nanmean((yf - fit) ** 2))
                if err < best[0]:
                    best = (err, td, amp)

            tau_evt = float(best[1]); amp_evt = float(best[2])

            extra_params = None
            # Attempt a richer per-event fit when progression is not fixed
            if dec_mode in ('linear', 'free_monotonic'):
                try:
                    local_t_ms = (tf - st) * 1000.0
                    if local_t_ms.size and np.isfinite(local_t_ms[0]):
                        local_t_ms = local_t_ms - float(local_t_ms[0])
                    fit_res = fit_average_event(
                        local_t_ms,
                        yf,
                        event_model_fit,
                        window_ms=(0.0, float(local_t_ms[-1]) if local_t_ms.size else 50.0),
                        onset_method=str(cfg.get('onset_method', 'inflection')),
                        onset_baseline_threshold=float(cfg.get('onset_baseline_threshold', 0.15)),
                        parameter_bounds=cfg.get('parameter_bounds', None),
                    )
                    if fit_res is not None:
                        params_dict = fit_res[0]
                        if isinstance(params_dict, dict):
                            extra_params = {
                                k: float(v)
                                for k, v in params_dict.items()
                                if k not in {'amp', 't_onset', '_recut'}
                            }
                except Exception:
                    extra_params = None

            return tau_evt, amp_evt, extra_params
        except Exception:
            return float(tau_d0_fallback), 1.0, None

    def _fit_all_events_on_average(tau_r_local, tau_d0_fallback):
        """Fit tau_d for each event individually on the average trace."""
        tau_vec = []
        amp_vec = []
        param_series: Dict[str, List[float]] = {}
        for i in range(n_pulses):
            tau_i, amp_i, extra = _estimate_single_event_tau(i, tau_r_local, tau_d0_fallback)
            tau_vec.append(tau_i)
            amp_vec.append(amp_i)

            filtered_params: Dict[str, float] = {}
            if isinstance(extra, dict):
                for key, val in extra.items():
                    if key in {'amp', 't_onset', '_recut'}:
                        continue
                    try:
                        filtered_params[key] = float(val)
                    except Exception:
                        filtered_params[key] = np.nan

            all_keys = set(param_series.keys()) | set(filtered_params.keys())
            for key in all_keys:
                param_series.setdefault(key, [])
                series = param_series[key]
                while len(series) < i:
                    series.append(np.nan)
                if key in filtered_params:
                    val = filtered_params[key]
                    series.append(float(val) if np.isfinite(val) else np.nan)
                else:
                    series.append(np.nan)

        tau_array = np.array(tau_vec, float)
        amp_array = np.array(amp_vec, float)
        for key, values in list(param_series.items()):
            while len(values) < n_pulses:
                values.append(np.nan)
            param_series[key] = np.array(values, float)
        try:
            tau_ms_str = ", ".join(f"{v*1000:.1f}" for v in tau_array)
            progress_print(f"[per-event fit] Individual τd (ms): [{tau_ms_str}]")
        except Exception:
            pass
        return tau_array, amp_array, param_series

    def _robust_linear_fit(x, y, max_iter=10, huber_delta=2.0):
        """Robust linear regression using IRLS with Huber weights.

        Args:
            x: Independent variable (pulse indices)
            y: Dependent variable (tau values)
            max_iter: Maximum IRLS iterations
            huber_delta: Huber threshold for outlier detection

        Returns:
            (a, b): intercept and slope
        """
        x = np.asarray(x, float)
        y = np.asarray(y, float)

        if len(x) < 2:
            return float(y[0]) if len(y) > 0 else 0.0, 0.0

        # Initial OLS fit
        A = np.column_stack([np.ones_like(x), x])
        try:
            coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            a, b = float(coef[0]), float(coef[1])
        except Exception:
            return float(y[0]), 0.0

        # IRLS iterations
        weights = np.ones_like(y)
        for iter_num in range(max_iter):
            # Compute residuals
            yfit = a + b * x
            residuals = y - yfit

            # Robust scale estimate (MAD)
            mad = float(np.nanmedian(np.abs(residuals - np.nanmedian(residuals))))
            if mad < 1e-10:
                break
            scale = 1.4826 * mad  # Scale factor for normal distribution

            # Huber weights
            normalized_resid = np.abs(residuals) / scale
            weights = np.where(
                normalized_resid <= huber_delta,
                1.0,
                huber_delta / normalized_resid
            )

            # Outlier detection (logging disabled for cleaner output)

            # Weighted least squares
            W = np.diag(weights)
            try:
                coef = np.linalg.lstsq(W @ A, W @ y, rcond=None)[0]
                a_new, b_new = float(coef[0]), float(coef[1])

                # Check convergence
                if abs(a_new - a) < 1e-6 and abs(b_new - b) < 1e-6:
                    break
                a, b = a_new, b_new
            except Exception:
                break

        return a, b

    def _monotonic_regression(y_in, direction='increasing'):
        """Apply isotonic regression to ensure monotonic values.

        Args:
            y_in: Input values
            direction: 'increasing' (non-decreasing) or 'decreasing' (non-increasing)
        """
        from scipy.interpolate import PchipInterpolator
        y = np.asarray(y_in, float)
        # Ensure monotonicity based on direction
        if direction == 'increasing':
            y_mono = np.maximum.accumulate(y)
        elif direction == 'decreasing':
            y_mono = np.minimum.accumulate(y)
        else:
            return y  # No monotonic constraint
        # Smooth with PCHIP interpolation while preserving monotonicity
        x = np.arange(len(y))
        try:
            # PCHIP preserves monotonicity if input is monotonic
            interp = PchipInterpolator(x, y_mono)
            y_smooth = interp(x)
            # Force monotonic again in case of numerical issues
            if direction == 'increasing':
                y_smooth = np.maximum.accumulate(y_smooth)
            elif direction == 'decreasing':
                y_smooth = np.minimum.accumulate(y_smooth)
            return y_smooth
        except Exception:
            return y_mono

    def _smooth_tau_arrays_together(tau_arrays, tau_fallbacks, param_names=None):
        """Smooth multiple tau arrays together while preserving their ratios.
        
        When tau_fast, tau_slow, tau_superslow are smoothed, we want to maintain
        consistent ratios across events. This function:
        1. Computes a reference tau (e.g., tau_fast) smoothed version
        2. Computes scale factor per event: smoothed_ref / raw_ref
        3. Applies the SAME scale factor to all tau arrays
        
        Args:
            tau_arrays: Dict of {name: array} for each tau parameter
            tau_fallbacks: Dict of {name: fallback_value} for each tau parameter
            param_names: Optional list of parameter names in order of importance
        
        Returns:
            Dict of {name: smoothed_array} for each tau parameter
        """
        if not tau_arrays:
            return {}
        
        # Choose reference tau (prefer tau_decay_fast, then first available)
        ref_name = None
        if param_names:
            for pn in param_names:
                if pn in tau_arrays:
                    ref_name = pn
                    break
        if ref_name is None:
            ref_name = list(tau_arrays.keys())[0]
        
        ref_raw = np.asarray(tau_arrays[ref_name], float)
        ref_fallback = tau_fallbacks.get(ref_name, 0.01)
        
        # Smooth the reference tau using _apply_progression
        _, ref_smoothed = _apply_progression(0.002, ref_raw, ref_fallback, 'global', ref_name)
        
        # Compute scale factor per event (avoid division by zero)
        scale_factor = np.ones_like(ref_raw)
        valid_mask = (np.abs(ref_raw) > 1e-9) & np.isfinite(ref_raw) & np.isfinite(ref_smoothed)
        scale_factor[valid_mask] = ref_smoothed[valid_mask] / ref_raw[valid_mask]
        
        # Apply same scale factor to all tau arrays
        result = {}
        for name, arr in tau_arrays.items():
            arr = np.asarray(arr, float)
            smoothed = arr * scale_factor
            # Ensure positive values
            smoothed = np.maximum(smoothed, 1e-4)
            result[name] = smoothed
        
        return result

    def _apply_progression(tau_r_in, tau_d_vec_in, tau_d0_in, source_method, param_name='tau_decay'):
        """Apply decay progression smoothing based on decay_progression_mode.

        This simplified version focuses on smoothing a tau vector using the configured mode.
        For two-pass NNLS, this is called AFTER the first pass to smooth per-event estimates.

        Args:
            tau_r_in: Rise time constant (passed through unchanged)
            tau_d_vec_in: Per-event decay time constants (to be smoothed)
            tau_d0_in: Fallback single decay constant
            source_method: 'global', 'average', or 'individual' (for logging only)
            param_name: Parameter name for logging
        
        Returns:
            (tau_r, smoothed_tau_vec): Rise time and smoothed decay vector
        """
        y = np.asarray(tau_d_vec_in, float)

        # Get progression rule for this parameter from model spec
        progression_rule = 'monotonic_increasing'  # default for backward compatibility
        if model_spec is not None:
            progression_rules = model_spec.get('progression_rules', {})
            if isinstance(progression_rules, dict):
                progression_rule = progression_rules.get(param_name, 'free')

        # Validate inputs
        if y.size != n_pulses:
            try:
                progress_print(f"[smooth] {param_name} has wrong size {y.size}, expected {n_pulses}. Using fallback.")
            except Exception:
                pass
            y = np.full(n_pulses, float(tau_d0_in) if np.isfinite(tau_d0_in) else 0.010)

        # Replace any NaN/Inf with fallback
        if not np.all(np.isfinite(y)):
            fallback_tau = float(tau_d0_in) if np.isfinite(tau_d0_in) else 0.010
            y[~np.isfinite(y)] = fallback_tau

        # If all values are invalid, use fallback
        if not np.isfinite(y).any():
            y = np.full(n_pulses, float(tau_d0_in) if np.isfinite(tau_d0_in) else 0.010)

        anchor_first = bool(cfg.get('anchor_first_tau', False))
        anchor_final = bool(cfg.get('anchor_final_tau', True))

        if dec_mode == 'fixed':
            # Use median tau for all events
            td_med = float(np.nanmedian(y)) if np.isfinite(y).any() else float(tau_d0_in)
            return tau_r_in, np.full(n_pulses, td_med)

        elif dec_mode == 'none':
            # No smoothing - return raw values (just ensure positive)
            return tau_r_in, np.maximum(y, 1e-4)

        elif dec_mode == 'linear':
            # Linear fit with optional anchors
            x = np.arange(n_pulses, dtype=float)
            
            if n_pulses < 2:
                return tau_r_in, np.maximum(y, 1e-4)
            
            # Robust linear fit
            a, b = _robust_linear_fit(x, y)
            
            # Apply slope constraint based on progression rule
            if progression_rule == 'monotonic_increasing':
                b = max(0.0, b)
            elif progression_rule == 'monotonic_decreasing':
                b = min(0.0, b)
            
            # Handle anchors
            if anchor_first and anchor_final and n_pulses > 2:
                tau_first, tau_final = float(y[0]), float(y[-1])
                b = (tau_final - tau_first) / max(1, n_pulses - 1)
                if progression_rule == 'monotonic_increasing':
                    b = max(0.0, b)
                elif progression_rule == 'monotonic_decreasing':
                    b = min(0.0, b)
                a = tau_first
                yfit = a + b * x
                yfit[0], yfit[-1] = tau_first, tau_final
            elif anchor_final and n_pulses > 1:
                tau_final = float(y[-1])
                a = tau_final - b * (n_pulses - 1)
                if a < 1e-4:
                    a = 1e-4
                    b = (tau_final - a) / max(1, n_pulses - 1)
                yfit = a + b * x
                yfit[-1] = tau_final
            elif anchor_first and n_pulses > 1:
                a = float(y[0])
                yfit = a + b * x
                yfit[0] = float(y[0])
            else:
                yfit = a + b * x

            # Ensure positive values
            yfit = np.maximum(yfit, 1e-4)
            return tau_r_in, yfit

        else:  # 'free_monotonic'
            # Monotonic regression using PCHIP smoothing
            mono_direction = None
            if progression_rule == 'monotonic_increasing':
                mono_direction = 'increasing'
            elif progression_rule == 'monotonic_decreasing':
                mono_direction = 'decreasing'

            # Apply monotonic regression
            if mono_direction:
                yfit = _monotonic_regression(y, mono_direction)
            else:
                yfit = y.copy()

            # Apply anchors
            if anchor_first and anchor_final and n_pulses > 2:
                yfit[0], yfit[-1] = float(y[0]), float(y[-1])
            elif anchor_final and n_pulses > 1:
                yfit[-1] = float(y[-1])
            elif anchor_first and n_pulses > 1:
                yfit[0] = float(y[0])

            # Ensure positive values
            yfit = np.maximum(yfit, 1e-4)
            return tau_r_in, yfit

    # Variables for optional display overlays and logging
    tau_last_display = None
    amp_last_display = None
    tau_d_vec_raw = None  # Initial per-event estimates before any constraints
    tau_d_vec_constrained = None  # After clipping/anchoring (for global mode)
    per_event_param_map: Dict[str, np.ndarray] = {}

    if fit_source == 'global':
        # Always use fit_average_event with consistent long window for proper kinetics fitting
        # Stim time verification disabled for cleaner output

        # Universal recut window: cap at 50ms, otherwise 6ms for very fast ISI
        isi_ms = isi * 1000.0
        if isi_ms > 6.0:
            post_ms_for_fit = min(50.0, isi_ms - 5.0)
        else:
            post_ms_for_fit = 6.0
        # Recut fitting window logging disabled for cleaner output

        # === Estimate tau_slow from last-event decay (bi-exp only) ===
        # For tri-exp, superslow is derived from the final event decay instead.
        tau_slow_from_decay = None
        if event_model != 'iglusnfr_tri':
            # Critical for high-frequency trains where recut window is too short to fit slow kinetics
            tau_fast_hint = None
            try:
                tau_fast_hint = float(cfg.get('event_model_settings', {}).get('tau_decay_fast', np.nan))
                if not np.isfinite(tau_fast_hint):
                    tau_fast_hint = None
            except Exception:
                tau_fast_hint = None
            tau_slow_from_decay, _ = estimate_tau_superslow_from_last_event_decay(
                t, y_avg, stim_times, isi, tau_fast_s=tau_fast_hint
            )
            if tau_slow_from_decay is not None:
                param_bounds = cfg.setdefault('parameter_bounds', {})
                if 'tau_decay_slow' not in param_bounds:
                    lower_slow = max(0.010, tau_slow_from_decay * 0.5)
                    upper_slow = min(0.400, tau_slow_from_decay * 3.0)
                    param_bounds['tau_decay_slow'] = (lower_slow, upper_slow)
                    progress_print(
                        f"[global] tau_slow bounds: {lower_slow*1000:.1f}-{upper_slow*1000:.1f} ms "
                        f"(from last-event: {tau_slow_from_decay*1000:.1f}ms)"
                    )

        need_snips = bool(
            cfg.get('plot', {}).get('enabled', False)
            or cfg.get('recut_snippets', False)
        )
        
        # Use early events only for fast kinetics estimation (default: first 3 events)
        early_events = int(cfg.get('early_events_only', 3))
        
        # Pass tau_slow from last-event decay to fix it during curve_fit
        # This allows fitting tau_fast from early events while using the true slow kinetics
        fixed_tau_slow = tau_slow_from_decay if (early_events > 0 and tau_slow_from_decay is not None) else None

        res = fit_average_event(
            t, Yd, event_model_fit, stim_times,
            oversample=int(cfg['recut_oversample']),
            projection=str(cfg['recut_projection']).lower(),
            peak_recenter=peak_recenter,
            return_snippets=need_snips,
            post_ms=post_ms_for_fit,  # Use long window for fitting
            onset_method=str(cfg.get('onset_method', 'inflection')),
            onset_baseline_threshold=float(cfg.get('onset_baseline_threshold', 0.15)),
            early_events_only=early_events,
            fixed_tau_slow=fixed_tau_slow,
            parameter_bounds=cfg.get('parameter_bounds', None),
        )

        fitted = None
        if res is not None:
            fitted, t_avg_evt, y_avg_evt = res

            # Pull recut outputs attached by helper
            if isinstance(fitted, dict) and ('_recut' in fitted):
                try:
                    t_rel_s, avg_s, snippets_s = fitted.pop('_recut')
                    recut_t_rel, recut_avg, recut_snippets = t_rel_s, avg_s, snippets_s
                except Exception:
                    pass

            # Apply parameter bounds to fitted values
            if isinstance(fitted, dict):
                param_bounds = cfg.get('parameter_bounds', {})
                if param_bounds:
                    fitted = apply_parameter_bounds(fitted, param_bounds)

            # Extract parameters from fitted dict
            try:
                global_fit_params = {
                    k: float(v)
                    for k, v in fitted.items()
                    if k != 'amp' and k != 't_onset' and k != '_recut'
                }
            except Exception:
                global_fit_params = {}

            tau_r = float(fitted.get('tau_rise', 0.002))
            tau_d0 = float(fitted.get('tau_decay', fitted.get('tau_decay_fast', 0.010)))
            event_t0_s = float(fitted.get('t_onset', 0.0)) / 1000.0


            # Carry over model-specific parameters (treat event_model_settings as initial guesses)
            cfg.setdefault('event_model_settings', {})
            if event_model == 'cooperative' and ('n_coop' in fitted):
                cfg['event_model_settings']['n_coop'] = float(fitted['n_coop'])
            elif event_model not in varying_supported_names:
                for k, v in fitted.items():
                    if k not in ('amp', 't_onset', '_recut') and np.isfinite(v):
                        cfg['event_model_settings'][k] = float(v)

            # For tri-exp: estimate tau_superslow from the final event decay (if not user-specified)
            if event_model == 'iglusnfr_tri':
                em_settings = cfg.get('event_model_settings', {}) or {}
                param_bounds = cfg.get('parameter_bounds', {}) or {}
                if 'tau_decay_superslow' not in em_settings:
                    tau_superslow_setting = None
                    min_superslow_frac = None
                    superslow_disabled = False
                    bound = param_bounds.get('tau_decay_superslow')
                    if bound and isinstance(bound, (tuple, list)) and len(bound) == 2:
                        lower, upper = float(bound[0]), float(bound[1])
                        if np.isfinite(lower) and np.isfinite(upper):
                            if abs(lower - upper) < 1e-12:
                                tau_superslow_setting = lower
                            else:
                                tau_superslow_setting = (lower + upper) / 2.0
                    if tau_superslow_setting is None:
                        tau_fast_fit = None
                        # Use tau_fast from event_model_settings (user-specified) for superslow estimation
                        # The global fit tau_fast can be too short due to early-events fitting,
                        # which causes the superslow estimate to be wrong.
                        # Priority: 1) event_model_settings, 2) parameter_bounds upper limit, 3) fitted value
                        tau_fast_for_estimate = None
                        try:
                            tau_fast_for_estimate = float(em_settings.get('tau_decay_fast', np.nan))
                        except Exception:
                            tau_fast_for_estimate = None
                        if tau_fast_for_estimate is None or not np.isfinite(tau_fast_for_estimate):
                            # Try parameter_bounds upper limit (expected max tau_fast)
                            try:
                                bounds = param_bounds.get('tau_decay_fast')
                                if bounds and isinstance(bounds, (tuple, list)) and len(bounds) >= 2:
                                    tau_fast_for_estimate = float(bounds[1])  # upper bound
                            except Exception:
                                tau_fast_for_estimate = None
                        if tau_fast_for_estimate is None or not np.isfinite(tau_fast_for_estimate):
                            try:
                                tau_fast_for_estimate = float(fitted.get('tau_decay_fast', np.nan))
                            except Exception:
                                tau_fast_for_estimate = None
                        tau_superslow_setting, min_superslow_frac = estimate_tau_superslow_from_last_event_decay(
                            t, y_avg, stim_times, isi, event_t0_s=event_t0_s, tau_fast_s=tau_fast_for_estimate
                        )
                    # Enforce superslow >= slow and optionally disable superslow if too close to slow
                    tau_slow_fit = None
                    try:
                        if isinstance(fitted, dict):
                            tau_slow_fit = float(fitted.get('tau_decay_slow', np.nan))
                    except Exception:
                        tau_slow_fit = None
                    if tau_slow_fit is None or not np.isfinite(tau_slow_fit):
                        try:
                            tau_slow_fit = float(em_settings.get('tau_decay_slow', np.nan))
                        except Exception:
                            tau_slow_fit = None

                    min_ratio = float(cfg.get('superslow_min_ratio', 1.0))
                    if not np.isfinite(min_ratio) or min_ratio < 1.0:
                        min_ratio = 1.0
                    if tau_superslow_setting is None:
                        cfg['template_variant_superslow_fracs'] = [0.0]
                        triexp_variant_grid = None
                        superslow_disabled = True
                        progress_print("[tri-exp] No superslow tau estimate; disabling superslow variants.")
                    else:
                        tau_superslow_setting = float(tau_superslow_setting)
                        force_slow_override = bool(cfg.get('force_tau_slow_override', False))
                        allow_slow_override = bool(cfg.get('allow_tau_slow_override', True))
                        if tau_slow_fit is not None and np.isfinite(tau_slow_fit):
                            # force_tau_slow_override: always use last-event tau as slow (supersedes allow)
                            if force_slow_override:
                                tau_slow_fit = float(tau_superslow_setting)
                                recut_slow_replaced = True
                                if isinstance(fitted, dict):
                                    fitted['tau_decay_slow'] = float(tau_slow_fit)
                                if isinstance(cfg.get('event_model_settings', None), dict):
                                    cfg['event_model_settings']['tau_decay_slow'] = float(tau_slow_fit)
                                global_fit_params['tau_decay_slow'] = float(tau_slow_fit)
                                progress_print(
                                    "[tri-exp] FORCED recut tau_decay_slow to last-event decay: "
                                    f"{tau_slow_fit*1000:.1f} ms"
                                )
                            elif tau_superslow_setting < tau_slow_fit:
                                if allow_slow_override:
                                    tau_slow_fit = float(tau_superslow_setting)
                                    recut_slow_replaced = True
                                    if isinstance(fitted, dict):
                                        fitted['tau_decay_slow'] = float(tau_slow_fit)
                                    if isinstance(cfg.get('event_model_settings', None), dict):
                                        cfg['event_model_settings']['tau_decay_slow'] = float(tau_slow_fit)
                                    global_fit_params['tau_decay_slow'] = float(tau_slow_fit)
                                    progress_print(
                                        "[tri-exp] Replaced recut tau_decay_slow with last-event decay: "
                                        f"{tau_slow_fit*1000:.1f} ms"
                                    )
                                else:
                                    tau_superslow_setting = float(tau_slow_fit)
                            if tau_superslow_setting < tau_slow_fit * min_ratio:
                                cfg['template_variant_superslow_fracs'] = [0.0]
                                triexp_variant_grid = None
                                superslow_disabled = True
                                progress_print(
                                    "[tri-exp] Superslow tau too close to slow "
                                    f"({tau_superslow_setting*1000:.1f}ms < {min_ratio:.2f}x {tau_slow_fit*1000:.1f}ms); "
                                    "disabling superslow variants."
                                )
                        # CHANGED: Always include 0.0 (bi-exp equivalent) in superslow grid
                        # so NNLS can select bi-exp if it fits better than tri-exp
                        if (not superslow_disabled) and min_superslow_frac is not None and np.isfinite(min_superslow_frac):
                            min_superslow_frac = float(np.clip(min_superslow_frac, 0.0, 1.0))
                            ss_fracs = cfg.get('template_variant_superslow_fracs', [0.0])
                            if not isinstance(ss_fracs, (list, tuple, np.ndarray)):
                                ss_fracs = [ss_fracs]
                            # Keep all original superslow fractions plus the estimate
                            ss_filtered = [0.0]  # Always include bi-exp equivalent
                            for val in ss_fracs:
                                try:
                                    fval = float(val)
                                except Exception:
                                    continue
                                if np.isfinite(fval) and fval >= 0.0:
                                    ss_filtered.append(fval)
                            # Also include the estimated fraction if not already present
                            if min_superslow_frac not in ss_filtered:
                                ss_filtered.append(min_superslow_frac)
                            ss_filtered = sorted(set(ss_filtered))
                            cfg['template_variant_superslow_fracs'] = ss_filtered
                            triexp_variant_grid = None
                            n_biexp = sum(1 for v in ss_filtered if v == 0.0)
                            progress_print(
                                f"[tri-exp] Superslow grid includes bi-exp (0.0) + estimate ({min_superslow_frac:.2f}); "
                                f"{len(ss_filtered)} fracs total"
                            )
                        cfg['event_model_settings']['tau_decay_superslow'] = float(tau_superslow_setting)

            # REBUILD variant kernel builder with fitted parameters
            if cfg.get('use_template_variants', False) and event_model in ('iglusnfr', 'iglusnfr_tri'):
                try:
                    try:
                        from Model_Calibration.event_models import get_event_model
                    except Exception:
                        from event_models import get_event_model  # type: ignore

                    is_tri = (event_model == 'iglusnfr_tri')
                    spec_iglu = get_event_model(event_model)
                    model_func = spec_iglu['func']

                    # Use event_model_settings as ground truth for kernel building when available
                    # The global fit values can be biased due to early-events fitting
                    em_settings = cfg.get('event_model_settings', {})
                    param_bounds = cfg.get('parameter_bounds', {}) or {}
                    force_slow_override = bool(cfg.get('force_tau_slow_override', False))

                    # Helper to get tau_fast from various sources
                    def _get_tau_fast_from_config():
                        # When force_slow_override is active, prefer parameter_bounds upper
                        # because em_settings gets populated from global fit which may be wrong
                        if force_slow_override:
                            bounds = param_bounds.get('tau_decay_fast')
                            if bounds and isinstance(bounds, (tuple, list)) and len(bounds) >= 2:
                                tau = float(bounds[1])  # upper bound = intended max tau_fast
                                return tau
                        # Normal priority: 1) em_settings, 2) parameter_bounds upper, 3) fitted
                        tau = float(em_settings.get('tau_decay_fast', np.nan))
                        if not np.isfinite(tau):
                            bounds = param_bounds.get('tau_decay_fast')
                            if bounds and isinstance(bounds, (tuple, list)) and len(bounds) >= 2:
                                tau = float(bounds[1])  # upper bound
                        if not np.isfinite(tau):
                            tau = float(fitted.get('tau_decay_fast', 0.005))
                        return tau

                    # Get tau values - prefer em_settings when force_tau_slow_override is active
                    override_used = False
                    if force_slow_override and is_tri:
                        # When force override is active, use event_model_settings values
                        # which have been set to the last-event decay estimate
                        tau_fast = _get_tau_fast_from_config()
                        tau_slow = float(em_settings.get('tau_decay_slow', np.nan))
                        if not np.isfinite(tau_slow):
                            tau_slow = float(fitted.get('tau_decay_slow', 0.015))
                        override_used = True
                    else:
                        # Normal case: use fitted values, fall back to em_settings
                        tau_fast = float(fitted.get('tau_decay_fast', np.nan))
                        tau_slow = float(fitted.get('tau_decay_slow', np.nan))
                        if not np.isfinite(tau_fast):
                            tau_fast = float(em_settings.get('tau_decay_fast', 0.005))
                            override_used = True
                        if not np.isfinite(tau_slow):
                            tau_slow = float(em_settings.get('tau_decay_slow', 0.015))
                            override_used = True
                    tau_superslow = float(em_settings.get('tau_decay_superslow', fitted.get('tau_decay_superslow', 0.040))) if is_tri else None

                    # Apply cap to tau_decay_slow if specified
                    max_tau_slow = cfg.get('max_tau_decay_slow', None)
                    if max_tau_slow is not None and tau_slow > max_tau_slow:
                        tau_slow = float(max_tau_slow)

                    fitted_params = {
                        'tau_decay_fast': tau_fast,
                        'tau_decay_slow': tau_slow,
                    }
                    if is_tri:
                        fitted_params['tau_decay_superslow'] = tau_superslow
                    # Check if force override is active (tau_slow == tau_superslow)
                    if is_tri and abs(tau_slow - tau_superslow) < 1e-6:
                        progress_print(f"[model] Force override confirmed: tau_slow=tau_superslow={tau_slow*1000:.1f}ms")

                    if is_tri:
                        # Tri-exponential variant builder
                        def _iglusnfr_variant_builder_fitted(
                            dt: np.ndarray,
                            tau_r: float,
                            tau_d: float,
                            frac_slow: float,
                            event_idx: Optional[int] = None,
                            n_events: Optional[int] = None,
                        ) -> np.ndarray:
                            """Build tri-exponential iGluSnFR kernel using fitted kinetics + ramped superslow."""
                            dt_ms = dt * 1000.0
                            # Defaults for fractions from settings (kept fixed across tau sweeps)
                            frac_fast_base = float(em_settings.get('frac_fast', fitted.get('frac_fast', 0.5)))
                            frac_slow_base = float(em_settings.get('frac_slow', fitted.get('frac_slow', 0.3)))
                            frac_fast_base = np.clip(frac_fast_base, 0.0, 1.0)
                            frac_slow_base = np.clip(frac_slow_base, 0.0, 1.0 - frac_fast_base)

                            default_superslow = max(0.0, 1.0 - frac_fast_base - frac_slow_base)
                            frac_slow_val = None
                            frac_superslow_max = None
                            tau_slow_override = None
                            if isinstance(frac_slow, (list, tuple)) and len(frac_slow) >= 2:
                                try:
                                    frac_slow_val = float(frac_slow[0])
                                    frac_superslow_max = float(frac_slow[1])
                                except Exception:
                                    frac_slow_val = None
                                    frac_superslow_max = None
                                if len(frac_slow) >= 3:
                                    try:
                                        tau_slow_override = float(frac_slow[2])
                                    except Exception:
                                        tau_slow_override = None
                            elif np.isscalar(frac_slow):
                                try:
                                    frac_slow_val = float(frac_slow)
                                except Exception:
                                    frac_slow_val = None

                            if frac_slow_val is None or not np.isfinite(frac_slow_val):
                                frac_slow_val = frac_slow_base
                            if frac_superslow_max is None or not np.isfinite(frac_superslow_max):
                                frac_superslow_max = default_superslow

                            # Use the requested superslow fraction directly (no ramp).
                            # Spillover-driven superslow progression is handled by the
                            # variant grid (ss_frac=0 available) + two-pass smoothing.
                            frac_superslow = max(0.0, min(1.0, frac_superslow_max))
                            frac_slow_use = max(0.0, min(1.0, frac_slow_val))
                            if frac_slow_use + frac_superslow > 1.0:
                                frac_slow_use = max(0.0, 1.0 - frac_superslow)
                            frac_fast = max(0.0, 1.0 - frac_slow_use - frac_superslow)
                            frac_intermediate = frac_slow_use

                            tau_superslow_s = float(fitted_params.get('tau_decay_superslow', 0.040))
                            tau_fast_base = float(fitted_params.get('tau_decay_fast', 0.003))
                            tau_slow_base = float(fitted_params.get('tau_decay_slow', 0.015))
                            # Use recut-derived taus for NNLS variants; only ratios vary.
                            tau_fast_s = float(tau_fast_base)
                            tau_slow_s = float(tau_slow_base)
                            if tau_slow_override is not None and np.isfinite(tau_slow_override) and tau_slow_override > 0:
                                tau_slow_s = float(tau_slow_override)
                            if tau_slow_s <= tau_fast_s or tau_slow_s >= tau_superslow_s:
                                return np.zeros_like(dt)  # hierarchy violated — zero kernel, NNLS assigns zero amplitude
                            params = [
                                1.0,  # amp (will be normalized)
                                tau_r,  # tau_rise from global fit
                                tau_fast_s,  # tau_decay_fast in SECONDS
                                tau_slow_s,  # tau_decay_slow (seconds)
                                tau_superslow_s,  # tau_decay_superslow (seconds)
                                frac_fast,  # fast fraction varies
                                frac_intermediate,  # intermediate fraction
                                0.0,  # t_onset
                            ]
                            y = model_func(dt_ms, *params)
                            peak_val = np.max(y) if np.any(y > 0) else 1.0
                            y_norm = y / max(peak_val, 1e-12)
                            
                            # Validate monotonic decay - reject non-physical kernels
                            if not _validate_kernel_monotonic_decay(y_norm):
                                # Fallback: use pure bi-exponential (frac_superslow=0)
                                params_fb = [
                                    1.0, tau_r, tau_fast_s, tau_slow_s, tau_superslow_s,
                                    frac_fast + frac_superslow, frac_intermediate, 0.0,
                                ]
                                y_fb = model_func(dt_ms, *params_fb)
                                peak_fb = np.max(y_fb) if np.any(y_fb > 0) else 1.0
                                return y_fb / max(peak_fb, 1e-12)
                            return y_norm
                    else:
                        # Bi-exponential variant builder
                        def _iglusnfr_variant_builder_fitted(
                            dt: np.ndarray,
                            tau_r: float,
                            tau_d: float,
                            frac_slow: float,
                            event_idx: Optional[int] = None,
                            n_events: Optional[int] = None,
                        ) -> np.ndarray:
                            """Build iGluSnFR kernel with specific slow component fraction using FITTED kinetics.
                            
                            Parameters are in seconds; model time axis is milliseconds.
                            """
                            dt_ms = dt * 1000.0
                            frac_slow_val = None
                            tau_slow_override = None
                            if isinstance(frac_slow, (list, tuple)) and len(frac_slow) >= 2:
                                try:
                                    frac_slow_val = float(frac_slow[0])
                                    tau_slow_override = float(frac_slow[1])
                                except Exception:
                                    frac_slow_val = None
                                    tau_slow_override = None
                            else:
                                try:
                                    frac_slow_val = float(frac_slow)
                                except Exception:
                                    frac_slow_val = None
                            if frac_slow_val is None or not np.isfinite(frac_slow_val):
                                frac_slow_val = 0.5
                            frac_fast = 1.0 - frac_slow_val
                            tau_slow_use = fitted_params['tau_decay_slow']
                            if tau_slow_override is not None and np.isfinite(tau_slow_override) and tau_slow_override > 0:
                                tau_slow_use = float(tau_slow_override)
                            # Use recut-derived fast tau for NNLS variants; only ratios vary.
                            tau_fast_use = fitted_params['tau_decay_fast']
                            params = [
                                1.0,  # amp (will be normalized)
                                tau_r,  # tau_rise from global fit
                                tau_fast_use,  # FITTED fast decay (seconds)
                                tau_slow_use,  # FITTED slow decay (seconds)
                                frac_fast,  # frac_fast varies across templates
                                0.0,  # t_onset
                            ]
                            y = model_func(dt_ms, *params)
                            peak_val = np.max(y) if np.any(y > 0) else 1.0
                            y_norm = y / max(peak_val, 1e-12)
                            
                            # Validate monotonic decay - reject non-physical kernels
                            if not _validate_kernel_monotonic_decay(y_norm):
                                # Fallback: use single exponential (all fast)
                                params_fb = [1.0, tau_r, tau_fast_use, tau_slow_use, 1.0, 0.0]
                                y_fb = model_func(dt_ms, *params_fb)
                                peak_fb = np.max(y_fb) if np.any(y_fb > 0) else 1.0
                                return y_fb / max(peak_fb, 1e-12)
                            return y_norm

                    _VARIANT_KERNEL_BUILDER = _iglusnfr_variant_builder_fitted
                    override_msg = " (OVERRIDDEN)" if override_used else ""
                    # Detect if force_tau_slow_override is active
                    force_slow_msg = ""
                    if is_tri and tau_superslow is not None and abs(tau_slow - tau_superslow) < 1e-6:
                        force_slow_msg = " (tau_slow FORCED to superslow)"
                    if is_tri:
                        progress_print(f"[model] Updated tri-exp variant kernel{override_msg}{force_slow_msg}: tau_fast={tau_fast*1000:.2f}ms, tau_slow={tau_slow*1000:.2f}ms, tau_superslow={tau_superslow*1000:.2f}ms")
                    else:
                        progress_print(f"[model] Updated variant kernel{override_msg}: tau_fast={tau_fast*1000:.2f}ms, tau_slow={tau_slow*1000:.2f}ms")
                except Exception as e:
                    progress_print(f"[warning] Failed to update variant kernel builder: {e}")

        else:
            # Fallback if curve_fit completely failed
            progress_print(f"[global] fit_average_event failed, using grid search fallback")
            tr_b, td0_b = _estimate_from_recut_average()
            tau_r = float(tr_b) if tr_b is not None else 0.002
            tau_d0 = float(td0_b) if td0_b is not None else 0.010
            global_fit_params = {'tau_rise': tau_r, 'tau_decay': tau_d0}
            event_t0_s = 0.0

            # Use recut data for plotting
            if (recut_t_rel is not None) and (recut_avg is not None):
                t_avg_evt = np.asarray(recut_t_rel, float) * 1000.0
                y_avg_evt = np.asarray(recut_avg, float)
            else:
                t_avg_evt = (t - float(train_start)) * 1000.0
                y_avg_evt = y_avg

        ev_model_name, n_coop_effective = _apply_event_model_from_cfg(verbose=False)  # Suppress duplicate message
        is_varying_model = ev_model_name in varying_supported_names
        progress_print(
            f"[fit][global] recut tau_r={tau_r*1000:.2f}ms "
            f"tau_d={tau_d0*1000:.2f}ms "
            f"t_onset={event_t0_s*1000:.2f}ms model={event_model}"
        )

        # Global fit_source: anchor global tau to middle event, then constrain per-event fits
        if dec_mode == 'fixed':
            # Fixed mode: use single global tau for all events
            tau_d_vec0 = np.full(n_pulses, float(tau_d0))
            tau_d_vec_raw = np.asarray(tau_d_vec0, float)
            tau_d_vec_constrained = None  # No constraints applied in fixed mode
        elif dec_mode in ('linear', 'free_monotonic'):
            # Global mode: keep a single tau_d unless anchors are requested.
            anchor_first = bool(cfg.get('anchor_first_tau', False))
            anchor_final = bool(cfg.get('anchor_final_tau', True))

            if anchor_first or anchor_final:
                tau_first = float(tau_d0)
                tau_last = float(tau_d0)
                amp_first = None
                amp_last = None
                if anchor_first:
                    tau_first, amp_first, _ = _estimate_single_event_tau(0, tau_r, tau_d0)
                if anchor_final:
                    tau_last, amp_last, _ = _estimate_single_event_tau(n_pulses - 1, tau_r, tau_d0)

                tau_d_vec_raw = np.full(n_pulses, float(tau_d0))
                tau_d_vec0 = np.asarray(tau_d_vec_raw, float)
                if anchor_first:
                    tau_d_vec0[0] = float(tau_first)
                if anchor_final:
                    tau_d_vec0[-1] = float(tau_last)
                    tau_last_display = float(tau_last)
                    if amp_last is not None:
                        amp_last_display = float(amp_last)
            else:
                tau_d_vec0 = np.full(n_pulses, float(tau_d0))
                tau_d_vec_raw = np.asarray(tau_d_vec0, float)
            tau_d_vec_constrained = None
        else:
            tau_d_vec0 = np.full(n_pulses, float(tau_d0))
            tau_d_vec_raw = np.asarray(tau_d_vec0, float)
            tau_d_vec_constrained = None

        # When two-pass is enabled, skip smoothing here - two-pass block handles it
        if two_pass_will_run:
            # Pass raw tau to first NNLS, smoothing happens in two-pass block
            tau_d_vec = np.asarray(tau_d_vec0, float)
        else:
            tau_r, tau_d_vec = _apply_progression(tau_r, tau_d_vec0, tau_d0, 'global')

        # Validate and fix tau_d_vec: negative tau are forbidden
        MIN_TAU_D = 0.001  # Minimum 1ms decay time

        # Check for negative or zero values (silent fix)
        if np.any(tau_d_vec <= 0):
            tau_d_vec = np.maximum(tau_d_vec, MIN_TAU_D)

        # Check for NaN/Inf (silent fix)
        if not np.all(np.isfinite(tau_d_vec)):
            tau_d_vec = np.where(np.isfinite(tau_d_vec), tau_d_vec, tau_d0)

        if is_varying_model:
            progress_print(f"[fit] source=global | τd0={tau_d_vec[0]*1000:.2f}ms → τdN={tau_d_vec[-1]*1000:.2f}ms | mode={dec_mode}")
    elif fit_source == 'individual':
        # Individual fit_source: fit each trial independently
        tau_rs = []
        tau_d_mat = []
        for j in range(Yd.shape[1]):
            try:
                trj, td0j, slopej, tdvecj = estimate_kinetics_from_average(
                    t, Yd[:, j], stim_times,
                    taur_grid_ms=cfg['kin_taur_grid_ms'], taud0_grid_ms=cfg['kin_taud0_grid_ms'],
                    slope_grid_ms=cfg['kin_slope_grid_ms'], pre_zoom_s=cfg['pre_zoom_s'], post_zoom_s=cfg['post_zoom_s'],
                    isi=isi, weight_mode=weight_mode, weight_tau_s=weight_tau_s,
                    sg_window=cfg['sg_window'], sg_poly=cfg['sg_poly'],
                    peak_window_s=peak_window_s, peak_weight=peak_weight,
                    last_event_tail_tau_s=cfg.get('nnls_last_event_tail_tau_s'),
                    event_t0_s=event_t0_s,
                )
                tau_rs.append(trj)
                tau_d_mat.append(tdvecj)
            except Exception:
                continue
        tau_r = float(np.nanmedian(tau_rs)) if tau_rs else 0.002
        if tau_d_mat:
            tau_d_vec0 = np.nanmedian(np.vstack(tau_d_mat), axis=0)
        else:
            tau_d_vec0 = np.full(n_pulses, 0.010)
        tau_d0 = float(tau_d_vec0[0])
        tau_d_vec_raw = np.asarray(tau_d_vec0, float)
        per_event_param_map = {}
        # Skip smoothing when two-pass is enabled
        if two_pass_will_run:
            tau_d_vec = np.asarray(tau_d_vec0, float)
        else:
            tau_r, tau_d_vec = _apply_progression(tau_r, tau_d_vec0, tau_d0, 'individual')
        if is_varying_model:
            progress_print(f"[fit] source=individual | τd0={tau_d_vec[0]*1000:.2f}ms → τdN={tau_d_vec[-1]*1000:.2f}ms | mode={dec_mode}")

    else:  # 'average'
        # Average fit_source: fit each event on the average trace
        if dec_mode == 'fixed':
            # For fixed mode, use grid search for single tau
            tau_r, tau_d0, slope, tau_d_vec0 = estimate_kinetics_from_average(
                t, y_avg, stim_times,
                taur_grid_ms=cfg['kin_taur_grid_ms'], taud0_grid_ms=cfg['kin_taud0_grid_ms'],
                slope_grid_ms=cfg['kin_slope_grid_ms'], pre_zoom_s=cfg['pre_zoom_s'], post_zoom_s=cfg['post_zoom_s'],
                isi=isi, weight_mode=weight_mode, weight_tau_s=weight_tau_s,
                sg_window=cfg['sg_window'], sg_poly=cfg['sg_poly'],
                peak_window_s=peak_window_s, peak_weight=peak_weight,
                last_event_tail_tau_s=cfg.get('nnls_last_event_tail_tau_s'),
                event_t0_s=event_t0_s,
            )
            tau_d_vec_raw = np.asarray(tau_d_vec0, float)
        elif dec_mode in ('linear', 'free_monotonic'):
            # Fit tau_r globally first
            tau_r, tau_d0, slope, _ = estimate_kinetics_from_average(
                t, y_avg, stim_times,
                taur_grid_ms=cfg['kin_taur_grid_ms'], taud0_grid_ms=cfg['kin_taud0_grid_ms'],
                slope_grid_ms=cfg['kin_slope_grid_ms'], pre_zoom_s=cfg['pre_zoom_s'], post_zoom_s=cfg['post_zoom_s'],
                isi=isi, weight_mode=weight_mode, weight_tau_s=weight_tau_s,
                sg_window=cfg['sg_window'], sg_poly=cfg['sg_poly'],
                peak_window_s=peak_window_s, peak_weight=peak_weight,
                last_event_tail_tau_s=cfg.get('nnls_last_event_tail_tau_s'),
                event_t0_s=event_t0_s,
            )
            # Now fit each individual event on the average trace
            tau_per_evt, amp_per_evt, param_map = _fit_all_events_on_average(tau_r, tau_d0)
            per_event_param_map = {k: np.asarray(v, float) for k, v in param_map.items()}
            tau_d_vec0 = tau_per_evt
            tau_d_vec_raw = np.asarray(tau_per_evt, float)
            tau_last_display = float(tau_per_evt[-1])
            amp_last_display = float(amp_per_evt[-1])
        else:
            tau_r, tau_d0, slope, tau_d_vec0 = estimate_kinetics_from_average(
                t, y_avg, stim_times,
                taur_grid_ms=cfg['kin_taur_grid_ms'], taud0_grid_ms=cfg['kin_taud0_grid_ms'],
                slope_grid_ms=cfg['kin_slope_grid_ms'], pre_zoom_s=cfg['pre_zoom_s'], post_zoom_s=cfg['post_zoom_s'],
                isi=isi, weight_mode=weight_mode, weight_tau_s=weight_tau_s,
                sg_window=cfg['sg_window'], sg_poly=cfg['sg_poly'],
                peak_window_s=peak_window_s, peak_weight=peak_weight,
                last_event_tail_tau_s=cfg.get('nnls_last_event_tail_tau_s'),
                event_t0_s=event_t0_s,
            )
            tau_d_vec_raw = np.asarray(tau_d_vec0, float)

        # Skip smoothing when two-pass is enabled
        if two_pass_will_run:
            tau_d_vec = np.asarray(tau_d_vec0, float)
        else:
            tau_r, tau_d_vec = _apply_progression(tau_r, tau_d_vec0, tau_d0, 'average')
        if is_varying_model:
            progress_print(f"[fit] source=average | τd0={tau_d_vec[0]*1000:.2f}ms → τdN={tau_d_vec[-1]*1000:.2f}ms | mode={dec_mode}")

        # Extract per-trial event snippets for visualization in average/individual mode
        if fit_source in ('average', 'individual') and Yd.shape[1] > 0:
            try:
                # Extract first event from each trial for overlay visualization
                snippets_list = []
                pre_s = cfg.get('pre_zoom_s', 0.15)
                post_s = min(isi, cfg.get('post_zoom_s', 0.60)) if isi > 0 else cfg.get('post_zoom_s', 0.60)

                # Use first event time
                if len(stim_times) > 0:
                    evt_time = stim_times[0] + event_t0_s
                    t_start = evt_time - pre_s
                    t_end = evt_time + post_s

                    # Create time masks once
                    trial_mask = (t >= t_start) & (t <= t_end)
                    baseline_mask = (t >= t_start) & (t < evt_time)

                    if np.any(trial_mask):
                        # Extract snippet from each trial with baseline normalization
                        for trial_idx in range(Yd.shape[1]):
                            y_trial = Yd[trial_mask, trial_idx]
                            if np.isfinite(y_trial).any():
                                # Calculate baseline from pre-stimulus period
                                y_baseline_region = Yd[baseline_mask, trial_idx]
                                if np.isfinite(y_baseline_region).any():
                                    baseline = float(np.nanmean(y_baseline_region))
                                else:
                                    baseline = float(np.nanmean(y_trial[:max(1, len(y_trial)//10)]))

                                # Baseline-normalize by subtracting baseline
                                y_normalized = y_trial - baseline
                                snippets_list.append(y_normalized)

                        if snippets_list:
                            # Create relative time vector
                            recut_t_rel = t[trial_mask] - evt_time
                            recut_snippets = snippets_list
                            # Calculate average after baseline normalization
                            recut_avg = np.nanmean(np.column_stack(snippets_list), axis=1) if len(snippets_list) > 1 else snippets_list[0]
            except Exception:
                pass

    # Build interpolated-setting summaries for diagnostics and downstream use
    anchor_first_cfg = bool(cfg.get('anchor_first_tau', False))
    anchor_final_cfg = bool(cfg.get('anchor_final_tau', True))
    anchor_applicable = dec_mode in ('linear', 'free_monotonic')
    mid_index = int(min(max(0, n_pulses // 2), max(0, n_pulses - 1))) if n_pulses > 0 else 0
    anchor_template = {
        'first': anchor_applicable and anchor_first_cfg and n_pulses > 0,
        'final': anchor_applicable and anchor_final_cfg and n_pulses > 0,
        'mid': anchor_applicable and (fit_source == 'global') and n_pulses > 0,
        'mid_index': mid_index,
    }
    progression_label_map = {
        'fixed': 'fixed (median)',
        'linear': 'linear regression',
        'free_monotonic': 'monotonic spline',
    }
    progression_label = progression_label_map.get(dec_mode, dec_mode)
    if anchor_applicable:
        if anchor_first_cfg and anchor_final_cfg:
            progression_label += ' (anchored first & final)'
        elif anchor_final_cfg:
            progression_label += ' (anchored to final)'
        elif anchor_first_cfg:
            progression_label += ' (anchored to first)'

    interpolated_settings = []
    if n_pulses > 0 and np.size(tau_d_vec):
        def _format_display(name: str, override: Optional[str] = None) -> str:
            if override is not None:
                return override
            if not name:
                return name
            if name.startswith('tau'):
                return 'τ' + name[3:]
            if name.startswith('t_'):
                return 't' + name[2:]
            return name

        def _register_setting(
            name: str,
            final_values,
            *,
            raw=None,
            constrained=None,
            label: Optional[str] = None,
            unit: str = 'ms',
            scale: float = 1000.0,
            note: Optional[str] = None,
            direction_note: Optional[str] = None,
            derivation: Optional[str] = None,
        ) -> None:
            arr_final = np.asarray(final_values, float)
            if arr_final.ndim == 0:
                arr_final = np.full(n_pulses, float(arr_final))
            if arr_final.size != n_pulses or not np.any(np.isfinite(arr_final)):
                return

            arr_raw = None
            if raw is not None:
                arr_raw = np.asarray(raw, float)
                if arr_raw.ndim == 0:
                    arr_raw = np.full(n_pulses, float(arr_raw))
                if arr_raw.size != n_pulses or not np.any(np.isfinite(arr_raw)):
                    arr_raw = None

            arr_constrained = None
            if constrained is not None:
                arr_constrained = np.asarray(constrained, float)
                if arr_constrained.ndim == 0:
                    arr_constrained = np.full(n_pulses, float(arr_constrained))
                if arr_constrained.size != n_pulses or not np.any(np.isfinite(arr_constrained)):
                    arr_constrained = None

            entry: Dict[str, Any] = {
                'name': name,
                'display': label or _format_display(name),
                'unit': unit,
                'scale': float(scale),
                'raw': arr_raw,
                'constrained': arr_constrained,
                'final': arr_final,
                'note': note,
                'direction_note': direction_note,
                'fit_source': fit_source,
                'decay_progression_mode': dec_mode,
                'progression_label': progression_label,
                'derivation': derivation or 'direct',
            }

            anchors: List[Dict[str, Any]] = []
            if anchor_template['first']:
                anchors.append({'kind': 'first', 'index': 0, 'value': float(arr_final[0])})
            if anchor_template['mid'] and 0 <= anchor_template['mid_index'] < n_pulses:
                idx = int(anchor_template['mid_index'])
                anchors.append({'kind': 'mid', 'index': idx, 'value': float(arr_final[idx])})
            if anchor_template['final']:
                anchors.append({'kind': 'final', 'index': n_pulses - 1, 'value': float(arr_final[-1])})
            entry['anchors'] = anchors
            interpolated_settings.append(entry)

        def _progress_metric_series(
            raw_series,
            base_value,
            *,
            param_name: str = 'unknown',
            min_value: float = 1e-4,
            clip: Optional[Tuple[Optional[float], Optional[float]]] = None,
        ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
            if raw_series is None:
                return None, None, None
            arr_raw = np.asarray(raw_series, float)
            if arr_raw.size != n_pulses or not np.any(np.isfinite(arr_raw)):
                return None, None, None

            finite_vals = arr_raw[np.isfinite(arr_raw)]
            anchor_val = None
            if base_value is not None and np.isfinite(base_value):
                anchor_val = float(base_value)

            fallback_candidates = []
            if anchor_val is not None:
                fallback_candidates.append(anchor_val)
            if finite_vals.size:
                fallback_candidates.append(float(np.nanmedian(finite_vals)))
                fallback_candidates.append(float(finite_vals[0]))

            fallback_val = None
            for candidate in fallback_candidates:
                if np.isfinite(candidate):
                    fallback_val = float(candidate)
                    break
            if fallback_val is None:
                fallback_val = 0.010

            if min_value is not None:
                fallback_val = max(float(min_value), float(fallback_val))

            arr_clean = arr_raw.copy()
            arr_clean[~np.isfinite(arr_clean)] = fallback_val
            if min_value is not None:
                arr_clean = np.maximum(arr_clean, float(min_value))

            arr_constrained = arr_clean.copy()
            if fit_source == 'global' and anchor_template['mid'] and anchor_val is not None:
                idx = anchor_template['mid_index']
                if 0 <= idx < n_pulses:
                    arr_constrained[idx] = anchor_val
            if min_value is not None:
                arr_constrained = np.maximum(arr_constrained, float(min_value))

            _, progressed = _apply_progression(tau_r, arr_constrained, fallback_val, fit_source, param_name)
            progressed = np.asarray(progressed, float)

            if clip is not None:
                lo, hi = clip
                if lo is not None:
                    progressed = np.maximum(progressed, float(lo))
                    arr_constrained = np.maximum(arr_constrained, float(lo))
                    arr_clean = np.maximum(arr_clean, float(lo))
                if hi is not None:
                    progressed = np.minimum(progressed, float(hi))
                    arr_constrained = np.minimum(arr_constrained, float(hi))
                    arr_clean = np.minimum(arr_clean, float(hi))

            return progressed, arr_clean, arr_constrained

        if np.isfinite(tau_r):
            global_fit_params.setdefault('tau_rise', float(tau_r))
        if np.isfinite(tau_d0):
            if 'tau_decay_fast' not in global_fit_params and 'tau_decay' not in global_fit_params:
                global_fit_params.setdefault('tau_decay', float(tau_d0))

        # Handle fast decay component
        # For bi-exponential models (e.g., iGluSnfr), tau_decay_fast should come from per_event_param_map
        # For single-exponential models, use tau_d_vec from the main progression
        fast_key = 'tau_decay_fast' if 'tau_decay_fast' in global_fit_params else 'tau_decay'
        fast_label = _format_display(fast_key)
        fast_base = global_fit_params.get('tau_decay_fast')
        if fast_base is None and 'tau_decay' in global_fit_params:
            fast_base = global_fit_params.get('tau_decay')

        # Check if model has separate fast/slow components (bi-exponential)
        has_biexp = 'tau_decay_fast' in global_fit_params and dec_mode in ('linear', 'free_monotonic')

        if has_biexp and 'tau_decay_fast' in per_event_param_map:
            # Use per-event fitted tau_decay_fast for bi-exponential models
            fast_seq, fast_raw, fast_constrained = _progress_metric_series(
                per_event_param_map.get('tau_decay_fast'),
                fast_base,
                param_name='tau_decay_fast',
            )
            if fast_seq is not None:
                _register_setting(
                    'tau_decay_fast',
                    fast_seq,
                    raw=fast_raw,
                    constrained=fast_constrained,
                    label=_format_display('tau_decay_fast'),
                    unit='ms',
                    scale=1000.0,
                    note='fast component (directly fitted progression)',
                    derivation='fitted_progression',
                )
            fast_global = fast_base
        else:
            # Use tau_d_vec for single-exponential models or fallback
            _register_setting(
                fast_key,
                tau_d_vec,
                raw=tau_d_vec_raw,
                constrained=tau_d_vec_constrained,
                label=fast_label,
                unit='ms',
                scale=1000.0,
                note='fast component (directly fitted progression)',
                derivation='fitted_progression',
            )
            fast_global = None
            if fast_key in global_fit_params and np.isfinite(global_fit_params[fast_key]):
                fast_global = float(global_fit_params[fast_key])
            elif 'tau_decay' in global_fit_params and np.isfinite(global_fit_params['tau_decay']):
                fast_global = float(global_fit_params['tau_decay'])
            elif np.isfinite(tau_d0):
                fast_global = float(tau_d0)

        slow_seq = None
        slow_seq_raw = None
        slow_seq_constrained = None
        slow_from_fit = False
        slow_base = global_fit_params.get('tau_decay_slow')
        if dec_mode in ('linear', 'free_monotonic'):
            slow_seq, slow_seq_raw, slow_seq_constrained = _progress_metric_series(
                per_event_param_map.get('tau_decay_slow'),
                slow_base,
                param_name='tau_decay_slow',
            )
            if slow_seq is not None:
                slow_from_fit = True
                _register_setting(
                    'tau_decay_slow',
                    slow_seq,
                    raw=slow_seq_raw,
                    constrained=slow_seq_constrained,
                    label=_format_display('tau_decay_slow'),
                    unit='ms',
                    scale=1000.0,
                    note='slower component (directly fitted progression)',
                    derivation='fitted_progression',
                )

        if (
            slow_seq is None
            and slow_base is not None
            and np.isfinite(slow_base)
            and fast_global is not None
            and fast_global > 0
        ):
            ratio_slow = float(slow_base) / float(fast_global)
            if np.isfinite(ratio_slow) and ratio_slow > 0:
                slow_seq = tau_d_vec * ratio_slow
                slow_seq_raw = tau_d_vec_raw * ratio_slow if tau_d_vec_raw is not None else None
                slow_seq_constrained = (
                    tau_d_vec_constrained * ratio_slow if tau_d_vec_constrained is not None else None
                )
                _register_setting(
                    'tau_decay_slow',
                    slow_seq,
                    raw=slow_seq_raw,
                    constrained=slow_seq_constrained,
                    label=_format_display('tau_decay_slow'),
                    unit='ms',
                    scale=1000.0,
                    note='slower component (scaled by global τ_slow/τ_fast)',
                    derivation='scaled_from_fast_progression',
                )

        if slow_seq is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio_final = np.divide(
                    slow_seq,
                    tau_d_vec,
                    out=np.full_like(slow_seq, np.nan),
                    where=np.isfinite(tau_d_vec) & (np.abs(tau_d_vec) > 0),
                )
            ratio_raw = None
            if slow_seq_raw is not None and tau_d_vec_raw is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio_raw = np.divide(
                        slow_seq_raw,
                        tau_d_vec_raw,
                        out=np.full_like(slow_seq_raw, np.nan),
                        where=np.isfinite(tau_d_vec_raw) & (np.abs(tau_d_vec_raw) > 0),
                    )
            ratio_constrained = None
            if slow_seq_constrained is not None and tau_d_vec_constrained is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio_constrained = np.divide(
                        slow_seq_constrained,
                        tau_d_vec_constrained,
                        out=np.full_like(slow_seq_constrained, np.nan),
                        where=np.isfinite(tau_d_vec_constrained) & (np.abs(tau_d_vec_constrained) > 0),
                    )
            if np.any(np.isfinite(ratio_final)):
                _register_setting(
                    'tau_ratio_slow_fast',
                    ratio_final,
                    raw=ratio_raw,
                    constrained=ratio_constrained,
                    label=_format_display('', 'τ_slow/τ_fast'),
                    unit='',
                    scale=1.0,
                    note='slow/fast decay ratio (from fitted slow & fast progressions)' if slow_from_fit else 'slow/fast decay ratio (global constant)',
                    derivation='derived_from_fitted_components' if slow_from_fit else 'global_ratio_constant',
                )

        rise_base = global_fit_params.get('tau_rise', float(tau_r) if np.isfinite(tau_r) else np.nan)
        rise_seq = None
        rise_raw = None
        rise_constrained = None
        rise_from_fit = False
        if dec_mode in ('linear', 'free_monotonic'):
            rise_seq, rise_raw, rise_constrained = _progress_metric_series(
                per_event_param_map.get('tau_rise'),
                rise_base,
                param_name='tau_rise',
            )
            if rise_seq is not None:
                rise_from_fit = True
                _register_setting(
                    'tau_rise',
                    rise_seq,
                    raw=rise_raw,
                    constrained=rise_constrained,
                    label=_format_display('tau_rise'),
                    unit='ms',
                    scale=1000.0,
                    direction_note='lower = faster rise',
                    note='rise component (directly fitted progression)',
                    derivation='fitted_progression',
                )

        if (
            rise_seq is None
            and rise_base is not None
            and np.isfinite(rise_base)
            and fast_global is not None
            and fast_global > 0
        ):
            ratio_rise = float(rise_base) / float(fast_global)
            if np.isfinite(ratio_rise) and ratio_rise > 0:
                rise_seq = tau_d_vec * ratio_rise
                rise_raw = tau_d_vec_raw * ratio_rise if tau_d_vec_raw is not None else None
                rise_constrained = (
                    tau_d_vec_constrained * ratio_rise if tau_d_vec_constrained is not None else None
                )
                _register_setting(
                    'tau_rise',
                    rise_seq,
                    raw=rise_raw,
                    constrained=rise_constrained,
                    label=_format_display('tau_rise'),
                    unit='ms',
                    scale=1000.0,
                    direction_note='lower = faster rise',
                    note='derived from fast decay using global τ_rise/τ_fast ratio',
                    derivation='scaled_from_fast_progression',
                )

        frac_fast = global_fit_params.get('frac_fast')
        frac_seq = None
        frac_raw = None
        frac_constrained = None
        frac_from_fit = False
        if dec_mode in ('linear', 'free_monotonic'):
            frac_seq, frac_raw, frac_constrained = _progress_metric_series(
                per_event_param_map.get('frac_fast'),
                frac_fast,
                param_name='frac_fast',
                min_value=1e-6,
                clip=(0.0, 1.0),
            )
            if frac_seq is not None:
                frac_from_fit = True
                _register_setting(
                    'frac_fast',
                    frac_seq,
                    raw=frac_raw,
                    constrained=frac_constrained,
                    label=_format_display('frac_fast'),
                    unit='',
                    scale=1.0,
                    note='fast component weight (directly fitted progression)',
                    derivation='fitted_progression',
                )

        if not frac_from_fit and frac_fast is not None and np.isfinite(frac_fast):
            frac_arr = np.full(n_pulses, float(frac_fast), float)
            _register_setting(
                'frac_fast',
                frac_arr,
                raw=frac_arr.copy(),
                constrained=None,
                label=_format_display('frac_fast'),
                unit='',
                scale=1.0,
                note='fast component weight (global constant)',
                derivation='global_constant',
            )

    # Update exponential weight tau if using global fit_source and auto tau
    if (weight_mode == 'exponential' and cfg.get('nnls_weight_tau_s', None) is None
        and cfg.get('fit_source', 'global') == 'global'):
        weight_tau_s = float(tau_d_vec[0])  # Use estimated tau_d

    # Validate tau_d_vec and apply fallbacks if needed
    tau_d_vec = np.asarray(tau_d_vec, float)
    if not np.all(np.isfinite(tau_d_vec)):
        bad_indices = np.where(~np.isfinite(tau_d_vec))[0]
        # Replace bad values with tau_d0 or nearest valid value (silent fix)
        if np.isfinite(tau_d0):
            tau_d_vec[~np.isfinite(tau_d_vec)] = tau_d0
        else:
            # Last resort: use 10ms default
            tau_d_vec[~np.isfinite(tau_d_vec)] = 0.010
        # Ensure monotonic after fixing
        tau_d_vec = np.maximum.accumulate(tau_d_vec)

    # Validate tau_r (silent fix)
    if not np.isfinite(tau_r):
        tau_r = 0.002

    # Constraint vector and debug logging disabled for cleaner output

    # Fit average trace (forward, no overlap) and measure amplitudes
    # Use template variants if enabled, otherwise use standard forward fitting
    variant_info_avg = None
    # Debug: check why variants might not be used
    _use_var = cfg.get('use_template_variants', False)
    _builder_set = _VARIANT_KERNEL_BUILDER is not None
    # NNLS debug logging intentionally suppressed to reduce noise.
    # Check if we should use the variant-based fitting approach
    use_variants_approach = cfg.get('use_template_variants', False) and _VARIANT_KERNEL_BUILDER is not None
    jitter_variant_ms = cfg.get('jitter_variant_ms', None)

    # Constrain jitter so t_onset stays within bounds when specified
    t_onset_bounds = cfg.get('parameter_bounds', {}).get('t_onset')
    if jitter_variant_ms is not None and t_onset_bounds is not None:
        try:
            if isinstance(t_onset_bounds, (tuple, list)) and len(t_onset_bounds) == 2:
                t_onset_min, t_onset_max = float(t_onset_bounds[0]), float(t_onset_bounds[1])
                if np.isfinite(t_onset_min) or np.isfinite(t_onset_max):
                    event_t0_ms = float(event_t0_s) * 1000.0
                    jitter_arr = np.asarray(jitter_variant_ms, float)
                    allowed = []
                    for j in jitter_arr:
                        t_onset_ms = event_t0_ms + float(j)
                        if np.isfinite(t_onset_min) and t_onset_ms < t_onset_min - 1e-9:
                            continue
                        if np.isfinite(t_onset_max) and t_onset_ms > t_onset_max + 1e-9:
                            continue
                        allowed.append(float(j))
                    if not allowed:
                        if np.isfinite(event_t0_ms):
                            if ((not np.isfinite(t_onset_min) or event_t0_ms >= t_onset_min - 1e-9)
                                    and (not np.isfinite(t_onset_max) or event_t0_ms <= t_onset_max + 1e-9)):
                                allowed = [0.0]
                            else:
                                clamped = event_t0_ms
                                if np.isfinite(t_onset_min) and clamped < t_onset_min:
                                    clamped = t_onset_min
                                if np.isfinite(t_onset_max) and clamped > t_onset_max:
                                    clamped = t_onset_max
                                allowed = [float(clamped - event_t0_ms)]
                        else:
                            allowed = [0.0]
                    jitter_variant_ms = np.array(sorted(set(allowed)), float)
                    progress_print(
                        "[NNLS] Jitter constrained by t_onset bounds "
                        f"({t_onset_min:.2f}-{t_onset_max:.2f} ms): {len(jitter_variant_ms)} shifts"
                    )
        except Exception:
            pass

    # Also use variants approach if jitter_variant_ms is specified (even without template variants)
    if jitter_variant_ms is not None and not use_variants_approach:
        use_variants_approach = True

    if use_variants_approach:
        # Use template variants approach: single NNLS with multiple templates per event
        expanded_variants = False
        if cfg.get('use_template_variants', False) and _VARIANT_KERNEL_BUILDER is not None:
            if event_model == 'iglusnfr_tri':
                variant_ratios = _build_triexp_variant_grid()
            else:
                variant_ratios = cfg.get('template_variant_ratios', [0.2, 0.4, 0.6, 0.8])
                # Optional: expand slow-fraction variants across a slow-tau grid (bi-exp only)
                # This stabilizes NNLS when tau_decay_slow bounds are wide.
                tau_slow_grid_ms = cfg.get('template_variant_tau_slow_ms', None)
                ratios_list = list(variant_ratios) if variant_ratios is not None else []
                has_tau_override = any(
                    isinstance(v, (tuple, list)) and len(v) >= 2 for v in ratios_list
                )
                tau_grid_s = None
                if (not has_tau_override) and tau_slow_grid_ms is None:
                    param_bounds = cfg.get('parameter_bounds', {}) or {}
                    bound = param_bounds.get('tau_decay_slow')
                    if bound and isinstance(bound, (tuple, list)) and len(bound) == 2:
                        lower, upper = float(bound[0]), float(bound[1])
                        if np.isfinite(lower) and np.isfinite(upper) and (upper - lower) > 0.005:
                            mid = 0.5 * (lower + upper)
                            candidates = [lower, mid, upper, lower * 1.25, lower * 1.5, lower * 2.0]
                            try:
                                tau_fast = float(tau_d_vec[0]) if np.size(tau_d_vec) else np.nan
                            except Exception:
                                tau_fast = np.nan
                            if np.isfinite(tau_fast) and tau_fast > 0:
                                for r in (2.0, 2.5, 3.0, 4.0):
                                    candidates.append(tau_fast * r)
                            tau_grid_s = []
                            for c in candidates:
                                if not np.isfinite(c):
                                    continue
                                c = min(max(c, lower), upper)
                                if c > 0:
                                    tau_grid_s.append(c)
                            tau_grid_s = sorted(set(tau_grid_s))
                if (not has_tau_override) and tau_grid_s is None and tau_slow_grid_ms is not None:
                    tau_grid_s = []
                    for v in tau_slow_grid_ms:
                        try:
                            val_s = float(v) / 1000.0
                        except Exception:
                            continue
                        if np.isfinite(val_s) and val_s > 0:
                            tau_grid_s.append(val_s)
                    tau_grid_s = sorted(set(tau_grid_s))
                expanded_variants = False
                if (not has_tau_override) and tau_grid_s:
                    base_ratios = ratios_list
                    expanded = []
                    for r in base_ratios:
                        try:
                            r_val = float(r)
                        except Exception:
                            continue
                        for tau_slow_s in tau_grid_s:
                            expanded.append((r_val, tau_slow_s))
                    if expanded:
                        variant_ratios = expanded
                        expanded_variants = True
                        # tau_slow variants logging disabled for cleaner output
                        pass
        else:
            # Jitter-only mode: determine optimal template ratio first if needed
            # For iGluSnFR, quickly test a few ratios to find the best one for this data
            if _VARIANT_KERNEL_BUILDER is not None and jitter_variant_ms is not None:
                # Quick ratio search on average trace (coarse grid)
                test_ratios = [0.0, 0.3, 0.5, 0.7, 1.0]
                best_ratio = 0.5
                best_error = float('inf')
                weights_test = _calculate_nnls_weights(
                    t,
                    stim_times,
                    isi,
                    weight_mode,
                    weight_tau_s,
                    y_ref=y_avg,
                    peak_window_s=peak_window_s,
                    peak_weight=peak_weight,
                )
                # Apply last-event tail downweighting
                last_event_tail_tau_s = cfg.get('nnls_last_event_tail_tau_s')
                pw_s = peak_window_s if peak_window_s is not None else 0.010
                weights_test = _apply_last_event_tail_weights(
                    weights_test, t, stim_times, isi, last_event_tail_tau_s, pw_s
                )
                for ratio in test_ratios:
                    kernel_cols = []
                    for i_event, st in enumerate(stim_times):
                        anchor = st + event_t0_s
                        k = _build_variant_kernel(
                            t - anchor, tau_r, tau_d_vec[i_event], ratio,
                            event_idx=i_event, n_events=len(stim_times),
                        )
                        kernel_cols.append(k)
                    X_test = np.column_stack(kernel_cols) if kernel_cols else np.zeros((t.size, 0))
                    a_test = _nnls_weighted(X_test, y_avg, weights_test)
                    yhat_test = X_test @ a_test
                    error = np.sqrt(np.nanmean((y_avg - yhat_test)**2))
                    if error < best_error:
                        best_error = error
                        best_ratio = ratio
                variant_ratios = [best_ratio]
                # Jitter-only mode logging disabled for cleaner output
            else:
                # Non-variant model: ratio doesn't matter
                variant_ratios = [0.5]

        # NNLS configuration logging disabled for cleaner output

        # Choose fitting mode: sequential greedy pass or standard simultaneous NNLS
        fit_mode = str(cfg.get('nnls_fit_mode', 'simultaneous')).strip().lower()
        use_sequential_fit = fit_mode == 'sequential'

        if use_sequential_fit:
            progress_print("[NNLS] Using sequential greedy fitting mode")
            a_avg, d_avg, X_avg, yhat_avg, comp_avg, variant_info_avg = fit_amplitudes_sequential(
                y_avg, t, stim_times, tau_r, tau_d_vec,
                variant_ratios=variant_ratios,
                isi=isi,
                event_t0_s=event_t0_s,
                jitter_variant_ms=jitter_variant_ms,
                peak_window_s=peak_window_s,
                peak_weight=peak_weight,
                weight_mode=weight_mode,
                weight_tau_s=weight_tau_s,
                last_event_post_s=float(cfg.get('post_zoom_s', 0.200)),
                last_event_tail_tau_s=cfg.get('nnls_last_event_tail_tau_s'),
            )
        else:
            # Choose variant selection strategy (soft mix vs hard per-event selection)
            variant_select = str(cfg.get('template_variant_select', 'soft')).strip().lower()
            hard_select = variant_select in ('hard', 'dominant', 'winner')

            a_avg, d_avg, X_avg, yhat_avg, comp_avg, variant_info_avg = fit_amplitudes_with_template_variants(
                y_avg, t, stim_times, tau_r, tau_d_vec,
                variant_ratios=variant_ratios,
                weight_mode=weight_mode,
                weight_tau_s=weight_tau_s,
                isi=isi,
                event_t0_s=event_t0_s,
                jitter_variant_ms=jitter_variant_ms,
                peak_window_s=peak_window_s,
                peak_weight=peak_weight,
                last_event_tail_tau_s=cfg.get('nnls_last_event_tail_tau_s'),
                hard_select=hard_select,
            )
        # Log selected variant summary for tri-exp (shows if bi-exp or tri-exp was preferred)
        if variant_info_avg and event_model == 'iglusnfr_tri':
            dom_ratios = variant_info_avg.get('dominant_template_ratio', [])
            if dom_ratios is not None and len(dom_ratios) > 0:
                # For tri-exp, ratios are tuples (slow_frac, superslow_frac) or (slow_frac, superslow_frac, tau_slow)
                is_tuple_format = all(isinstance(r, (tuple, list)) and len(r) >= 2 for r in dom_ratios)
                if is_tuple_format:
                    ss_fracs = [float(r[1]) for r in dom_ratios]
                    n_biexp = sum(1 for f in ss_fracs if f < 0.01)  # superslow < 1% = effectively bi-exp
                    n_triexp = len(ss_fracs) - n_biexp
                    ss_str = ", ".join(f"{f:.2f}" for f in ss_fracs)
                    progress_print(f"[tri-exp] Superslow fracs per event: [{ss_str}] ({n_biexp} bi-exp, {n_triexp} tri-exp)")
                else:
                    # Scalar format - just slow fraction, no superslow
                    slow_fracs = [float(r) for r in dom_ratios]
                    slow_str = ", ".join(f"{f:.2f}" for f in slow_fracs)
                    progress_print(f"[tri-exp] Slow fracs per event: [{slow_str}] (scalar format, no superslow info)")
    else:
        # Standard forward fitting with single template per event
        a_avg, d_avg, X_avg, yhat_avg, comp_avg = fit_amplitudes_no_overlap_forward(
            y_avg, t, stim_times, tau_r, tau_d_vec,
            pre_zoom_s=cfg['pre_zoom_s'], post_zoom_s=cfg['post_zoom_s'],
            robust=True, huber_delta=cfg['huber_delta'], irls_iters=cfg['irls_iters'],
            allow_shift=allow_shift, delta_max_s=cfg['delta_max_s'], delta_step_s=cfg['delta_step_s'],
            shift_min_s=cfg['shift_min_s'], event_t0_s=event_t0_s,
        )

    # ============================================================================
    # TWO-PASS NNLS: Smooth tau evolution and refit with fixed kinetics
    # ============================================================================
    # If decay_progression_mode is not 'none' and template variants were used,
    # do a second pass with smoothed RATIO values (only amplitude + jitter vary)
    # The goal: smooth the slow/fast RATIO evolution across the train, not tau_d itself
    _is_sequential = (
        variant_info_avg is not None
        and variant_info_avg.get('sequential', False)
    )
    two_pass_enabled = (
        dec_mode not in ('none', 'fixed')
        and (cfg.get('use_template_variants', False) or _is_sequential)
        and variant_info_avg is not None
    )
    
    if two_pass_enabled:
        try:
            progress_print(f"[TWO-PASS] Starting second pass with smoothed slow-fraction (mode={dec_mode})")
            
            # Extract dominant ratios from pass 1 - these are the slow fractions per event
            dominant_ratios_pass1 = variant_info_avg.get('dominant_template_ratio', None)
            dominant_jitters_pass1 = variant_info_avg.get('dominant_jitter_ms', d_avg * 1000.0)
            
            if dominant_ratios_pass1 is None:
                raise ValueError("No dominant_template_ratio from pass 1")
            
            # Determine format of dominant_ratios_pass1:
            # - Scalar array: pure bi-exp [0.2, 0.4, 0.6, ...]
            # - Tuples (frac, tau_slow): bi-exp with tau_slow grid [(0.2, 0.012), (0.4, 0.012), ...]
            # - Tuples (frac_slow, frac_superslow, tau_superslow): tri-exp
            sample_ratio = dominant_ratios_pass1[0] if len(dominant_ratios_pass1) > 0 else None
            is_scalar_ratios = np.isscalar(sample_ratio)
            is_biexp_tuple = (
                isinstance(sample_ratio, (tuple, list)) and 
                len(sample_ratio) == 2
            )
            is_triexp_tuple = (
                isinstance(sample_ratio, (tuple, list)) and 
                len(sample_ratio) >= 3
            )
            
            if is_scalar_ratios:
                # Pure bi-exp model: smooth the slow fraction directly
                raw_fracs = np.asarray(dominant_ratios_pass1, float)
            elif is_biexp_tuple:
                # Bi-exp with tau_slow grid: extract slow fraction (first element of tuple)
                raw_fracs = np.array([float(r[0]) for r in dominant_ratios_pass1])
                raw_tau_slow = np.array([float(r[1]) for r in dominant_ratios_pass1])
            elif is_triexp_tuple:
                # Tri-exp model: ratios are tuples like (frac_slow, frac_superslow) or (frac_slow, frac_superslow, tau_slow)
                # Extract both slow and superslow fractions for smoothing
                raw_slow_fracs = np.array([float(r[0]) for r in dominant_ratios_pass1])
                raw_superslow_fracs = np.array([float(r[1]) for r in dominant_ratios_pass1])
                raw_tau_slow_tri = None
                if len(dominant_ratios_pass1[0]) >= 3:
                    raw_tau_slow_tri = np.array([float(r[2]) for r in dominant_ratios_pass1])
                raw_fracs = raw_slow_fracs  # Will be smoothed below
            else:
                raise ValueError(f"Unknown dominant_ratio format: {type(sample_ratio)}")
            
            if raw_fracs is not None:
                # Smooth ratios using _apply_progression
                # Use progression rule 'monotonic_increasing' for slow fraction
                _, smoothed_fracs = _apply_progression(
                    tau_r, raw_fracs, float(np.nanmedian(raw_fracs)), 
                    'global', 'slow_fraction'
                )
                
                # Clip to valid range [0, 1]
                smoothed_fracs = np.clip(smoothed_fracs, 0.0, 1.0)
                
                # Build pass2_variant_ratios in the same format as input
                if is_scalar_ratios:
                    # Pure bi-exp: just scalars
                    pass2_variant_ratios = list(smoothed_fracs)
                    smoothed_ratios = smoothed_fracs
                elif is_triexp_tuple:
                    # Tri-exp: smooth superslow fractions too
                    _, smoothed_superslow = _apply_progression(
                        tau_r, raw_superslow_fracs, float(np.nanmedian(raw_superslow_fracs)),
                        'global', 'slow_fraction'  # Use same progression rule
                    )
                    smoothed_superslow = np.clip(smoothed_superslow, 0.0, 1.0)
                    
                    # Ensure slow + superslow <= 1.0
                    for i in range(len(smoothed_fracs)):
                        total = smoothed_fracs[i] + smoothed_superslow[i]
                        if total > 1.0:
                            # Scale down proportionally
                            scale = 1.0 / total
                            smoothed_fracs[i] *= scale
                            smoothed_superslow[i] *= scale
                    
                    # Build tuples: (frac_slow, frac_superslow) or (frac_slow, frac_superslow, tau_slow)
                    if raw_tau_slow_tri is not None:
                        pass2_variant_ratios = [
                            (float(smoothed_fracs[i]), float(smoothed_superslow[i]), float(raw_tau_slow_tri[i]))
                            for i in range(len(smoothed_fracs))
                        ]
                    else:
                        pass2_variant_ratios = [
                            (float(smoothed_fracs[i]), float(smoothed_superslow[i]))
                            for i in range(len(smoothed_fracs))
                        ]
                    smoothed_ratios = smoothed_fracs  # Just slow fraction for reporting
                    
                    # Log the smoothed superslow fractions
                    ss_raw_str = ", ".join(f"{f:.2f}" for f in raw_superslow_fracs)
                    ss_smooth_str = ", ".join(f"{f:.2f}" for f in smoothed_superslow)
                    progress_print(f"[TWO-PASS tri-exp] Superslow: [{ss_raw_str}] -> [{ss_smooth_str}]")
                else:
                    # Bi-exp with tau_slow: keep the tau_slow from pass 1, use smoothed fraction
                    pass2_variant_ratios = [(float(f), float(raw_tau_slow[i])) for i, f in enumerate(smoothed_fracs)]
                    smoothed_ratios = smoothed_fracs
            
            # Keep tau_d_vec from pass 1 (already estimated, use as-is for pass 2)
            # The smoothing is on RATIOS, not on tau_d_fast
            pass2_tau_d = tau_d_vec.copy()
            
            # Run second pass with FIXED smoothed ratios per event
            # Build design matrix directly with one ratio per event (no variant search)
            n_events = len(stim_times)
            jitter_variant_s = np.asarray(jitter_variant_ms, float) / 1000.0 if jitter_variant_ms is not None else np.array([0.0])
            n_jitter = len(jitter_variant_s)
            
            # Build kernels: each event has only jitter variants, fixed ratio
            kernel_columns = []
            for i_event in range(n_events):
                st = float(stim_times[i_event])
                tau_d = float(pass2_tau_d[i_event])
                frac_ratio = pass2_variant_ratios[i_event]  # THIS event's smoothed ratio
                for jitter_s in jitter_variant_s:
                    anchor = st + event_t0_s + jitter_s
                    k = _build_variant_kernel(
                        t - anchor, tau_r, tau_d, frac_ratio,
                        event_idx=i_event, n_events=n_events,
                    )
                    kernel_columns.append(k)
            
            X_p2 = np.column_stack(kernel_columns) if kernel_columns else np.zeros((t.size, 0))
            
            # Compute weights
            weights_p2 = _calculate_nnls_weights(
                t,
                stim_times,
                isi,
                weight_mode,
                weight_tau_s,
                y_ref=y_avg,
                peak_window_s=peak_window_s,
                peak_weight=peak_weight,
            )
            # Apply last-event tail downweighting
            last_event_tail_tau_s = cfg.get('nnls_last_event_tail_tau_s')
            pw_s = peak_window_s if peak_window_s is not None else 0.010
            weights_p2 = _apply_last_event_tail_weights(
                weights_p2, t, stim_times, isi, last_event_tail_tau_s, pw_s
            )
            
            # NNLS solve
            a_p2 = _nnls_weighted(X_p2, y_avg, weights_p2)
            yhat_avg_p2 = X_p2 @ a_p2
            
            # Reshape to [n_events, n_jitter]
            a_p2_reshaped = a_p2.reshape(n_events, n_jitter)
            
            # Find dominant jitter per event
            dom_jitter_idx = np.argmax(a_p2_reshaped, axis=1)
            d_avg_p2 = np.array([jitter_variant_s[j] for j in dom_jitter_idx])
            
            # Sum amplitudes across jitter variants
            a_avg_p2 = np.sum(a_p2_reshaped, axis=1)
            
            # Build components
            comp_avg_p2 = []
            X_avg_p2 = np.zeros((t.size, n_events))
            for i_event in range(n_events):
                st = float(stim_times[i_event])
                tau_d = float(pass2_tau_d[i_event])
                frac_ratio = pass2_variant_ratios[i_event]
                comp_event = np.zeros_like(y_avg)
                for i_jitter, jitter_s in enumerate(jitter_variant_s):
                    idx = i_event * n_jitter + i_jitter
                    amp_var = a_p2[idx]
                    if amp_var > 0:
                        anchor = st + event_t0_s + jitter_s
                        k = _build_variant_kernel(
                            t - anchor, tau_r, tau_d, frac_ratio,
                            event_idx=i_event, n_events=n_events,
                        )
                        comp_event += amp_var * k
                comp_avg_p2.append(comp_event)
                X_avg_p2[:, i_event] = comp_event / max(a_avg_p2[i_event], 1e-12)
            
            # Build variant_info for pass 2
            variant_info_p2 = {
                'n_template_variants': 1,  # Fixed ratio per event
                'n_jitter_variants': n_jitter,
                'variant_ratios': pass2_variant_ratios,
                'dominant_template_ratio': pass2_variant_ratios,  # These are the smoothed ratios
                'dominant_jitter_ms': d_avg_p2 * 1000.0,
            }
            
            # Compare RMS error between passes
            zmask_cmp = (t >= float(train_start) - cfg['pre_zoom_s']) & (t < float(train_start) + float(isi) * int(n_pulses) + cfg['post_zoom_s'])
            _accept_pass2 = True
            if np.any(zmask_cmp):
                rms_pass1 = np.sqrt(np.nanmean((y_avg[zmask_cmp] - yhat_avg[zmask_cmp])**2))
                rms_pass2 = np.sqrt(np.nanmean((y_avg[zmask_cmp] - yhat_avg_p2[zmask_cmp])**2))
                improvement = (rms_pass1 - rms_pass2) / rms_pass1 * 100 if rms_pass1 > 0 else 0
                progress_print(f"[TWO-PASS] Slow-fraction smoothed ({dec_mode}): RMS change={improvement:+.1f}%")
                if bool(cfg.get('nnls_two_pass_guard', False)) and improvement < 0:
                    _accept_pass2 = False
                    progress_print(f"[TWO-PASS] Pass 2 worsened fit — keeping pass 1 result")

            if _accept_pass2:
                a_avg, d_avg, X_avg, yhat_avg, comp_avg = a_avg_p2, d_avg_p2, X_avg_p2, yhat_avg_p2, comp_avg_p2
                variant_info_avg = variant_info_p2
                variant_info_avg['two_pass'] = True
                variant_info_avg['raw_slow_fraction'] = np.asarray(dominant_ratios_pass1, float) if is_scalar_ratios else dominant_ratios_pass1
                variant_info_avg['smoothed_slow_fraction'] = smoothed_ratios
            
        except Exception as e:
            progress_print(f"[TWO-PASS] Failed: {e}")

    # NNLS fit diagnostic logging disabled for cleaner output
    # (per-event RMS, amplitudes, jitter details removed)

    need_avg_sg = (
        ('savgol' in traces)
        or (meas == 'SAVGOL')
        or (failm == 'SAVGOL')
        or (weight_mode == 'savgol')
    )
    if need_avg_sg:
        y_sg_avg = sg_smooth(fill_nans_timewise(y_avg, t), sgW, sgP)
    else:
        y_sg_avg = None

    amp_nnls_avg = compute_localmax_corrected_amps(
        t, yhat_avg, stim_times, win_ms, n_avg, pre_ms, d_avg, tau_r, tau_d_vec,
        event_t0_s=event_t0_s,
    )
    amp_nnls_corr_avg = compute_peak_corrected_from_components(
        t, yhat_avg, stim_times, comp_avg, win_ms=win_ms, pre_ms=pre_ms
    )
    def _norm(a):
        a = np.asarray(a, float)
        d = a[0] if a.size else np.nan
        return a / d if np.isfinite(d) and abs(d) > 1e-12 else a * np.nan
    ppr_nnls_avg = _norm(amp_nnls_avg)
    ppr_nnls_corr_avg = _norm(amp_nnls_corr_avg)
    amp_raw_avg = compute_localmax_corrected_amps(
        t, y_avg, stim_times, win_ms, n_avg, pre_ms, d_avg, tau_r, tau_d_vec,
        event_t0_s=event_t0_s,
    )
    amp_raw_corr_avg = compute_peak_corrected_from_components(
        t, y_avg, stim_times, comp_avg, win_ms=win_ms, pre_ms=pre_ms
    )
    amp_sg_avg = compute_localmax_corrected_amps(
        t,
        y_sg_avg if y_sg_avg is not None else y_avg,
        stim_times,
        win_ms,
        n_avg,
        pre_ms,
        d_avg,
        tau_r,
        tau_d_vec,
        event_t0_s=event_t0_s,
        guide_y=yhat_avg,
    )
    amp_sg_corr_avg = compute_peak_corrected_from_components(
        t, (y_sg_avg if y_sg_avg is not None else y_avg), stim_times, comp_avg, win_ms=win_ms, pre_ms=pre_ms
    )

    # Optional: average plot with a left event-fit panel (0-50 ms) + right main plot
    figure = None
    fit_diag_figure = None
    if want_plot:
        # If showing only average (no trials, no baseline), use single row even with residuals
        # Residuals will be shown in a small inset instead of a separate row
        need_second_row = plot_residuals and (plot_trials or baseline_figs)
        if need_second_row:
            figure = plt.figure(figsize=(12, 9.2))
            gs = figure.add_gridspec(2, 2, height_ratios=[2.4, 1.4], width_ratios=[1.5, 4], wspace=0.15, hspace=0.28)
        else:
            figure = plt.figure(figsize=(12, 5))
            gs = figure.add_gridspec(1, 2, width_ratios=[1.5, 4], wspace=0.15)

        # Add filename as title if provided
        if filename is not None:
            figure.suptitle(filename, fontsize=11, fontweight='bold', y=0.98)

        resid_avg = None
        model_avg_for_resid = None
        if plot_residuals:
            if meas == 'SAVGOL' and (y_sg_avg is not None):
                model_avg_for_resid = y_sg_avg
            else:
                model_avg_for_resid = yhat_avg
            if model_avg_for_resid is not None:
                try:
                    resid_avg = np.asarray(y_avg - model_avg_for_resid, float)
                except Exception:
                    resid_avg = None
        # Left: aggregated event + model fit (−3..next stim−guard)
        axL = figure.add_subplot(gs[0, 0])
        _trim_spines(axL)
        try:
            # IMPORTANT: Use recut time vectors when available to ensure alignment
            # between snippets and their time axis
            if recut_t_rel is not None and recut_avg is not None:
                t_ms_evt = recut_t_rel * 1000.0
                y_evt = recut_avg
            elif 't_avg_evt' in locals():
                t_ms_evt = t_avg_evt
                y_evt = y_avg_evt
            else:
                t_ms_evt = (t - float(train_start)) * 1000.0
                y_evt = y_avg

            isi_ms = float(isi) * 1000.0
            # Display window: from -3 ms before stim to just before next stim
            # Use a 3 ms guard before the next stimulus to avoid overlap
            min_x = -3.0
            max_x = max(isi_ms - 3.0, 0.0)
            m0 = (t_ms_evt >= min_x) & (t_ms_evt < max_x)

            # Always show individual snippets if available (regardless of fit_source mode)
            if recut_snippets is not None and len(recut_snippets) > 0:
                try:
                    t_rel_ms = recut_t_rel * 1000.0 if recut_t_rel is not None else t_ms_evt
                    # Debug: print snippet information
                    # Plot debug messages (commented out - enable if needed for debugging)
                    # try:
                    #     progress_print(f"[plot] recut_t_rel is {'None' if recut_t_rel is None else f'array of size {recut_t_rel.size}'}")
                    #     progress_print(f"[plot] t_ms_evt range: {t_ms_evt[0]:.1f} to {t_ms_evt[-1]:.1f} ms, size {t_ms_evt.size}")
                    #     progress_print(f"[plot] t_rel_ms (used for plotting) range: {t_rel_ms[0]:.1f} to {t_rel_ms[-1]:.1f} ms, size {t_rel_ms.size}")
                    #     if t_rel_ms.size > 1:
                    #         dt_plot = np.median(np.diff(t_rel_ms))
                    #         progress_print(f"[plot] Time step dt = {dt_plot:.5f} ms ({1000.0/dt_plot:.1f} points per ms)")
                    #     progress_print(f"[plot] Plotting {len(recut_snippets)} snippets, each of size {recut_snippets[0].size if len(recut_snippets) > 0 else 'N/A'}")
                    #     if len(recut_snippets) > 0 and recut_snippets[0].size != t_rel_ms.size:
                    #         progress_print(f"[plot] WARNING: Size mismatch! snippet size {recut_snippets[0].size} != t_rel_ms size {t_rel_ms.size}")
                    # except Exception as e:
                    #     progress_print(f"[plot] Debug failed: {e}")
                    #     pass
                    # Plot individual snippets with low alpha
                    for i, snippet in enumerate(recut_snippets):
                        snippet_arr = np.asarray(snippet, float)
                        if snippet_arr.size == t_rel_ms.size:
                            m_snip = (t_rel_ms >= min_x) & (t_rel_ms < max_x)
                            axL.plot(t_rel_ms[m_snip], snippet_arr[m_snip],
                                   color='gray', alpha=0.1, lw=0.5, zorder=1)
                    # Plot average on top in black
                    if recut_avg is not None:
                        avg_arr = np.asarray(recut_avg, float)
                        if avg_arr.size == t_rel_ms.size:
                            m_avg = (t_rel_ms >= min_x) & (t_rel_ms < max_x)
                            axL.plot(t_rel_ms[m_avg], avg_arr[m_avg],
                                   color='k', lw=2.0, label='Average', zorder=2)
                    else:
                        axL.plot(t_ms_evt[m0], y_evt[m0], color='k', lw=1.5, label='Average', zorder=2)
                except Exception as e:
                    # Fallback to simple average plot
                    try:
                        progress_print(f"[plot] Failed to plot snippets: {e}")
                    except Exception:
                        pass
                    axL.plot(t_ms_evt[m0], y_evt[m0], color='k', lw=1.5, label='Average')
            else:
                # No snippets available: show single averaged event
                axL.plot(t_ms_evt[m0], y_evt[m0], color='k', lw=1.5, label='Average')
            # Overlay best-fit library model matching current kernel choice
            try:
                try:
                    from Model_Calibration.event_models import get_event_model
                except Exception:
                    from event_models import get_event_model  # type: ignore
                # Resolve the effective model name
                _name = event_model_fit if event_model == 'iglusnfr_tri' else event_model
                if _name.startswith('library:'):
                    _name = _name.split(':', 1)[1].strip().lower()
                spec = get_event_model(_name)
                tf = t_ms_evt[m0]; yf = y_evt[m0]
                # Build overlay params to reflect the model actually used:
                popt = None
                if _name in {'double_exp','cooperative','bilinear'}:
                    # Use the kinetics selected for this run (tau_r, tau_d0)
                    # Respect any fitted t_onset so the overlay shifts correctly
                    t_onset_ms = 0.0
                    try:
                        if 'fitted' in locals() and fitted is not None:
                            t_onset_ms = float(fitted.get('t_onset', 0.0))
                    except Exception:
                        t_onset_ms = 0.0
                    if _name == 'double_exp':
                        # [amp, tau_rise(s), tau_decay(s), t_onset(ms)]
                        pars = [1.0, float(tau_r), float(tau_d0), t_onset_ms]
                    elif _name == 'cooperative':
                        ems = cfg.get('event_model_settings', {}) or {}
                        n_used = float(ems.get('n_coop', 2.0))
                        # [amp, tau_rise(s), tau_decay(s), n_coop, t_onset(ms)]
                        pars = [1.0, float(tau_r), float(tau_d0), n_used, t_onset_ms]
                    else:  # bilinear expects ms values for rise/decay durations
                        pars = [1.0, float(tau_r)*1000.0, float(tau_d0)*1000.0, t_onset_ms]
                    yshape = spec['func'](tf, *pars)
                    denom = float(np.sum(yshape**2)) if np.isfinite(yshape).any() else 0.0
                    amp_ls = float(np.sum(yf*yshape))/denom if denom > 0 else 1.0
                    pars[0] = amp_ls
                    popt = pars
                    yhat_ev = spec['func'](tf, *popt)
                else:
                    # For fixed-template models, DO NOT refit here: honor
                    # the effective event_model_settings that were applied to
                    # the kernel earlier. Build the parameter vector in the
                    # order expected by the spec and only solve a linear LS
                    # for amplitude so the overlay matches scale.
                    params = []
                    t_onset_ms = 0.0
                    try:
                        if 'fitted' in locals() and fitted is not None:
                            t_onset_ms = float(fitted.get('t_onset', 0.0))
                    except Exception:
                        t_onset_ms = 0.0
                    ems = cfg.get('event_model_settings', {}) or {}
                    for name in spec['params']:
                        if name == 'amp':
                            params.append(1.0)
                        elif name == 't_onset':
                            params.append(t_onset_ms)
                        else:
                            # Prefer values from the recent global fit if available;
                            # then event_model_settings (initial guesses) if fit missing;
                            # then tau_r/tau_d0 mapping; finally default p0.
                            if 'fitted' in locals() and isinstance(fitted, dict) and name in fitted and np.isfinite(fitted.get(name, np.nan)):
                                params.append(float(fitted[name]))
                            elif name in ems and np.isfinite(ems.get(name, np.nan)):
                                params.append(float(ems[name]))
                            elif name == 'tau_decay':
                                params.append(float(tau_d0))  # seconds
                            elif name == 'tau_rise':
                                params.append(float(tau_r))
                            else:
                                # If unknown, fall back to p0 for stability
                                p0 = spec['p0_func'](yf, tf)
                                idx = spec['params'].index(name)
                                params.append(float(p0[idx]))
                    # Compute LS amplitude against the fixed shape
                    yshape = spec['func'](tf, *params)
                    denom = float(np.sum(yshape**2)) if np.isfinite(yshape).any() else 0.0
                    amp_ls = float(np.sum(yf * yshape)) / denom if denom > 0 else 1.0
                    params[0] = amp_ls
                    popt = params
                    yhat_ev = spec['func'](tf, *popt)
                fit_color = 'crimson'
                fit_suffix = ''
                if event_model == 'iglusnfr_tri' and recut_slow_replaced:
                    fit_color = 'limegreen'
                    fit_suffix = '\n(τ_slow override)'
                    progress_print(f"[plot] Using overridden tau_decay_slow: {fitted.get('tau_decay_slow', 0)*1000:.1f}ms")
                axL.plot(tf, yhat_ev, color=fit_color, ls='--', lw=1.8, label=_name)
                try:
                    # Omit 't_onset' and stack vertically; include amp at top for context
                    pairs = [(n, v) for n, v in zip(spec['params'], popt)]
                    pairs = [(n, v) for n, v in pairs if n != 't_onset']
                    _pb = cfg.get('parameter_bounds', {}) or {}
                    _lines = [f"{n}={v:.3g}" for n, v in pairs]
                    _colors = []
                    for n, v in pairs:
                        _b = _pb.get(n)
                        if _b and isinstance(_b, (tuple, list)) and len(_b) == 2:
                            _lo, _hi = float(_b[0]), float(_b[1])
                            if np.isfinite(_lo) and abs(v - _lo) < 1e-12 * max(1, abs(_lo)):
                                _colors.append('blue')
                            elif np.isfinite(_hi) and abs(v - _hi) < 1e-12 * max(1, abs(_hi)):
                                _colors.append('red')
                            else:
                                _colors.append('black')
                        else:
                            _colors.append('black')
                    if fit_suffix:
                        _lines.append(fit_suffix.strip())
                        _colors.append('black')
                    _y0 = 0.98
                    _dy = 0.065
                    for _li, (_ltxt, _lcol) in enumerate(zip(_lines, _colors)):
                        axL.text(0.98, _y0 - _li * _dy, _ltxt,
                                 transform=axL.transAxes, fontsize=8, color=_lcol,
                                 va='top', ha='right',
                                 bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8, lw=0))
                except Exception:
                    pass
            except Exception:
                pass
            axL.axvline(0.0, color='k', ls=':', alpha=0.5, lw=0.8)
            axL.set_xlim(min_x, max_x)
            # Autoscale y with a small margin to avoid a squashed panel
            try:
                # For average/individual mode with snippets, scale to snippet data
                if fit_source in ('average', 'individual') and recut_snippets is not None and len(recut_snippets) > 0:
                    # Collect all snippet values in the display window
                    all_vals = []
                    t_rel_ms = recut_t_rel * 1000.0 if recut_t_rel is not None else t_ms_evt
                    m_window = (t_rel_ms >= min_x) & (t_rel_ms < max_x)
                    for snippet in recut_snippets:
                        snippet_arr = np.asarray(snippet, float)
                        if snippet_arr.size == t_rel_ms.size:
                            all_vals.extend(snippet_arr[m_window][np.isfinite(snippet_arr[m_window])])
                    if all_vals:
                        # Use 5th and 95th percentile to exclude outliers
                        ymins = float(np.percentile(all_vals, 5))
                        ymaxs = float(np.percentile(all_vals, 95))
                    else:
                        ymins, ymaxs = 0.0, 1.0
                else:
                    # Global mode: use y_evt
                    y_slice = y_evt[m0]
                    ymins = np.nanmin(y_slice) if np.size(y_slice) else 0.0
                    ymaxs = np.nanmax(y_slice) if np.size(y_slice) else 1.0

                # Include model fit in Y range
                if 'yhat_ev' in locals():
                    ymins = min(ymins, float(np.nanmin(yhat_ev)))
                    ymaxs = max(ymaxs, float(np.nanmax(yhat_ev)))

                span = max(1e-6, ymaxs - ymins)
                pad = 0.1 * span
                axL.set_ylim(ymins - pad, ymaxs + pad)
            except Exception:
                pass
            axL.set_xlabel('Time (ms)')
            axL.set_ylabel('ΔF/F0' if use_dff else 'ΔF')
            try:
                axL.set_title(f'Event fit (−3.0–{max_x:.1f} ms)', fontsize=10)
            except Exception:
                axL.set_title('Event fit', fontsize=10)
        except Exception:
            pass

        # Right: main average plot
        ax = figure.add_subplot(gs[0, 1])
        _trim_spines(ax)
        zmask, z0, z1 = time_zoom_mask(t, float(train_start), float(isi), int(n_pulses), cfg['pre_zoom_s'], cfg['post_zoom_s'])
        tz = t[zmask]
        for st in stim_times:
            ax.axvline(st, color='k', linestyle=':', linewidth=0.8, alpha=0.6)
        if 'raw' in traces:
            ax.plot(tz, y_avg[zmask], label='raw', color='0.6')
        if 'savgol' in traces and y_sg_avg is not None:
            ax.plot(tz, y_sg_avg[zmask], label='savgol', color='tab:green')
        if 'nnls' in traces:
            ax.plot(tz, yhat_avg[zmask], label='nnls model', color='tab:blue')
            # Debug: print blue NNLS values at last event
            last_stim = float(stim_times[-1])
            post_last_mask = (tz >= last_stim) & (tz <= last_stim + 0.3)
        if plot_residuals and resid_avg is not None:
            try:
                resid_offset = -1
                ax.plot(
                    tz,
                    resid_avg[zmask] + resid_offset,
                    label='residual (offset)',
                    color='tab:purple',
                    linewidth=1.2,
                    alpha=0.9,
                )
                ax.axhline(resid_offset, color='tab:purple', linestyle=':', linewidth=0.8, alpha=0.7)
            except Exception:
                pass
        # Replace the orange line plotting section with:
        # Replace the orange line plotting with cumulative reconstruction:
        if show_decay and 'nnls' in traces and comp_avg is not None:
            cumulative = np.zeros_like(y_avg)
            for p in range(len(stim_times)):
                if p >= len(comp_avg):
                    continue
                # Add this component to the cumulative sum
                cumulative = cumulative + comp_avg[p]

                # Plot the cumulative reconstruction up to this point
                # This shows the "baseline" including all decay from previous pulses
                ax.plot(
                    tz,
                    cumulative[zmask],
                    color='tab:orange',
                    linestyle='--',
                    linewidth=1.0,
                    alpha=0.6 - p * 0.04,  # Fade with each pulse
                )
        # Last-event pre-refit fit overlay (commented out - removed per user request)
        # try:
        #     if dec_mode in ('linear','free_monotonic') and (tau_last_display is not None) and (amp_last_display is not None):
        #         last_st = float(stim_times[-1]) + event_t0_s
        #         k_last = _KERNEL_FUN(tz - last_st, tau_r, float(tau_last_display))
        #         ax.plot(tz, float(amp_last_display) * k_last, color='crimson', linestyle='--', linewidth=1.4, alpha=0.9, label='last fit (pre-refit)')
        # except Exception:
        #     pass

        # Optional: overlay peak markers and residual-at-peak triangles
        if plot_peaks_details:
            # Choose which series defines the "measurement" trace for peak picking
            if meas == 'SAVGOL' and (y_sg_avg is not None):
                y_for_peaks = y_sg_avg
                amp_corr_for_avg_dots = amp_sg_corr_avg
            elif meas == 'RAW':
                y_for_peaks = y_avg
                amp_corr_for_avg_dots = amp_raw_corr_avg
            else:
                y_for_peaks = yhat_avg
                amp_corr_for_avg_dots = amp_nnls_corr_avg

            # Build cumulative baseline from previous pulses only using NNLS components
            # For pulse p, baseline_prev[p, :] = sum_{k < p} comp_avg[k]
            baseline_prev_only = []
            if comp_avg is not None:
                cum = np.zeros_like(y_avg)
                for p in range(len(stim_times)):
                    baseline_prev_only.append(cum.copy())
                    if p < len(comp_avg):
                        cum = cum + comp_avg[p]

            peak_ts: list = []
            peak_vals: list = []
            resid_vals: list = []
            for p, st in enumerate(stim_times):
                # Find peak directly on the NNLS model so the red dot sits on the blue line
                tp, vp = pick_peak_on_series(t, y_for_peaks, float(st), win_ms, pre_ms)
                peak_ts.append(float(tp))
                peak_vals.append(float(vp))
                # Residual-under-peak = baseline from prior pulses at that time
                try:
                    i0 = int(np.argmin(np.abs(t - tp)))
                    base_prev = baseline_prev_only[p][i0] if baseline_prev_only else 0.0
                    resid_vals.append(float(base_prev))
                except Exception:
                    resid_vals.append(np.nan)
            
            # Red circles at peaks (on the chosen measurement trace)
            try:
                ax.scatter(peak_ts, peak_vals, s=70, color='red', edgecolors='white', linewidths=0.9, zorder=6, label='peaks')
            except Exception:
                pass
            # Down-pointing triangles for residual at peak time
            try:
                # Triangles sit on the orange dashed baseline (previous pulses only)
                ax.scatter(peak_ts, resid_vals, s=60, marker='v', facecolors='white', edgecolors='tab:red', linewidths=1.0, zorder=5, label='residual at peak')
            except Exception:
                pass
                
        # Overlay last-event decay fit (orange dashed line)
        try:
            global _LAST_EVENT_DECAY_FIT
            if _LAST_EVENT_DECAY_FIT is not None:
                le_fit = _LAST_EVENT_DECAY_FIT
                tau_slow = le_fit.get('tau_slow', np.nan)

                if np.isfinite(tau_slow) and 0 < tau_slow < 0.350:
                    t_decay_start = float(le_fit.get('t_start', float(stim_times[-1]) + 0.005))
                    baseline_fit = float(le_fit.get('baseline', 0.0))

                    # Always anchor amplitude to the NNLS model at t_decay_start —
                    # the stored log-fit a_slow can be wrong when baseline is mis-estimated.
                    ref = yhat_avg if yhat_avg is not None else y_avg
                    a_slow = float(np.interp(t_decay_start, t, ref)) - baseline_fit

                    # Only draw if the model predicts a real positive signal at that point
                    if not (np.isfinite(a_slow) and a_slow > 0):
                        raise ValueError("no positive signal at decay start")

                    t_end = min(z1, t_decay_start + max(5.0 * tau_slow, 0.200))
                    t_fit = np.linspace(t_decay_start, t_end, 200)
                    y_fit = baseline_fit + a_slow * np.exp(-(t_fit - t_decay_start) / tau_slow)

                    ax.plot(
                        t_fit,
                        y_fit,
                        color='tab:orange',
                        ls='--',
                        lw=2.0,
                        alpha=0.85,
                        label=f'last-event fit (τ={tau_slow*1000:.1f}ms)',
                    )
        except Exception:
            pass
            
        ax.set_xlim(z0, z1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('ΔF/F0' if use_dff else 'ΔF')
        ax.legend(loc='upper left', frameon=False)
        ax.set_title('Average trace (selected overlays)')

        # Residual diagnostics panel (average): place directly under the main
        # average panel and show a small inset histogram (no separate figure).
        if plot_residuals and (resid_avg is not None) and (model_avg_for_resid is not None):
            try:
                if need_second_row:
                    # Bottom-right: residual trace aligned with the top-right panel
                    axR = figure.add_subplot(gs[1, 1], sharex=ax)
                    axR.plot(tz, resid_avg[zmask], color='tab:purple', lw=1.2, label='residual (avg − model)')
                    axR.axvline(float(train_start), color='k', ls=':', lw=0.8, alpha=0.6)
                    axR.set_xlim(z0, z1)
                    axR.set_xlabel('Time (s)')
                    axR.set_ylabel('ΔF/F0' if use_dff else 'ΔF')
                    axR.set_title('Residuals (average)')
                    axR.legend(loc='upper right', frameon=False, fontsize=8)
                    _trim_spines(axR)

                    # Inset histogram of residuals in the zoom window with fixed bins and Gaussian fit
                    try:
                        ax_in = axR.inset_axes([0.70, 0.55, 0.28, 0.4])
                        rv_train = np.asarray(resid_avg[zmask], float)
                        rv_train = rv_train[np.isfinite(rv_train)]
                        rv_base = None
                        try:
                            rv_base = np.asarray(resid_avg[baseline_mask], float)
                            rv_base = rv_base[np.isfinite(rv_base)]
                        except Exception:
                            rv_base = None
                        if rv_train.size:
                            bin_w = (0.01 if use_dff else 10.0)
                            lo = float(np.nanmin(rv_train))
                            hi = float(np.nanmax(rv_train))
                            if not np.isfinite(lo):
                                lo = 0.0
                            if not np.isfinite(hi) or hi <= lo:
                                hi = lo + bin_w
                            edges = np.arange(lo, hi + bin_w, bin_w)
                            counts, edges = np.histogram(rv_train, bins=edges)
                            widths = np.diff(edges)
                            ax_in.bar(edges[:-1], counts, width=widths, align='edge',
                                      color='#6b4fa3', alpha=0.9, edgecolor='none')
                            # Baseline-only Gaussian fit, scaled to train residual count
                            if rv_base is not None and rv_base.size:
                                try:
                                    mu = float(np.nanmean(rv_base))
                                    sigma = float(np.nanstd(rv_base))
                                except Exception:
                                    mu, sigma = float('nan'), float('nan')
                                if np.isfinite(sigma) and sigma > 0:
                                    centers = edges[:-1] + 0.5 * widths
                                    pdf_centers = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(
                                        -0.5 * ((centers - mu) / sigma) ** 2
                                    )
                                    N = rv_train.size
                                    expected = N * bin_w * pdf_centers
                                    peak_expected = float(np.nanmax(expected)) if expected.size else 0.0
                                    cutoff = max(0.5, 0.01 * peak_expected)
                                    base = np.where(expected >= cutoff,
                                                    np.minimum(counts.astype(float), expected),
                                                    0.0)
                                    excess = counts.astype(float) - base
                                    if np.any(excess > 0):
                                        ax_in.bar(edges[:-1], excess, width=widths, align='edge',
                                                  bottom=base, color='tab:red', alpha=0.85, edgecolor='none')
                                    x = np.linspace(edges[0], edges[-1], 400)
                                    pdf = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(
                                        -0.5 * ((x - mu) / sigma) ** 2
                                    )
                                    y = N * bin_w * pdf
                                    ax_in.plot(x, y, color='#26457a', linewidth=1.4, label='baseline fit')
                        ax_in.set_title('residual', fontsize=8)
                        ax_in.tick_params(labelsize=7)
                    except Exception:
                        pass

                    # Add an empty placeholder under the left event-fit panel to
                    # keep the grid balanced.
                    try:
                        ax_placeholder = figure.add_subplot(gs[1, 0])
                        ax_placeholder.axis('off')
                    except Exception:
                        pass
                else:
                    # When showing only average, add small inset at bottom-left of main plot
                    try:
                        # Smaller inset at bottom-left [left, bottom, width, height]
                        ax_in = ax.inset_axes([0.02, 0.02, 0.18, 0.22])
                        rv_train = np.asarray(resid_avg[zmask], float)
                        rv_train = rv_train[np.isfinite(rv_train)]
                        rv_base = None
                        try:
                            rv_base = np.asarray(resid_avg[baseline_mask], float)
                            rv_base = rv_base[np.isfinite(rv_base)]
                        except Exception:
                            rv_base = None
                        if rv_train.size:
                            bin_w = (0.01 if use_dff else 10.0)
                            lo = float(np.nanmin(rv_train))
                            hi = float(np.nanmax(rv_train))
                            if not np.isfinite(lo):
                                lo = 0.0
                            if not np.isfinite(hi) or hi <= lo:
                                hi = lo + bin_w
                            edges = np.arange(lo, hi + bin_w, bin_w)
                            counts, edges = np.histogram(rv_train, bins=edges)
                            widths = np.diff(edges)
                            ax_in.bar(edges[:-1], counts, width=widths, align='edge',
                                      color='#6b4fa3', alpha=0.9, edgecolor='none')
                            # Baseline-only Gaussian fit, scaled to train residual count
                            if rv_base is not None and rv_base.size:
                                try:
                                    mu = float(np.nanmean(rv_base))
                                    sigma = float(np.nanstd(rv_base))
                                except Exception:
                                    mu, sigma = float('nan'), float('nan')
                                if np.isfinite(sigma) and sigma > 0:
                                    centers = edges[:-1] + 0.5 * widths
                                    pdf_centers = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(
                                        -0.5 * ((centers - mu) / sigma) ** 2
                                    )
                                    N = rv_train.size
                                    expected = N * bin_w * pdf_centers
                                    peak_expected = float(np.nanmax(expected)) if expected.size else 0.0
                                    cutoff = max(0.5, 0.01 * peak_expected)
                                    base = np.where(expected >= cutoff,
                                                    np.minimum(counts.astype(float), expected),
                                                    0.0)
                                    excess = counts.astype(float) - base
                                    if np.any(excess > 0):
                                        ax_in.bar(edges[:-1], excess, width=widths, align='edge',
                                                  bottom=base, color='tab:red', alpha=0.85, edgecolor='none')
                                    x = np.linspace(edges[0], edges[-1], 400)
                                    pdf = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(
                                        -0.5 * ((x - mu) / sigma) ** 2
                                    )
                                    y = N * bin_w * pdf
                                    ax_in.plot(x, y, color='#26457a', linewidth=1.4)
                        ax_in.set_title('residuals', fontsize=7)
                        ax_in.tick_params(labelsize=6)
                        # Remove top and right spines
                        ax_in.spines['top'].set_visible(False)
                        ax_in.spines['right'].set_visible(False)
                        # Set transparent background
                        ax_in.patch.set_facecolor('none')
                        ax_in.patch.set_alpha(0.0)
                    except Exception:
                        pass
            except Exception:
                pass

        # Show the complete average figure immediately
        try:
            plt.show(block=False); plt.pause(0.05)
        except Exception:
            pass

        # Fit diagnostic figure: show immediately after average figure
        if cfg.get('fit_diagnostic_plot', False) and interpolated_settings:
            try:
                # Layout with uniform setting panels and a dedicated legend panel
                # at the top-left of the settings grid. Weight kernel remains
                # full-width on the first row.
                import math
                # Order settings (put fast decay first if present)
                settings_ordered = list(interpolated_settings)
                for i, info in enumerate(interpolated_settings):
                    name_i = str(info.get('name', '')).lower()
                    if name_i in {'tau_decay', 'tau_decay_fast'} and i != 0:
                        settings_ordered.insert(0, settings_ordered.pop(i))
                        break

                total_panels = 1 + len(settings_ordered)  # +1 for legend
                n_cols = min(3, total_panels) if total_panels > 0 else 1
                n_rows = int(math.ceil(total_panels / float(n_cols)))

                fig_width = 5.5 + 3.2 * n_cols
                fit_diag_figure = plt.figure(figsize=(fig_width, 6.8))
                height_ratios = [1.2] + [1.0] * n_rows
                gs_fd = fit_diag_figure.add_gridspec(
                    1 + n_rows, n_cols, height_ratios=height_ratios, hspace=0.28, wspace=0.22
                )

                # Weight kernel axis (full width)
                ax_w = fit_diag_figure.add_subplot(gs_fd[0, :])

                # Weight kernel panel
                if weight_mode == 'savgol' and y_sg_avg is None:
                    y_vis = sg_smooth(fill_nans_timewise(y_avg, t), sgW, sgP)
                else:
                    y_vis = y_sg_avg
                weights = _calculate_nnls_weights(
                    t,
                    stim_times,
                    isi,
                    weight_mode,
                    weight_tau_s,
                    y_ref=y_vis,
                    peak_window_s=peak_window_s,
                    peak_weight=peak_weight,
                )
                # Apply last-event tail downweighting for visualization
                last_event_tail_tau_s = cfg.get('nnls_last_event_tail_tau_s')
                pw_s = peak_window_s if peak_window_s is not None else 0.010
                weights = _apply_last_event_tail_weights(
                    weights, t, stim_times, isi, last_event_tail_tau_s, pw_s
                )
                ax_w.plot(t, weights, color='#1f77b4', linewidth=2.0, label=f'{weight_mode} weight')
                for i, st in enumerate(stim_times):
                    label = 'stimulus' if i == 0 else None
                    ax_w.axvline(st, color='red', linestyle='--', alpha=0.6, linewidth=0.9, label=label)
                ax_w.set_xlabel('Time (s)')
                ax_w.set_ylabel('Weight')
                if weight_tau_s is not None:
                    ax_w.set_title(f'Weight kernel (τ={weight_tau_s*1000:.1f} ms)')
                else:
                    ax_w.set_title('Weight kernel')
                # No per-axis legend; a global legend is rendered separately
                ax_w.grid(True, alpha=0.2)

                anchor_styles = {
                    'first': {'color': 'green', 'vline': '-.', 'hline': ':', 'marker': 's', 'size': 250},
                    'mid': {'color': '#1f77b4', 'vline': '--', 'hline': ':', 'marker': 'D', 'size': 200},
                    'final': {'color': 'purple', 'vline': '-.', 'hline': ':', 'marker': '*', 'size': 300},
                }
                anchor_names = {
                    'first': 'first event',
                    'mid': 'mid event',
                    'final': 'final event',
                }

                def _plot_setting_axis(ax, info: Dict[str, Any]) -> None:
                    _trim_spines(ax)
                    pulses = np.arange(1, np.asarray(info['final']).size + 1, dtype=float)
                    scale = float(info.get('scale', 1.0))
                    label = str(info.get('display', info.get('name', 'setting')))
                    unit = str(info.get('unit', ''))
                    final_vals = np.asarray(info['final'], float) * scale

                    raw_vals = info.get('raw')
                    if raw_vals is not None and np.any(np.isfinite(raw_vals)):
                        ax.plot(
                            pulses,
                            np.asarray(raw_vals, float) * scale,
                            marker='o',
                            markersize=8,
                            color='#d62728',
                            linestyle=':',
                            linewidth=1.5,
                            alpha=0.8,
                            zorder=3,
                        )

                    constrained_vals = info.get('constrained')
                    if constrained_vals is not None and np.any(np.isfinite(constrained_vals)):
                        ax.scatter(
                            pulses,
                            np.asarray(constrained_vals, float) * scale,
                            marker='x',
                            s=80,
                            color='#ff7f0e',
                            linewidths=2,
                            alpha=0.8,
                            zorder=4,
                        )

                    ax.plot(
                        pulses,
                        final_vals,
                        marker='s',
                        markersize=6,
                        color='#2ca02c',
                        linewidth=2.5,
                        alpha=0.9,
                        zorder=5,
                    )

                    for anchor in info.get('anchors', []):
                        kind = anchor.get('kind')
                        style = anchor_styles.get(kind)
                        if style is None:
                            continue
                        idx = int(anchor.get('index', 0))
                        if idx < 0 or idx >= len(pulses):
                            continue
                        value = float(anchor.get('value', np.nan))
                        if not np.isfinite(value):
                            continue
                        y_disp = value * scale
                        x_pos = pulses[idx]
                        ax.axvline(
                            x_pos,
                            color=style['color'],
                            linestyle=style['vline'],
                            alpha=0.6,
                            linewidth=2.0,
                        )
                        ax.axhline(
                            y_disp,
                            color=style['color'],
                            linestyle=style['hline'],
                            alpha=0.4,
                            linewidth=1.5,
                        )
                        ax.scatter(
                            [x_pos],
                            [y_disp],
                            marker=style['marker'],
                            s=style['size'],
                            color=style['color'],
                            alpha=0.7,
                            zorder=10,
                            edgecolors='black',
                            linewidths=1.5,
                        )

                    ax.set_xlabel('Pulse #')
                    ylabel = label if not unit else f"{label} ({unit})"
                    ax.set_ylabel(ylabel)
                    if 'tau' in str(info.get('name', '')).lower() or 'tau' in label.lower():
                        ax.set_ylim(bottom=0.0)
                    # Simplify titles: show only the parameter label
                    ax.set_title(label)
                    ax.grid(True, alpha=0.3)

                # Legend panel at top-left of settings grid
                from matplotlib.lines import Line2D
                ax_leg = fit_diag_figure.add_subplot(gs_fd[1, 0])
                ax_leg.axis('off')
                handles = [
                    Line2D([0], [0], marker='o', color='#d62728', linestyle=':', linewidth=1.5,
                           markersize=8, label='initial per-event', markerfacecolor='#d62728'),
                    Line2D([0], [0], marker='x', color='#ff7f0e', linestyle='None',
                           markersize=8, markeredgewidth=2, label='constrained'),
                    Line2D([0], [0], marker='s', color='#2ca02c', linestyle='-', linewidth=2.5,
                           markersize=6, label='final progression'),
                    Line2D([0], [0], marker='s', color='green', linestyle='-', linewidth=2.0,
                           markersize=8, label='anchor: first'),
                    Line2D([0], [0], marker='D', color='#1f77b4', linestyle='--', linewidth=2.0,
                           markersize=8, label='anchor: mid'),
                    Line2D([0], [0], marker='*', color='purple', linestyle='-.', linewidth=2.0,
                           markersize=12, label='anchor: final'),
                ]
                ax_leg.legend(handles=handles, loc='center', ncol=3, frameon=True, fontsize=9, framealpha=0.9, title='Legend')

                # Fill the remaining slots row-major, skipping the legend cell
                idx = 0
                for r in range(1, 1 + n_rows):
                    for c in range(n_cols):
                        if r == 1 and c == 0:
                            continue  # legend cell already filled
                        if idx >= len(settings_ordered):
                            break
                        ax_s = fit_diag_figure.add_subplot(gs_fd[r, c])
                        _plot_setting_axis(ax_s, settings_ordered[idx])
                        idx += 1

                fit_diag_figure.tight_layout()
                try:
                    plt.show(block=False); plt.pause(0.05)
                except Exception:
                    pass
            except Exception as e:
                print(f"[warning] Failed to render fit diagnostics: {e}")

    # Per‑trial metrics and null thresholds (MAD rule) for A1
    per_trial: List[Dict] = []
    thr_list: List[float] = []
    pval_list: List[float] = []
    figures_trials = []  # optional per-trial figures
    
    # Get dominant template ratios from average fit for per-trial fitting
    dominant_template_ratios = None
    if variant_info_avg is not None:
        dominant_template_ratios = variant_info_avg.get('dominant_template_ratio')
    
    for j in range(Yd.shape[1]):
        yj = Yd[:, j]
        # Always compute SG-smoothed series; may be used for SAVGOL-based thresholds
        yj_sg = sg_smooth(yj, sgW, sgP)
        
        # Use variant kernel fitting if template variants were used for the average
        if dominant_template_ratios is not None and _VARIANT_KERNEL_BUILDER is not None:
            a_t, d_t, X_t, yhat_t, comp_t = fit_amplitudes_with_fixed_variants(
                yj, t, stim_times, tau_r, tau_d_vec,
                dominant_template_ratios=dominant_template_ratios,
                pre_zoom_s=cfg['pre_zoom_s'], post_zoom_s=cfg['post_zoom_s'],
                robust=True, huber_delta=cfg['huber_delta'], irls_iters=cfg['irls_iters'],
                allow_shift=allow_shift, delta_max_s=cfg['delta_max_s'], delta_step_s=cfg['delta_step_s'],
                shift_min_s=cfg['shift_min_s'], event_t0_s=event_t0_s,
            )
        else:
            a_t, d_t, X_t, yhat_t, comp_t = fit_amplitudes_no_overlap_forward(
                yj, t, stim_times, tau_r, tau_d_vec,
                pre_zoom_s=cfg['pre_zoom_s'], post_zoom_s=cfg['post_zoom_s'],
                robust=True, huber_delta=cfg['huber_delta'], irls_iters=cfg['irls_iters'],
                allow_shift=allow_shift, delta_max_s=cfg['delta_max_s'], delta_step_s=cfg['delta_step_s'],
                shift_min_s=cfg['shift_min_s'], event_t0_s=event_t0_s,
            )
        amp_raw = compute_localmax_corrected_amps(
            t, yj, stim_times, win_ms, n_avg, pre_ms, d_t, tau_r, tau_d_vec,
            event_t0_s=event_t0_s,
        )
        amp_raw_corr = compute_peak_corrected_from_components(
            t, yj, stim_times, comp_t, win_ms=win_ms, pre_ms=pre_ms
        )
        amp_sg = compute_localmax_corrected_amps(
            t, yj_sg, stim_times, win_ms, n_avg, pre_ms, d_t, tau_r, tau_d_vec,
            event_t0_s=event_t0_s, guide_y=yhat_t,
        )
        amp_sg_corr = compute_peak_corrected_from_components(
            t, yj_sg, stim_times, comp_t, win_ms=win_ms, pre_ms=pre_ms
        )
        amp_nn = compute_localmax_corrected_amps(
            t, yhat_t, stim_times, win_ms, n_avg, pre_ms, d_t, tau_r, tau_d_vec,
            event_t0_s=event_t0_s,
        )
        amp_nn_corr = compute_peak_corrected_from_components(
            t, yhat_t, stim_times, comp_t, win_ms=win_ms, pre_ms=pre_ms
        )

        # Choose threshold rule and null amplitude strategy (auto follows fail_method)
        eff_mode = ('sd' if failm == 'SAVGOL' or failm == 'RAW' else 'mad') if thr_mode == 'auto' else thr_mode
        if eff_mode not in ('mad', 'sd'):
            eff_mode = 'mad'

        if failm == 'SAVGOL' and eff_mode == 'sd':
            # Compute null amplitudes directly on SG baseline via windowed maxima
            idx = np.flatnonzero(baseline_mask)
            if idx.size >= 10:
                baseline_start = t[idx[0]]; baseline_end = t[idx[-1]]
                null_start = max(baseline_start, float(train_start) - cfg['f0_window_s'])
                null_end = min(baseline_end, float(train_start))
                # Candidate starts cover the full f0 window and exclude any
                # start whose measurement window would touch the train.
                st_min = null_start
                st_max = null_end - (cfg['peak_window_ms'] / 1000.0)
                cand = (t >= st_min) & (t <= st_max)
                starts = t[cand]
                if starts.size > int(cfg['null_sim_max_points']):
                    ii = np.linspace(0, starts.size - 1, int(cfg['null_sim_max_points'])).round().astype(int)
                    starts = starts[ii]
                null_amps = windowed_max(t, yj_sg, list(starts), win_ms, n_avg, pre_ms) if starts.size else np.array([])
            else:
                null_amps = np.array([])
            # Keep the sliding-NNLS null amplitudes as an additional per-trial
            # output so downstream analyses can inspect the full baseline-event
            # distribution independent of the threshold rule.
            null_amps_nnls = sample_null_amplitudes_consistent(
                yj, t, baseline_mask, tau_r, tau_d0,
                train_start=float(train_start), f0_window_s=cfg['f0_window_s'],
                pre_zoom_s=cfg['pre_zoom_s'], post_zoom_s=cfg['post_zoom_s'],
                robust=True, huber_delta=cfg['huber_delta'], irls_iters=cfg['irls_iters'],
                allow_shift=allow_shift, delta_max_s=cfg['delta_max_s'], delta_step_s=cfg['delta_step_s'],
                null_min_post_zoom_s=cfg['null_min_post_zoom_s'], null_sim_max_points=cfg['null_sim_max_points'],
                peak_window_ms=cfg['peak_window_ms'], peak_avg_points=cfg['peak_avg_points'], pre_peak_ms=cfg['pre_peak_ms'],
                shift_min_s=cfg['shift_min_s'], n_samples=1000,
                seed=cfg.get('seed', 42), event_t0_s=event_t0_s,
            )
        elif failm == 'RAW' and eff_mode == 'sd':
            # SD rule on RAW baseline windowed maxima
            idx = np.flatnonzero(baseline_mask)
            if idx.size >= 10:
                baseline_start = t[idx[0]]; baseline_end = t[idx[-1]]
                null_start = max(baseline_start, float(train_start) - cfg['f0_window_s'])
                null_end = min(baseline_end, float(train_start))
                st_min = null_start
                st_max = null_end - (cfg['peak_window_ms'] / 1000.0)
                cand = (t >= st_min) & (t <= st_max)
                starts = t[cand]
                if starts.size > int(cfg['null_sim_max_points']):
                    ii = np.linspace(0, starts.size - 1, int(cfg['null_sim_max_points'])).round().astype(int)
                    starts = starts[ii]
                null_amps = windowed_max(t, yj, list(starts), win_ms, n_avg, pre_ms) if starts.size else np.array([])
            else:
                null_amps = np.array([])
            # Keep the sliding-NNLS null amplitudes as an additional per-trial
            # output so downstream analyses can inspect the full baseline-event
            # distribution independent of the threshold rule.
            null_amps_nnls = sample_null_amplitudes_consistent(
                yj, t, baseline_mask, tau_r, tau_d0,
                train_start=float(train_start), f0_window_s=cfg['f0_window_s'],
                pre_zoom_s=cfg['pre_zoom_s'], post_zoom_s=cfg['post_zoom_s'],
                robust=True, huber_delta=cfg['huber_delta'], irls_iters=cfg['irls_iters'],
                allow_shift=allow_shift, delta_max_s=cfg['delta_max_s'], delta_step_s=cfg['delta_step_s'],
                null_min_post_zoom_s=cfg['null_min_post_zoom_s'], null_sim_max_points=cfg['null_sim_max_points'],
                peak_window_ms=cfg['peak_window_ms'], peak_avg_points=cfg['peak_avg_points'], pre_peak_ms=cfg['pre_peak_ms'],
                shift_min_s=cfg['shift_min_s'], n_samples=1000,
                seed=cfg.get('seed', 42), event_t0_s=event_t0_s,
            )
        else:
            # NNLS-consistent null on pre-train window using the single-pulse estimator
            null_amps = sample_null_amplitudes_consistent(
                yj, t, baseline_mask, tau_r, tau_d0,
                train_start=float(train_start), f0_window_s=cfg['f0_window_s'],
                pre_zoom_s=cfg['pre_zoom_s'], post_zoom_s=cfg['post_zoom_s'],
                robust=True, huber_delta=cfg['huber_delta'], irls_iters=cfg['irls_iters'],
                allow_shift=allow_shift, delta_max_s=cfg['delta_max_s'], delta_step_s=cfg['delta_step_s'],
                null_min_post_zoom_s=cfg['null_min_post_zoom_s'], null_sim_max_points=cfg['null_sim_max_points'],
                peak_window_ms=cfg['peak_window_ms'], peak_avg_points=cfg['peak_avg_points'], pre_peak_ms=cfg['pre_peak_ms'],
                shift_min_s=cfg['shift_min_s'], n_samples=1000,
                seed=cfg.get('seed', 42), event_t0_s=event_t0_s,
            )
            null_amps_nnls = np.asarray(null_amps, float)

        thr1, pfun = baseline_threshold_and_pval(null_amps, null_N, mode=eff_mode)

        # Pick amplitude series for p-values/classification
        if meas == 'SAVGOL':
            a_for_p = amp_sg_corr
        elif meas == 'RAW':
            a_for_p = amp_raw_corr
        else:
            a_for_p = amp_nn_corr

        a1 = float(a_for_p[0]) if a_for_p.size else np.nan
        p1 = float(pfun(a1)) if np.isfinite(a1) else np.nan
        thr_list.append(thr1); pval_list.append(p1)
        # Compute p-values for pulses 2 and 3 using the same baseline threshold as A1
        p2 = float(pfun(a_for_p[1])) if (a_for_p.size >= 2 and np.isfinite(a_for_p[1])) else np.nan
        p3 = float(pfun(a_for_p[2])) if (a_for_p.size >= 3 and np.isfinite(a_for_p[2])) else np.nan

        # Preserve per-trial amplitudes before optional floor so exports can
        # report both values even when amplitude_floor_to_noise=True.
        amp_raw_unfloored = np.asarray(amp_raw, float).copy()
        amp_raw_corr_unfloored = np.asarray(amp_raw_corr, float).copy()
        amp_sg_unfloored = np.asarray(amp_sg, float).copy()
        amp_sg_corr_unfloored = np.asarray(amp_sg_corr, float).copy()
        amp_nn_unfloored = np.asarray(amp_nn, float).copy()
        amp_nn_corr_unfloored = np.asarray(amp_nn_corr, float).copy()

        # Per-trial noise spread used by the threshold rule.
        noise_level = np.nan
        null_arr = np.asarray(null_amps, float)
        null_arr = null_arr[np.isfinite(null_arr)]
        null_nnls_arr = np.asarray(null_amps_nnls, float)
        null_nnls_arr = null_nnls_arr[np.isfinite(null_nnls_arr)]
        # Baseline null-amplitude summaries used to derive threshold statistics.
        # "Including zeros" uses all finite null amplitudes.
        # "Excluding zeros" removes exact/near-zero entries.
        baseline_null_mean_including_zero = np.nan
        baseline_null_median_including_zero = np.nan
        baseline_null_mean_excluding_zero = np.nan
        baseline_null_median_excluding_zero = np.nan
        if null_arr.size:
            baseline_null_mean_including_zero = float(np.nanmean(null_arr))
            baseline_null_median_including_zero = float(np.nanmedian(null_arr))
            nz_mask = np.abs(null_arr) > 1e-12
            if np.any(nz_mask):
                nz_vals = null_arr[nz_mask]
                baseline_null_mean_excluding_zero = float(np.nanmean(nz_vals))
                baseline_null_median_excluding_zero = float(np.nanmedian(nz_vals))
            if eff_mode == 'sd':
                noise_level = float(np.nanstd(null_arr))
            else:
                med0 = float(np.nanmedian(null_arr))
                mad0 = float(np.nanmedian(np.abs(null_arr - med0)))
                noise_level = float(1.4826 * mad0)

        # Floor corrected amplitudes only when the absolute (uncorrected) amplitude
        # is below the noise threshold.  Uncorrected amplitudes stay raw so the
        # user can detect floored events later by comparing to thr_shared.
        if cfg.get('amplitude_floor_to_noise', False):
            raw_below = amp_raw < thr1
            sg_below  = amp_sg  < thr1
            nn_below  = amp_nn  < thr1
            amp_raw_corr[raw_below] = np.maximum(amp_raw_corr[raw_below], thr1)
            amp_sg_corr[sg_below]   = np.maximum(amp_sg_corr[sg_below],  thr1)
            amp_nn_corr[nn_below]   = np.maximum(amp_nn_corr[nn_below],  thr1)

        per_trial.append({
            'trial_processed_index_1based': int(j + 1),
            'trial_input_col_0based': int(kept_trial_cols_0based[j]) if j < kept_trial_cols_0based.size else int(j),
            'trial_input_col_1based': int(kept_trial_cols_0based[j] + 1) if j < kept_trial_cols_0based.size else int(j + 1),
            'F0': float(F0[j]) if j < F0.size and np.isfinite(F0[j]) else np.nan,
            'amp_raw': amp_raw,
            'amp_raw_corr': amp_raw_corr,
            'amp_savgol': amp_sg,
            'amp_savgol_corr': amp_sg_corr,
            'amp_nnls': amp_nn,
            'amp_nnls_corr': amp_nn_corr,
            'amp_raw_unfloored': amp_raw_unfloored,
            'amp_raw_corr_unfloored': amp_raw_corr_unfloored,
            'amp_savgol_unfloored': amp_sg_unfloored,
            'amp_savgol_corr_unfloored': amp_sg_corr_unfloored,
            'amp_nnls_unfloored': amp_nn_unfloored,
            'amp_nnls_corr_unfloored': amp_nn_corr_unfloored,
            'ppr_raw': _norm(amp_raw),
            'ppr_savgol': _norm(amp_sg),
            'ppr_nnls': _norm(amp_nn),
            'ppr_raw_corr': _norm(amp_raw_corr),
            'ppr_savgol_corr': _norm(amp_sg_corr),
            'ppr_nnls_corr': _norm(amp_nn_corr),
            'a_coeff': a_t,
            'delta_s': d_t,
            'y_proc': yj,
            'yhat': yhat_t,
            'components': comp_t,
            'thr_shared': thr1,
            'noise_level': noise_level,
            'noise_mode': eff_mode,
            'null_amps_nnls': null_nnls_arr,
            'baseline_null_mean_including_zero': baseline_null_mean_including_zero,
            'baseline_null_median_including_zero': baseline_null_median_including_zero,
            'baseline_null_mean_excluding_zero': baseline_null_mean_excluding_zero,
            'baseline_null_median_excluding_zero': baseline_null_median_excluding_zero,
            'pval_amp1': p1,
            'pval_amp2': p2,
            'pval_amp3': p3,
        })

        # Optional: per-trial plot
        if want_plot and plot_trials:
            zmask_t, z0, z1 = time_zoom_mask(
                t, float(train_start), float(isi), int(n_pulses), cfg['pre_zoom_s'], cfg['post_zoom_s']
            )
            tz = t[zmask_t]
            base_mask = (t < float(train_start))

            # Layout: main train panel on top; optional residuals directly
            # underneath; optional baseline at the bottom. Residuals share the
            # time axis with the train; baseline is not time-aligned.
            want_resid_row = bool(plot_residuals)
            want_base_row = bool(baseline_figs)
            if want_resid_row or want_base_row:
                n_rows = 1 + int(want_resid_row) + int(want_base_row)
                if n_rows == 3:
                    ratios = [2.4, 1.3, 1.6]
                elif want_resid_row and not want_base_row:
                    ratios = [2.4, 1.3]
                else:
                    ratios = [2.4, 1.6]
                fig_t = plt.figure(figsize=(11, 9.0))
                gs_t = fig_t.add_gridspec(n_rows, 1, height_ratios=ratios, hspace=0.25)
                ax_train = fig_t.add_subplot(gs_t[0, 0])
                row = 1
                ax_resid = None
                ax_base = None
                if want_resid_row:
                    ax_resid = fig_t.add_subplot(gs_t[row, 0], sharex=ax_train)
                    row += 1
                if want_base_row:
                    ax_base = fig_t.add_subplot(gs_t[row, 0])  # no sharex (not time-aligned)
            else:
                fig_t, ax_train = plt.subplots(figsize=(10, 4.8))
                ax_resid = None
                ax_base = None

            # Train panel
            for st in stim_times:
                ax_train.axvline(st, color='k', linestyle=':', linewidth=0.8, alpha=0.6)
            if 'raw' in traces:
                ax_train.plot(tz, yj[zmask_t], label='raw', color='0.6')
            if 'savgol' in traces and yj_sg is not None:
                ax_train.plot(tz, yj_sg[zmask_t], label='savgol', color='tab:green')
            if 'nnls' in traces:
                ax_train.plot(tz, yhat_t[zmask_t], label='nnls model', color='tab:blue')
            if show_decay and 'nnls' in traces and comp_t is not None:
                cumulative_t = np.zeros_like(yj)
                for p in range(len(stim_times)):
                    if p >= len(comp_t):
                        continue
                    cumulative_t = cumulative_t + comp_t[p]
                    ax_train.plot(
                        tz,
                        cumulative_t[zmask_t],
                        color='tab:orange',
                        linestyle='--',
                        linewidth=1.0,
                        alpha=0.6 - p * 0.04,
                    )
            # Peak markers and residual-at-peak triangles for this trial
            if plot_peaks_details:
                if meas == 'SAVGOL' and (yj_sg is not None):
                    y_for_peaks_t = yj_sg
                    amp_corr_for_dots = amp_sg_corr_unfloored
                elif meas == 'RAW':
                    y_for_peaks_t = yj
                    amp_corr_for_dots = amp_raw_corr_unfloored
                else:
                    y_for_peaks_t = yhat_t
                    amp_corr_for_dots = amp_nn_corr_unfloored

                # Precompute per-pulse cumulative baseline from previous pulses (NNLS comps)
                baseline_prev_only_t = []
                if comp_t is not None:
                    cum_t = np.zeros_like(yj)
                    for pp in range(len(stim_times)):
                        baseline_prev_only_t.append(cum_t.copy())
                        if pp < len(comp_t):
                            cum_t = cum_t + comp_t[pp]

                peak_ts_t, peak_vals_t, resid_vals_t = [], [], []
                for pp, stp in enumerate(stim_times):
                    # Find peak directly on the NNLS model so the red dot sits on the blue line
                    tp, vp = pick_peak_on_series(t, y_for_peaks_t, float(stp), win_ms, pre_ms)
                    peak_ts_t.append(float(tp))
                    peak_vals_t.append(float(vp))
                    try:
                        i0 = int(np.argmin(np.abs(t - tp)))
                        base_prev = baseline_prev_only_t[pp][i0] if baseline_prev_only_t else 0.0
                        resid_vals_t.append(float(base_prev))
                    except Exception:
                        resid_vals_t.append(np.nan)
                # Color A1 dot by success/failure status
                try:
                    a1_val = float(amp_corr_for_dots[0]) if len(amp_corr_for_dots) else np.nan
                    a1_status = 'success' if (np.isfinite(a1_val) and np.isfinite(thr1) and a1_val > thr1) else 'failure'
                    a1_color = 'tab:green' if a1_status == 'success' else 'red'
                    # A1 dot with status color
                    ax_train.scatter([peak_ts_t[0]], [peak_vals_t[0]], s=80, color=a1_color, edgecolors='white', linewidths=0.9, zorder=7)
                    # Remaining dots in red
                    if len(peak_ts_t) > 1:
                        ax_train.scatter(peak_ts_t[1:], peak_vals_t[1:], s=60, color='red', edgecolors='white', linewidths=0.9, zorder=6)
                except Exception:
                    ax_train.scatter(peak_ts_t, peak_vals_t, s=60, color='red', edgecolors='white', linewidths=0.9, zorder=6)
                try:
                    ax_train.scatter(peak_ts_t, resid_vals_t, s=55, marker='v', facecolors='white', edgecolors='tab:red', linewidths=1.0, zorder=5)
                except Exception:
                    pass
            # Failure threshold line (thin red dotted) + A1 annotation
            if np.isfinite(thr1):
                ax_train.axhline(thr1, color='red', linestyle=':', linewidth=0.8)
                try:
                    a1_v = float(amp_corr_for_dots[0]) if plot_peaks_details and len(amp_corr_for_dots) else np.nan
                    if np.isfinite(a1_v):
                        a1_s = 'PASS' if a1_v > thr1 else 'FAIL'
                        ax_train.text(
                            0.01, 0.97,
                            f'A1={a1_v:.4f}  thr={thr1:.4f}  [{a1_s}]',
                            transform=ax_train.transAxes, fontsize=8,
                            va='top', ha='left',
                            color='tab:green' if a1_s == 'PASS' else 'red',
                            fontfamily='monospace',
                        )
                except Exception:
                    pass
            ax_train.set_xlim(z0, z1)
            ax_train.set_xlabel('Time (s)')
            ax_train.set_ylabel('ΔF/F0' if use_dff else 'ΔF')
            ax_train.legend(loc='upper right', frameon=False)
            ax_train.set_title(f'Trial {j+1}: train window')
            _trim_spines(ax_train)

            # Residuals panel directly underneath the main panel
            if ax_resid is not None:
                try:
                    model_t = (yj_sg if (meas == 'SAVGOL' and yj_sg is not None) else yhat_t)
                    resid_t = yj - model_t
                    ax_resid.plot(tz, resid_t[zmask_t], color='tab:purple', lw=1.2, label='residual (trial − model)')
                    ax_resid.axvline(float(train_start), color='k', ls=':', lw=0.8, alpha=0.6)
                    ax_resid.set_xlim(z0, z1)
                    ax_resid.set_ylabel('ΔF/F0' if use_dff else 'ΔF')
                    ax_resid.set_title('Residuals in train window')
                    ax_resid.legend(loc='upper right', frameon=False, fontsize=8)
                    # Inset histogram of residuals (zoom window) with fixed bins and Gaussian fit
                    try:
                        ax_in_r = ax_resid.inset_axes([0.65, 0.55, 0.33, 0.4])
                        rdata = np.asarray(resid_t[zmask_t], float)
                        rdata = rdata[np.isfinite(rdata)]
                        rbase = None
                        try:
                            rbase = np.asarray(resid_t[base_mask], float)
                            rbase = rbase[np.isfinite(rbase)]
                        except Exception:
                            rbase = None
                        if rdata.size:
                            bin_w = (0.01 if use_dff else 10.0)
                            lo = float(np.nanmin(rdata))
                            hi = float(np.nanmax(rdata))
                            if not np.isfinite(lo):
                                lo = 0.0
                            if not np.isfinite(hi) or hi <= lo:
                                hi = lo + bin_w
                            edges = np.arange(lo, hi + bin_w, bin_w)
                            counts, edges = np.histogram(rdata, bins=edges)
                            widths = np.diff(edges)
                            ax_in_r.bar(edges[:-1], counts, width=widths, align='edge',
                                        color='#6b4fa3', alpha=0.9, edgecolor='none')
                            # Baseline-only Gaussian fit, scaled to train residual count
                            if rbase is not None and rbase.size:
                                try:
                                    mu = float(np.nanmean(rbase))
                                    sigma = float(np.nanstd(rbase))
                                except Exception:
                                    mu, sigma = float('nan'), float('nan')
                                if np.isfinite(sigma) and sigma > 0:
                                    centers = edges[:-1] + 0.5 * widths
                                    pdf_centers = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(
                                        -0.5 * ((centers - mu) / sigma) ** 2
                                    )
                                    N = rdata.size
                                    expected = N * bin_w * pdf_centers
                                    peak_expected = float(np.nanmax(expected)) if expected.size else 0.0
                                    cutoff = max(0.5, 0.01 * peak_expected)
                                    base = np.where(expected >= cutoff,
                                                    np.minimum(counts.astype(float), expected),
                                                    0.0)
                                    excess = counts.astype(float) - base
                                    if np.any(excess > 0):
                                        ax_in_r.bar(edges[:-1], excess, width=widths, align='edge',
                                                    bottom=base, color='tab:red', alpha=0.85, edgecolor='none')
                                    x = np.linspace(edges[0], edges[-1], 400)
                                    pdf = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(
                                        -0.5 * ((x - mu) / sigma) ** 2
                                    )
                                    y = N * bin_w * pdf
                                    ax_in_r.plot(x, y, color='#26457a', linewidth=1.4, label='baseline fit')
                        ax_in_r.set_title('residual', fontsize=8)
                        ax_in_r.tick_params(labelsize=7)
                    except Exception:
                        pass
                except Exception:
                    pass
                _trim_spines(ax_resid)

            # Baseline panel (pre-train) at the bottom (if requested)
            if ax_base is not None:
                base_mask = (t < float(train_start))
                tb = t[base_mask]
                use_savgol_baseline = False
                if tb.size:
                    # Highlight the f0 baseline window used for null sampling
                    try:
                        baseline_start = tb[0]
                        baseline_end = tb[-1]
                        null_start = max(baseline_start, float(train_start) - cfg['f0_window_s'])
                        # Shade from null_start up to first stimulus time
                        ax_base.axvspan(null_start, float(train_start), facecolor='#ffd500', alpha=0.18, zorder=0.1)
                        # Vertical dotted line at first stimulus time
                        ax_base.axvline(float(train_start), color='k', ls=':', lw=0.8, alpha=0.6)
                        # Ensure the x-range reaches the dotted line
                        ax_base.set_xlim(baseline_start, float(train_start))
                    except Exception:
                        pass
                    use_savgol_baseline = (
                        (meas == 'NNLS')
                        and (failm != meas)
                        and (yj_sg is not None)
                    )
                    if 'raw' in traces:
                        ax_base.plot(tb, yj[base_mask], color='0.4', linewidth=1.0, label='baseline')
                    if use_savgol_baseline:
                        ax_base.plot(
                            tb,
                            yj_sg[base_mask],
                            color='red',
                            linewidth=1.1,
                            label='savgol',
                        )
                    else:
                        # Overlay a subset of null-fit events in red
                        baseline_start = tb[0]; baseline_end = tb[-1]
                        null_start = max(baseline_start, float(train_start) - cfg['f0_window_s'])
                        null_end = min(baseline_end, float(train_start))
                        st_min = null_start + cfg['pre_zoom_s']
                        st_max = null_end - cfg['null_min_post_zoom_s']
                        cand_mask = (t >= st_min) & (t <= st_max)
                        starts_full = t[cand_mask]
                        if starts_full.size:
                            events = []  # (amp, start, shift)
                            for stcand in starts_full:
                                avail_post = min(cfg['post_zoom_s'], float(train_start) - stcand - 1e-6, null_end - stcand)
                                if avail_post < cfg['null_min_post_zoom_s']:
                                    continue
                                a_hat_b, d_hat_b = _fit_single_pulse_amp(
                                    yj, t, float(stcand), tau_r, tau_d0,
                                    pre_zoom_s=cfg['pre_zoom_s'], post_zoom_s=avail_post,
                                    robust=True, huber_delta=cfg['huber_delta'], irls_iters=cfg['irls_iters'],
                                    allow_shift=True, delta_max_s=cfg['delta_max_s'], delta_step_s=cfg['delta_step_s'],
                                    shift_min_s=cfg['shift_min_s'], event_t0_s=event_t0_s,
                                )
                                events.append((float(a_hat_b), float(stcand), float(d_hat_b)))
                            events = [e for e in events if np.isfinite(e[0]) and e[0] > 0]
                            events.sort(key=lambda e: e[0], reverse=True)
                            draw_n = min(20, len(events))
                            for a_hat_b, stcand, d_hat_b in events[:draw_n]:
                                anchor_b = float(stcand) + event_t0_s
                                y_evt_b = a_hat_b * _KERNEL_FUN(tb - (anchor_b + d_hat_b), tau_r, tau_d0)
                                ax_base.plot(tb, y_evt_b, color='red', alpha=0.5, linewidth=1.0)
                    # Inset histogram of null amplitudes with threshold and fitted noise curve
                    try:
                        ax_in = ax_base.inset_axes([0.65, 0.55, 0.33, 0.4])
                        data = np.asarray(null_amps, float)
                        data = data[np.isfinite(data)]
                        if data.size:
                            # Fixed bin width based on measurement scale
                            bin_w = (0.01 if use_dff else 10.0)
                            lo = 0.0  # null amplitudes are non‑negative
                            hi = float(np.nanmax(data)) if np.isfinite(np.nanmax(data)) else 0.0
                            if hi <= lo:
                                hi = lo + bin_w
                            edges = np.arange(lo, hi + bin_w, bin_w)
                            # Draw histogram and capture bin counts
                            n_hist, bins_hist, _ = ax_in.hist(
                                data, bins=edges, color='#c9d4e8', edgecolor='#4f6aa3'
                            )
                            # Choose noise model overlay based on baseline measurement method:
                            #   - NNLS => half‑normal (cut Gaussian, non‑negative)
                            #   - RAW/SAVGOL => Gaussian
                            if failm == 'NNLS':
                                # Zero‑inflated half‑normal: mass at 0 for the clipped negative side
                                # plus a continuous half‑normal on x>0.
                                N = data.size
                                zero_tol = max(1e-12, 0.5 * bin_w)
                                is_zero = data <= zero_tol
                                n_zero = int(np.sum(is_zero))
                                p0 = float(n_zero) / float(N) if N else 0.0
                                pos = data[~is_zero]
                                # MLE for half‑normal sigma on strictly positive samples
                                try:
                                    sigma = float(np.sqrt(np.mean(np.square(np.clip(pos, 0.0, None))))) if pos.size else float('nan')
                                except Exception:
                                    sigma = float('nan')
                                x0, x1 = ax_in.get_xlim()
                                x = np.linspace(x0, x1, 400)
                                if np.isfinite(sigma) and sigma > 0:
                                    x_clip = np.clip(x, 0.0, None)
                                    pdf = (np.sqrt(2.0) / (sigma * np.sqrt(np.pi))) * np.exp(-(x_clip**2) / (2.0 * sigma * sigma))
                                    pdf = np.where(x >= 0.0, pdf, 0.0)
                                    y = (N * (1.0 - p0) * bin_w) * pdf  # scale continuous part
                                    ax_in.plot(x, y, color='#26457a', linewidth=1.4, label='fit')
                                # Axis scaling: ignore the 0-bin height; use next highest bin
                                try:
                                    max_other = float(np.nanmax(n_hist[1:])) if n_hist.size > 1 else 0.0
                                except Exception:
                                    max_other = 0.0
                                ymax = max(1.0, max_other * 1.15)
                                ax_in.set_ylim(0.0, ymax)
                                # Annotate the zero-bin count instead of scaling to it
                                if n_zero > 0:
                                    ax_in.text(
                                        0.02 * (x1 - x0) + x0,
                                        ymax * 0.92,
                                        f"0-bin: {n_zero}",
                                        fontsize=7,
                                        ha='left', va='top', color='#26457a',
                                        bbox=dict(boxstyle='round,pad=0.12', facecolor='white', alpha=0.7, lw=0)
                                    )
                            else:
                                # Gaussian fit
                                try:
                                    mu = float(np.nanmean(data))
                                    sig = float(np.nanstd(data))
                                except Exception:
                                    mu, sig = float('nan'), float('nan')
                                if np.isfinite(sig) and sig > 0:
                                    x0, x1 = ax_in.get_xlim()
                                    x = np.linspace(x0, x1, 400)
                                    pdf = (1.0 / (np.sqrt(2.0 * np.pi) * sig)) * np.exp(-0.5 * ((x - mu) / sig) ** 2)
                                    N = data.size
                                    y = N * bin_w * pdf
                                    ax_in.plot(x, y, color='#26457a', linewidth=1.4, label='fit')
                            if np.isfinite(thr1):
                                ax_in.axvline(thr1, color='red', linestyle='--', linewidth=0.9)
                        ax_in.set_title('noise', fontsize=8)
                        ax_in.tick_params(labelsize=7)
                    except Exception:
                        pass
                ax_base.set_xlabel('Time (s)')
                ax_base.set_ylabel('ΔF/F0' if use_dff else 'ΔF')
                if use_savgol_baseline:
                    ax_base.set_title('Baseline window (savgol)')
                else:
                    ax_base.set_title('Baseline window + null-fit events')
                _trim_spines(ax_base)

            # Match Y limits across comparable panels (exclude residuals)
            try:
                ylims = [ax_train.get_ylim()]
                if ax_base is not None:
                    ylims.append(ax_base.get_ylim())
                ymin = min([y[0] for y in ylims])
                ymax = max([y[1] for y in ylims])
                ax_train.set_ylim(ymin, ymax)
                if ax_base is not None:
                    ax_base.set_ylim(ymin, ymax)
            except Exception:
                pass

            # Show trial figures immediately after they are created
            try:
                plt.show(block=False); plt.pause(0.01)
            except Exception:
                pass
            figures_trials.append(fig_t)

    # Floor average amplitudes to noise threshold if option enabled (prevents div by near-zero in PPR)
    # Keep unfloored copies for amplitude reporting; only use floored values for PPR.
    amp_raw_avg_unfloored = np.asarray(amp_raw_avg, float).copy()
    amp_raw_corr_avg_unfloored = np.asarray(amp_raw_corr_avg, float).copy()
    amp_sg_avg_unfloored = np.asarray(amp_sg_avg, float).copy()
    amp_sg_corr_avg_unfloored = np.asarray(amp_sg_corr_avg, float).copy()
    amp_nnls_avg_unfloored = np.asarray(amp_nnls_avg, float).copy()
    amp_nnls_corr_avg_unfloored = np.asarray(amp_nnls_corr_avg, float).copy()
    median_threshold = np.nan
    # Resolve average-specific options (fall back to per-trial settings if not specified)
    _avg_floor_opt = cfg.get('average_amplitude_floor_to_noise')
    avg_floor_enabled = bool(_avg_floor_opt) if _avg_floor_opt is not None else bool(cfg.get('amplitude_floor_to_noise', False))
    _avg_null_N_opt = cfg.get('average_null_N')
    avg_null_N = float(_avg_null_N_opt) if _avg_null_N_opt is not None else null_N
    if avg_floor_enabled and len(thr_list) > 0:
        # Recompute per-trial thresholds with average_null_N if it differs from null_N
        if avg_null_N != null_N:
            avg_thr_list = []
            for rt in per_trial:
                null_arr_t = np.asarray(rt.get('null_amps_nnls', []), float)
                null_arr_t = null_arr_t[np.isfinite(null_arr_t)]
                if null_arr_t.size:
                    thr_t, _ = baseline_threshold_and_pval(null_arr_t, avg_null_N, mode=rt.get('noise_mode', 'mad'))
                    avg_thr_list.append(thr_t)
            if avg_thr_list:
                median_threshold = float(np.median(avg_thr_list))
            progress_print(f"[floor-avg] Using average_null_N={avg_null_N:.2f} (per-trial null_N={null_N:.2f})")
            progress_print(f"[floor-avg] Recomputed per-trial thresholds: {[f'{t:.4f}' for t in avg_thr_list]}")
        else:
            median_threshold = float(np.median(thr_list))
        progress_print(f"[floor-avg] Amplitude floor (median threshold): {median_threshold:.6f}")
        progress_print(f"[floor-avg] Original per-trial thresholds: {[f'{t:.4f}' for t in thr_list]}")
        # Floor corrected amplitudes only when the absolute (uncorrected)
        # amplitude is below the median noise threshold.
        raw_avg_below  = amp_raw_avg  < median_threshold
        sg_avg_below   = amp_sg_avg   < median_threshold
        nnls_avg_below = amp_nnls_avg < median_threshold
        amp_raw_corr_avg_floored = amp_raw_corr_avg.copy()
        amp_raw_corr_avg_floored[raw_avg_below] = np.maximum(amp_raw_corr_avg_floored[raw_avg_below], median_threshold)
        amp_sg_corr_avg_floored = amp_sg_corr_avg.copy()
        amp_sg_corr_avg_floored[sg_avg_below] = np.maximum(amp_sg_corr_avg_floored[sg_avg_below], median_threshold)
        amp_nnls_corr_avg_floored = amp_nnls_corr_avg.copy()
        amp_nnls_corr_avg_floored[nnls_avg_below] = np.maximum(amp_nnls_corr_avg_floored[nnls_avg_below], median_threshold)
        # Recalculate PPR with floored corrected amplitudes
        ppr_nnls_avg = _norm(amp_nnls_avg)
        ppr_nnls_corr_avg = _norm(amp_nnls_corr_avg_floored)

    figure_residual_buildup = None
    if plot_residual_buildup and per_trial:
        try:
            comps_trials = [r.get('components') for r in per_trial if r.get('components') is not None]
            if comps_trials:
                n_p = len(comps_trials[0]) if comps_trials[0] else 0
                comp_avg = []
                for p in range(n_p):
                    comp_stack = []
                    for comp_list in comps_trials:
                        if comp_list and len(comp_list) > p:
                            comp_stack.append(np.asarray(comp_list[p], float))
                    if comp_stack:
                        comp_avg.append(np.nanmean(np.vstack(comp_stack), axis=0))
                if len(comp_avg) == n_p:
                    cum_before_last = np.sum(comp_avg[:-1], axis=0) if n_p > 1 else np.zeros_like(y_avg)
                    resid_before_last = y_avg - cum_before_last
                    resid_full = y_avg - yhat_avg if yhat_avg.size == y_avg.size else None
                    zmask = (t >= float(train_start) - cfg['pre_zoom_s']) & (
                        t <= float(train_start) + float(isi) * int(n_pulses) + cfg['post_zoom_s']
                    )
                    figure_residual_buildup, axes_res = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
                    axes_res[0].plot(t[zmask], y_avg[zmask], color='black', label='avg trace')
                    axes_res[0].plot(t[zmask], cum_before_last[zmask], color='tab:orange', label='sum first N-1')
                    if yhat_avg.size == y_avg.size:
                        axes_res[0].plot(t[zmask], yhat_avg[zmask], color='tab:blue', label='nnls model')
                    axes_res[0].set_ylabel("ΔF/F0")
                    axes_res[0].set_title("Average trace vs cumulative fit")
                    axes_res[0].legend(loc='upper right')
                    axes_res[1].plot(t[zmask], resid_before_last[zmask], color='tab:red', label='residual after N-1')
                    if resid_full is not None:
                        axes_res[1].plot(t[zmask], resid_full[zmask], color='tab:purple', label='full residual')
                    axes_res[1].axhline(0.0, color='0.6', linestyle=':')
                    axes_res[1].set_xlabel("Time (s)")
                    axes_res[1].set_ylabel("Residual")
                    axes_res[1].set_title("Residual buildup check")
                    axes_res[1].legend(loc='upper right')
                    plt.tight_layout()
        except Exception:
            figure_residual_buildup = None

    figure_nnls_residual = None
    if plot_nnls_residual:
        try:
            if t.size and y_avg.size and yhat_avg.size == y_avg.size:
                resid = y_avg - yhat_avg
                figure_nnls_residual, ax_resid = plt.subplots(figsize=(8, 3))
                ax_resid.plot(t, resid, color='tab:purple', linewidth=1.2)
                ax_resid.axhline(0.0, color='0.6', linestyle=':')
                ax_resid.set_title("NNLS residual (avg - model)")
                ax_resid.set_xlabel("Time (s)")
                ax_resid.set_ylabel("Residual ΔF/F0")
                try:
                    handles, labels = ax_resid.get_legend_handles_labels()
                    if handles:
                        ax_resid.legend(loc='upper right')
                except Exception:
                    pass
                plt.tight_layout()
        except Exception:
            figure_nnls_residual = None

    # Parameter evolution figure: show how amplitude fractions evolve across events
    # For tri-exp: frac_fast, frac_slow, frac_superslow (sum to 1)
    # For bi-exp: frac_fast, frac_slow (sum to 1)
    # Also show amplitudes (raw + corrected) and PPR
    figure_param_evolution = None
    if plot_param_evolution and variant_info_avg is not None:
        try:
            is_tri = (event_model == 'iglusnfr_tri')
            dominant_ratios = variant_info_avg.get('dominant_template_ratio', [])
            template_variant_sums = variant_info_avg.get('template_variant_sums', None)
            variant_ratios = variant_info_avg.get('variant_ratios', None)
            hard_select = bool(variant_info_avg.get('hard_select', False))

            use_soft_weights = (
                (template_variant_sums is not None)
                and (variant_ratios is not None)
                and (not hard_select)
                and (len(dominant_ratios) == n_pulses)
                and (np.asarray(template_variant_sums).shape[0] == n_pulses)
                and (len(variant_ratios) == np.asarray(template_variant_sums).shape[1])
            )

            if dominant_ratios is not None and len(dominant_ratios) == n_pulses:
                event_indices = np.arange(1, n_pulses + 1)
                
                # Create 2x2 figure
                fig_pe, axes = plt.subplots(2, 2, figsize=(12, 8))
                ax1, ax2, ax3, ax4 = axes.flatten()
                
                if is_tri:
                    # Tri-exponential: extract (frac_slow, frac_superslow_max) per event
                    frac_slow_arr = np.zeros(n_pulses)
                    frac_superslow_arr = np.zeros(n_pulses)
                    frac_fast_arr = np.zeros(n_pulses)

                    for i in range(n_pulses):
                        if use_soft_weights:
                            weights = np.asarray(template_variant_sums[i], float)
                            wsum = float(np.sum(weights))
                            if wsum > 0:
                                wnorm = weights / wsum
                                for w, r in zip(wnorm, variant_ratios):
                                    if isinstance(r, (list, tuple)) and len(r) >= 2:
                                        frac_slow_base = float(r[0])
                                        frac_superslow_max = float(r[1])
                                        ramp = float(i) / float(n_pulses - 1) if n_pulses > 1 else 0.0
                                        frac_superslow = frac_superslow_max * ramp
                                        frac_slow = frac_slow_base
                                        if frac_slow + frac_superslow > 1.0:
                                            frac_slow = max(0.0, 1.0 - frac_superslow)
                                        frac_fast = max(0.0, 1.0 - frac_slow - frac_superslow)
                                    else:
                                        frac_slow = float(r) if np.isscalar(r) else 0.5
                                        frac_superslow = 0.0
                                        frac_fast = 1.0 - frac_slow
                                    frac_fast_arr[i] += w * frac_fast
                                    frac_slow_arr[i] += w * frac_slow
                                    frac_superslow_arr[i] += w * frac_superslow
                            else:
                                r = dominant_ratios[i]
                                if isinstance(r, (list, tuple)) and len(r) >= 2:
                                    frac_slow_base = float(r[0])
                                    frac_superslow_max = float(r[1])
                                    ramp = float(i) / float(n_pulses - 1) if n_pulses > 1 else 0.0
                                    frac_superslow = frac_superslow_max * ramp
                                    frac_slow = frac_slow_base
                                    if frac_slow + frac_superslow > 1.0:
                                        frac_slow = max(0.0, 1.0 - frac_superslow)
                                    frac_fast = max(0.0, 1.0 - frac_slow - frac_superslow)
                                else:
                                    frac_slow = float(r) if np.isscalar(r) else 0.5
                                    frac_superslow = 0.0
                                    frac_fast = 1.0 - frac_slow
                                frac_fast_arr[i] = frac_fast
                                frac_slow_arr[i] = frac_slow
                                frac_superslow_arr[i] = frac_superslow
                        else:
                            r = dominant_ratios[i]
                            if isinstance(r, (list, tuple)) and len(r) >= 2:
                                frac_slow_base = float(r[0])
                                frac_superslow_max = float(r[1])
                                ramp = float(i) / float(n_pulses - 1) if n_pulses > 1 else 0.0
                                frac_superslow = frac_superslow_max * ramp
                                frac_slow = frac_slow_base
                                if frac_slow + frac_superslow > 1.0:
                                    frac_slow = max(0.0, 1.0 - frac_superslow)
                                frac_fast = max(0.0, 1.0 - frac_slow - frac_superslow)
                            else:
                                frac_slow = float(r) if np.isscalar(r) else 0.5
                                frac_superslow = 0.0
                                frac_fast = 1.0 - frac_slow

                            frac_fast_arr[i] = frac_fast
                            frac_slow_arr[i] = frac_slow
                            frac_superslow_arr[i] = frac_superslow
                    
                    # Panel 1: stacked fractions
                    ax1.stackplot(event_indices, frac_fast_arr, frac_slow_arr, frac_superslow_arr,
                                  labels=['frac_fast', 'frac_slow', 'frac_superslow'],
                                  colors=['tab:blue', 'tab:orange', 'tab:green'], alpha=0.7)
                    ax1.plot(event_indices, frac_fast_arr, 'o-', color='tab:blue', markersize=5, lw=1.5)
                    ax1.plot(event_indices, frac_fast_arr + frac_slow_arr, 's-', color='tab:orange', markersize=5, lw=1.5)
                    ax1.set_xlabel('Event #')
                    ax1.set_ylabel('Cumulative fraction')
                    ax1.set_title('Component fractions (stacked)')
                    ax1.set_ylim(0, 1.05)
                    ax1.set_xticks(event_indices)
                    ax1.legend(loc='upper left', fontsize=8)
                    ax1.grid(True, alpha=0.3)
                    _trim_spines(ax1)
                    
                    # Panel 2: individual fraction lines
                    ax2.plot(event_indices, frac_fast_arr, 'o-', color='tab:blue', label='frac_fast', markersize=6, lw=1.5)
                    ax2.plot(event_indices, frac_slow_arr, 's-', color='tab:orange', label='frac_slow', markersize=6, lw=1.5)
                    ax2.plot(event_indices, frac_superslow_arr, '^-', color='tab:green', label='frac_superslow', markersize=6, lw=1.5)
                    ax2.set_xlabel('Event #')
                    ax2.set_ylabel('Fraction')
                    ax2.set_title('Individual component fractions')
                    ax2.set_ylim(0, 1.05)
                    ax2.set_xticks(event_indices)
                    ax2.legend(loc='best', fontsize=8)
                    ax2.grid(True, alpha=0.3)
                    _trim_spines(ax2)
                    
                else:
                    # Bi-exponential: frac_slow per event
                    frac_slow_arr = np.zeros(n_pulses)
                    for i in range(n_pulses):
                        if use_soft_weights:
                            weights = np.asarray(template_variant_sums[i], float)
                            wsum = float(np.sum(weights))
                            if wsum > 0:
                                wnorm = weights / wsum
                                for w, r in zip(wnorm, variant_ratios):
                                    if isinstance(r, (list, tuple)):
                                        frac_slow = float(r[0]) if len(r) > 0 else 0.5
                                    else:
                                        frac_slow = float(r)
                                    frac_slow_arr[i] += w * frac_slow
                            else:
                                r = dominant_ratios[i]
                                if isinstance(r, (list, tuple)):
                                    frac_slow_arr[i] = float(r[0]) if len(r) > 0 else 0.5
                                else:
                                    frac_slow_arr[i] = float(r)
                        else:
                            r = dominant_ratios[i]
                            if isinstance(r, (list, tuple)):
                                frac_slow_arr[i] = float(r[0]) if len(r) > 0 else 0.5
                            else:
                                frac_slow_arr[i] = float(r)
                    frac_fast_arr = 1.0 - frac_slow_arr
                    
                    # Panel 1: stacked fractions
                    ax1.stackplot(event_indices, frac_fast_arr, frac_slow_arr,
                                  labels=['frac_fast', 'frac_slow'],
                                  colors=['tab:blue', 'tab:orange'], alpha=0.7)
                    ax1.plot(event_indices, frac_fast_arr, 'o-', color='tab:blue', markersize=5, lw=1.5)
                    ax1.set_xlabel('Event #')
                    ax1.set_ylabel('Cumulative fraction')
                    ax1.set_title('Component fractions (stacked)')
                    ax1.set_ylim(0, 1.05)
                    ax1.set_xticks(event_indices)
                    ax1.legend(loc='upper left', fontsize=8)
                    ax1.grid(True, alpha=0.3)
                    _trim_spines(ax1)
                    
                    # Panel 2: individual fraction lines
                    ax2.plot(event_indices, frac_fast_arr, 'o-', color='tab:blue', label='frac_fast', markersize=6, lw=1.5)
                    ax2.plot(event_indices, frac_slow_arr, 's-', color='tab:orange', label='frac_slow', markersize=6, lw=1.5)
                    ax2.set_xlabel('Event #')
                    ax2.set_ylabel('Fraction')
                    ax2.set_title('Individual component fractions')
                    ax2.set_ylim(0, 1.05)
                    ax2.set_xticks(event_indices)
                    ax2.legend(loc='best', fontsize=8)
                    ax2.grid(True, alpha=0.3)
                    _trim_spines(ax2)
                
                # Panel 3: Amplitudes (raw and corrected overlayed)
                ax3.plot(event_indices, amp_nnls_avg[:n_pulses], 'o-', color='tab:gray', 
                         label='NNLS raw', markersize=6, lw=1.5, alpha=0.7)
                ax3.plot(event_indices, amp_nnls_corr_avg[:n_pulses], 's-', color='tab:red', 
                         label='NNLS corrected', markersize=6, lw=2)
                if np.isfinite(median_threshold):
                    ax3.axhline(median_threshold, color='tab:orange', linestyle='--', lw=1.5, alpha=0.8,
                                label=f'PPR floor ({median_threshold:.3f})')
                ax3.set_xlabel('Event #')
                ax3.set_ylabel('Amplitude (ΔF/F₀)')
                ax3.set_title('NNLS Amplitudes')
                ax3.set_ylim(bottom=0.0)
                ax3.set_xticks(event_indices)
                ax3.legend(loc='best', fontsize=8)
                ax3.grid(True, alpha=0.3)
                _trim_spines(ax3)
                
                # Panel 4: PPR (raw + corrected)
                ax4.plot(event_indices, ppr_nnls_avg[:n_pulses], 'o-', color='tab:gray',
                         markersize=5, lw=1.4, alpha=0.7, label='PPR (raw)')
                ax4.plot(event_indices, ppr_nnls_corr_avg[:n_pulses], 'o-', color='tab:purple',
                         markersize=6, lw=2, label='PPR (corrected)')
                ax4.axhline(1.0, color='gray', linestyle='--', lw=1, alpha=0.7)
                ax4.set_xlabel('Event #')
                ax4.set_ylabel('PPR (normalized to A₁)')
                ax4.set_title('Paired-Pulse Ratio')
                ax4.set_ylim(bottom=0.0)
                ax4.set_xticks(event_indices)
                ax4.legend(loc='best', fontsize=8)
                ax4.grid(True, alpha=0.3)
                _trim_spines(ax4)
                
                model_label = 'tri-exponential' if is_tri else 'bi-exponential'
                fig_pe.suptitle(f'Parameter evolution ({model_label})', fontsize=12, fontweight='bold')
                plt.tight_layout()
                figure_param_evolution = fig_pe
                
        except Exception as e:
            progress_print(f"[warning] Failed to create param evolution figure: {e}")
            figure_param_evolution = None

    model_time_s = np.asarray(t, float)
    model_yhat_avg = np.asarray(yhat_avg, float)
    model_oversample = max(1, int(cfg.get('recut_oversample', 1)))
    if (
        model_oversample > 1
        and t.size > 1
        and variant_info_avg is None
        and np.asarray(X_avg).ndim == 2
        and np.asarray(X_avg).shape[1] == len(stim_times)
    ):
        model_time_s = np.linspace(
            float(t[0]),
            float(t[-1]),
            (t.size - 1) * model_oversample + 1,
        )
        model_yhat_avg = np.zeros_like(model_time_s)
        for pulse_idx, stim_time in enumerate(stim_times):
            anchor = float(stim_time) + float(event_t0_s) + float(d_avg[pulse_idx])
            kernel_dense = _KERNEL_FUN(
                model_time_s - anchor,
                float(tau_r),
                float(tau_d_vec[pulse_idx]),
            )
            # Some kernels normalize on their evaluation grid. Match each dense
            # kernel to the original NNLS design column so oversampling changes
            # display resolution, not the fitted amplitude.
            kernel_at_samples = np.interp(t, model_time_s, kernel_dense)
            design_column = np.asarray(X_avg[:, pulse_idx], float)
            denom = float(np.dot(kernel_at_samples, kernel_at_samples))
            scale = (
                float(np.dot(design_column, kernel_at_samples)) / denom
                if denom > 0.0
                else 1.0
            )
            model_yhat_avg += float(a_avg[pulse_idx]) * scale * kernel_dense

    return {
        'tau_r_s': float(tau_r),
        'tau_d_s': np.asarray(tau_d_vec, float),
        'stim_times_s': np.asarray(stim_times, float),
        'event_model': {
            'name': ev_model_name,
            'n_coop': (float(n_coop_effective) if n_coop_effective is not None else None),
        },
        'interpolated_settings': interpolated_settings,
        'average': {
            'amp_raw': np.asarray(amp_raw_avg_unfloored, float),
            'amp_savgol': np.asarray(amp_sg_avg_unfloored, float),
            'amp_nnls': np.asarray(amp_nnls_avg_unfloored, float),
            'amp_raw_corr': np.asarray(amp_raw_corr_avg_unfloored, float),
            'amp_savgol_corr': np.asarray(amp_sg_corr_avg_unfloored, float),
            'amp_nnls_corr': np.asarray(amp_nnls_corr_avg_unfloored, float),
            'ppr_nnls': np.asarray(ppr_nnls_avg, float),
            'ppr_nnls_corr': np.asarray(ppr_nnls_corr_avg, float),
            'y_avg': np.asarray(y_avg, float),
            'yhat_avg': np.asarray(yhat_avg, float),
            'model_time_s': model_time_s,
            'yhat_avg_oversampled': model_yhat_avg,
        },
        'recut_snippets': recut_snippets,
        'recut_t_rel': np.asarray(recut_t_rel, float) if recut_t_rel is not None else None,
        'recut_avg': np.asarray(recut_avg, float) if recut_avg is not None else None,
        'per_event_param_map': per_event_param_map,
        'variant_info': variant_info_avg,  # NNLS variant info with dominant fractions per event
        'per_trial': per_trial,
        'trial_input_cols_kept_0based': np.asarray(kept_trial_cols_0based, int),
        'trial_input_cols_kept_1based': np.asarray(kept_trial_cols_0based + 1, int),
        'trial_input_cols_dropped_0based': np.asarray(dropped_trial_cols_0based, int),
        'trial_input_cols_dropped_1based': np.asarray(dropped_trial_cols_0based + 1, int),
        'time_s': np.asarray(t, float),
        'threshold_amp1': np.asarray(thr_list, float),
        'median_threshold_floor': float(median_threshold) if np.isfinite(median_threshold) else np.nan,
        'pval_amp1': np.asarray(pval_list, float),
        'figure': figure,
        'figure_fit_diagnostic': fit_diag_figure,
        'figure_residual_buildup': figure_residual_buildup,
        'figure_nnls_residual': figure_nnls_residual,
        'figure_param_evolution': figure_param_evolution,
        'figure_event_model': None,
        'figures_trials': figures_trials,
    }


# ----------------------------------------------
# Convenience: export multiple folders to Excel
# ----------------------------------------------
def export_folders_to_excel(paths,
                            out_file,
                            *,
                            train_start: float,
                            isi: float,
                            n_pulses: int,
                            options: Optional[Dict] = None,
                            save_traces: bool = True):
    """Process all .xlsx files in each folder and write a multi-sheet Excel.

    - One sheet per input folder (sheet named after the folder's basename)
    - Each sheet: one row per file with AMP1..AMPn, PPR2/1.., optional %Fail1..3
    - If save_traces=True (default), also writes a companion file 
      `<out_file_stem>_traces.xlsx` containing preprocessed average traces
      with one sheet per condition (columns: Time, ID1, ID2, ...)

    Options follow extract_metrics; 'measurement' selects the amplitude series
    used for export; 'fail_method' controls failure classification.
    """
    import os, glob, zipfile
    import pandas as pd

    if isinstance(paths, str):
        paths = [paths]

    def _is_valid_xlsx(path: str) -> bool:
        try:
            with zipfile.ZipFile(path) as z:
                return '[Content_Types].xml' in z.namelist()
        except Exception:
            return False

    cfg = {} if options is None else dict(options)
    meas = str(cfg.get('measurement', 'NNLS')).strip().upper()
    failm = str((cfg.get('fail_method') or meas)).strip().upper()

    # Store traces for companion file: {sheet_name: {id: (time, y_avg)}}
    traces_by_sheet: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}

    with pd.ExcelWriter(out_file) as writer:
        wrote_any = False
        for folder in paths:
            files = [p for p in glob.glob(os.path.join(folder, "*.xlsx")) if _is_valid_xlsx(p)]
            if not files:
                continue
            rows = []
            sheet = os.path.basename(os.path.normpath(folder))[:31]
            traces_by_sheet[sheet] = {}

            for fp in files:
                try:
                    df = pd.read_excel(fp, sheet_name=0)
                    t_raw = pd.to_numeric(df.iloc[:, -1], errors='coerce').to_numpy(float)
                    _all_cols = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').to_numpy(float)
                    # Auto-detect and skip the average column (penultimate = mean of preceding)
                    _skip_avg = False
                    if _all_cols.shape[1] >= 2:
                        _cand = _all_cols[:, -1]
                        _prec = _all_cols[:, :-1]
                        _cmean = np.nanmean(_prec, axis=1)
                        _fin = np.isfinite(_cand) & np.isfinite(_cmean)
                        if _fin.sum() > 10:
                            _cc = np.corrcoef(_cand[_fin], _cmean[_fin])[0, 1]
                            if _cc > 0.99:
                                _skip_avg = True
                    X = _all_cols[:, :-1] if _skip_avg else _all_cols
                    ok = np.isfinite(t_raw)
                    t = t_raw[ok] ; trials = X[ok, :]
                    
                    # add time offset of 2ms
                    t += 0.002

                    res = extract_metrics(t, trials, train_start=train_start, isi=isi, n_pulses=n_pulses, options=cfg)

                    # Choose corrected amplitude series per measurement when available
                    if meas == 'SAVGOL':
                        amp_avg = res['average'].get('amp_savgol_corr', res['average']['amp_savgol'])
                    elif meas == 'RAW':
                        amp_avg = res['average'].get('amp_raw_corr', res['average']['amp_raw'])
                    else:
                        amp_avg = res['average'].get('amp_nnls_corr', res['average']['amp_nnls'])

                    # Build row
                    base = os.path.splitext(os.path.basename(fp))[0]
                    row = {'ID': base}
                    amp_avg = [float(x) if x is not None else float('nan') for x in amp_avg]
                    for i, v in enumerate(amp_avg, start=1):
                        row[f"AMP{i}"] = v
                    a1 = row.get('AMP1')
                    for i in range(2, int(n_pulses) + 1):
                        row[f"PPR{i}/1"] = (row.get(f"AMP{i}") / a1) if (a1 not in (None, 0) and pd.notna(a1)) else float('nan')

                    # Failure % for pulses 1..3 using fail_method amplitudes and shared thresholds if present
                    # Collect per-trial from res['per_trial']
                    def _amp_vec_per_trial(key):
                        vals = []
                        for r in res['per_trial']:
                            arr = r.get(key)
                            vals.append(arr if arr is not None else [])
                        return vals

                    if failm == 'SAVGOL':
                        per_amp_corr = _amp_vec_per_trial('amp_savgol_corr')
                        per_amp_uncorr = _amp_vec_per_trial('amp_savgol')
                    elif failm == 'RAW':
                        per_amp_corr = _amp_vec_per_trial('amp_raw_corr')
                        per_amp_uncorr = _amp_vec_per_trial('amp_raw')
                    else:
                        per_amp_corr = _amp_vec_per_trial('amp_nnls_corr')
                        per_amp_uncorr = _amp_vec_per_trial('amp_nnls')

                    # Use per-trial shared threshold and compare first 3 pulses
                    n_fail = [0, 0, 0]
                    n_valid = [0, 0, 0]
                    for idx_trial, r in enumerate(res['per_trial']):
                        thr = r.get('thr_shared')
                        amps_corr = per_amp_corr[idx_trial]
                        amps_uncorr = per_amp_uncorr[idx_trial]
                        if (
                            thr is None
                            or not isinstance(amps_corr, (list, tuple, np.ndarray))
                            or not isinstance(amps_uncorr, (list, tuple, np.ndarray))
                        ):
                            continue
                        for k in range(3):
                            vals_k = []
                            if len(amps_corr) > k:
                                ak_corr = float(amps_corr[k])
                                if np.isfinite(ak_corr):
                                    vals_k.append(ak_corr)
                            if len(amps_uncorr) > k:
                                ak_uncorr = float(amps_uncorr[k])
                                if np.isfinite(ak_uncorr):
                                    vals_k.append(ak_uncorr)
                            if vals_k and np.isfinite(thr):
                                n_valid[k] += 1
                                if min(vals_k) < float(thr):
                                    n_fail[k] += 1
                    for k in range(3):
                        row[f"%Fail{k+1}"] = round((n_fail[k] / n_valid[k]) * 100.0, 2) if n_valid[k] else float('nan')

                    rows.append(row)

                    # Store average trace for companion file
                    if save_traces:
                        y_avg = res['average'].get('y_avg')
                        time_s = res.get('time_s')
                        if y_avg is not None and time_s is not None:
                            traces_by_sheet[sheet][base] = (np.asarray(time_s, float), np.asarray(y_avg, float))

                except Exception:
                    continue

            if rows:
                import pandas as _pd
                df_out = _pd.DataFrame(rows)
                # Ensure ordered columns
                cols = ['ID'] + [f"AMP{i}" for i in range(1, int(n_pulses) + 1)] + [f"PPR{i}/1" for i in range(2, int(n_pulses) + 1)] + [f"%Fail{i}" for i in range(1, 4)]
                for c in cols:
                    if c not in df_out.columns:
                        df_out[c] = float('nan')
                df_out = df_out[cols]
                df_out.to_excel(writer, sheet_name=sheet, index=False)
                wrote_any = True

        if not wrote_any:
            # Placeholder sheet
            import pandas as _pd
            _pd.DataFrame({"info": ["No valid data found"]}).to_excel(writer, sheet_name="Summary", index=False)

    # Write companion traces and times files if requested
    # Two separate files are created:
    #   - <out_stem>_traces.xlsx: one column per bouton ID (no time column)
    #   - <out_stem>_times.xlsx: one column per bouton ID with time vectors
    # Both files have the same sheet/column organization for easy pairing.
    if save_traces and traces_by_sheet:
        out_stem = os.path.splitext(out_file)[0]
        traces_file = f"{out_stem}_traces.xlsx"
        times_file = f"{out_stem}_times.xlsx"
        with pd.ExcelWriter(traces_file) as trace_writer, pd.ExcelWriter(times_file) as time_writer:
            wrote_traces = False
            for sheet_name, id_traces in traces_by_sheet.items():
                if not id_traces:
                    continue
                # Build DataFrames: one column per bouton ID (no Time column in traces)
                # Use pd.Series to handle different lengths per column
                trace_dict = {}
                time_dict = {}
                for bid, (t_vec, y_avg) in sorted(id_traces.items()):
                    trace_dict[bid] = pd.Series(y_avg)
                    time_dict[bid] = pd.Series(t_vec)
                trace_df = pd.DataFrame(trace_dict)
                time_df = pd.DataFrame(time_dict)
                trace_df.to_excel(trace_writer, sheet_name=sheet_name[:31], index=False)
                time_df.to_excel(time_writer, sheet_name=sheet_name[:31], index=False)
                wrote_traces = True
            if not wrote_traces:
                pd.DataFrame({"info": ["No traces found"]}).to_excel(trace_writer, sheet_name="Summary", index=False)
                pd.DataFrame({"info": ["No traces found"]}).to_excel(time_writer, sheet_name="Summary", index=False)
        progress_print(f"[export] Saved average traces to: {traces_file}")
        progress_print(f"[export] Saved time vectors to: {times_file}")
