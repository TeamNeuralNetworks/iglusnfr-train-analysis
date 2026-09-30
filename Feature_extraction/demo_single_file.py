"""
demo_single_file.py - Single-file analysis for iGluSnFR pulse trains

Quick reference for options (see Model_Calibration/event_models.py for details):
  - event_model: 'double_exp', 'iglusnfr', 'single_exp', 'cooperative', etc.
  - decay_progression_mode: 'fixed', 'linear', 'free_monotonic', 'none'
  - fit_source: 'global', 'average', 'individual'
  - nnls_weight_mode: 'uniform', 'linear', 'exponential', 'savgol', 'peak'
"""

import os, sys, json, math, numpy as np, pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# =============================================================================
#                         USER CONFIGURATION - EDIT HERE
# =============================================================================

# --- Data paths ---
DATA_ROOT = os.path.abspath(os.environ.get(
    "GLUSNFR_DATA_ROOT", os.path.join(os.path.expanduser("~"), "Desktop", "FINAL PUBLICATION TEST PROCESS")
))
OUT_DIR = os.path.join(DATA_ROOT, "Testout")

# Recordings are loaded from the well-organized raw stage (release/raw/<uid>.csv +
# release/boutons_manifest.csv), resolved by uid or legacy_id via dataset_tools.raw_loader
# - never from the private per-condition .xlsx tree.

# --- File list: uncomment to process multiple files in one run ---
# When FILES is set (not None), the script loops over all entries.
# When FILES is None, it processes the single TARGET_LEGACY_ID below.
# Entries are legacy_id values (the old raw filename stem, without ".xlsx") or uid
# values (<base_uid>__<condition>) from release/boutons_manifest.csv.
FILES = [
    "250128_Fibre2_Bouton_5",
    "250305_Fibre4_Bouton_4",
    "20190801_linescan1_20Hz_10pulses_2.5mMCa_bouton3_traces_converted",
    "20191017_linescan3_50Hz_10pulses_2.5mMCa_bouton1_traces_converted",
    "20191017_linescan3_50Hz_10pulses_2.5mMCa_bouton2_traces_converted",
    "20191017_linescan3_50Hz_10pulses_2.5mMCa_bouton4_traces_converted",
    "20201022_linescan1_50Hz_10pulses_2.5mMCa_bouton1_traces_converted",
    "20201022_linescan1_50Hz_10pulses_2.5mMCa_bouton2_traces_converted",
    "20201022_linescan1_50Hz_10pulses_2.5mMCa_bouton4_traces_converted",
    "20201030_linescan2_50Hz_10pulses_2.5mMCa_bouton3_traces_converted",
    "20210128_linescan5_20Hz_10pulses_2.5mMCa_bouton2_traces_converted",
    "20210512_linescan2_50Hz_10pulses_2.5mMCa_bouton6_traces_converted",
    "20210518_linescan1_50Hz_10pulses_2.5mMCa_bouton1_traces_converted",
    "20210518_linescan1_50Hz_10pulses_2.5mMCa_bouton2_traces_converted",
    "20210722_linescan3_50Hz_10pulses_1.5mMCa_bouton3_traces_converted",
    "20210722_linescan3_50Hz_10pulses_1.5mMCa_bouton12_traces_converted",
    "20220726_linescan5_50Hz_10pulses_1.5mMCa_bouton2_traces_converted",
    "20220726_linescan5_50Hz_10pulses_4mMCa_bouton2_traces_converted",
    "20220726_linescan6_50Hz_10pulses_4mMCa_bouton1_traces_converted",
]



# --- Select recording ---
# If CONDITION is empty/None, resolved via the manifest; set it only to disambiguate
# a legacy_id that matches more than one condition.
CONDITION = ""
#TARGET_LEGACY_ID = "20191017_linescan3_50Hz_10pulses_2.5mMCa_bouton2_traces_converted"
#TARGET_LEGACY_ID = "20210722_linescan3_50Hz_10pulses_1.5mMCa_bouton12_traces_converted"
#TARGET_LEGACY_ID = "20210512_linescan2_50Hz_10pulses_2.5mMCa_bouton1_traces_converted"
TARGET_LEGACY_ID = "20220726_linescan5_50Hz_10pulses_4mMCa_bouton2_traces_converted"
TARGET_LEGACY_ID = "20220726_linescan5_50Hz_10pulses_1.5mMCa_bouton2_traces_converted"


# --- Select analysis preset ---
PRESET_NAME = 'iglusnfr_optimized'  # Options: 'iglusnfr_optimized', 'double_exp', 'single_exp_fixed_8ms'

# --- Manual overrides (set to None to use the manifest) ---
OVERRIDE_ISI = None         # e.g., 0.02 for 50Hz, 0.05 for 20Hz
OVERRIDE_BASELINE = None    # e.g., 0.498 or 0.998
OVERRIDE_N_PULSES = None    # e.g., 10

# Fixed pulse-column count for the output summary table (independent of any one
# recording's resolved n_pulses).
DEFAULT_N_PULSES = 10

# =============================================================================
#                         ANALYSIS OPTIONS PRESETS
# =============================================================================

def _build_options_presets(peak_window_ms, pre_zoom_s, post_zoom_s):
    """Build options presets with ISI-aware parameters."""
    return {
        'iglusnfr_optimized': {
            # --- Preprocessing ---
            'normalize_dff': True,                                          # Normalize to dF/F0
            'bleach': True,                                                 # Apply bleach correction
            'sg_window': 9,                                                 # Savitzky-Golay filter window (must be odd)
            'sg_poly': 2,                                                   # Savitzky-Golay filter polynomial order
            
            # --- Kinetics ---
            'fit_source': 'global',                                         # 'global', 'average', 'individual' ; this controls the source of data for kinetics fitting
            'decay_progression_mode': 'linear',                               # 'fixed', 'linear', 'free_monotonic', 'none' ; this controls how decay kinetics evolve over pulses
            'anchor_final_tau': False,                                      # Whether to anchor the final event tau to the last-event estimate
            'anchor_first_tau': False,                                      # Whether to anchor the first event tau to a fixed value
            
            # --- Event Model ---
            'event_model': 'iglusnfr_tri',                                  # 'double_exp', 'iglusnfr', 'iglusnfr_tri', 'single_exp', 'cooperative'
            'parameter_bounds': {
                'tau_decay_fast': (0.003, 0.008),                           # fast decay bounds (s)
                'tau_decay_slow': (0.008, 0.035),                           # slow decay bounds (s)
                'tau_superslow': (0.035, 0.090),                            # superslow decay bounds (s) - only for tri-exponential
                'amplitude_ratio': (0.0, 1.0),                              # amplitude ratio bounds (0 to 1) ; 0 means all fast, 1 means all slow
            },
            'early_events_only': 0,                                         # Use only first N events for kinetics fitting (0 = all events)
            
            # --- Recut/Averaging ---
            'recut_projection': 'mean',                                     # 'mean', 'median'
            'recut_oversample': 20,                                         # Oversampling factor for recut snippets ; data is projected onto a finer time grid
            'recut_peak_recenter': 0,                                       # Recenter recut snippets on peak (0 = no recentering)
            'recut_snippets': True,                                         # Whether to extract recut snippets for visualization
            
            # --- Onset Detection ---
            'onset_method': 'baseline_threshold',                           # 'baseline_threshold', 'derivative' ; method for onset detection of recut snippets
            'onset_baseline_threshold': 0.10,                               # Threshold (fraction of peak) for baseline_threshold onset detection
            
            # --- PPR Safety ---
            'amplitude_floor_to_noise': True,                               # Floor per-trial pulse amplitudes to the per-trial A1 threshold (thr1) before per-trial PPR; the average trace is not floored (see next option)
            'average_amplitude_floor_to_noise': False,                      # Separate control for average trace floor; None => follow amplitude_floor_to_noise
            'average_null_N': None,                                         # Separate null_N multiplier for average trace floor; None => follow null_N
            
            # --- NNLS Fitting ---
            'nnls_fit_mode': 'sequential',                                  # 'simultaneous' (all events jointly) or 'sequential' (greedy forward pass, resolves fast/superslow degeneracy)
            'nnls_weight_mode': 'savgol',                                   # 'uniform', 'linear', 'exponential', 'savgol', 'peak' ; weighting scheme for NNLS fitting
            'nnls_weight_tau_s': None,                                      # Time constant for exponential weighting (s) ; only used if nnls_weight_mode is 'exponential'
            'nnls_peak_window_s': 0.010,                                    # Peak-emphasis window after each stimulus (s)
            'nnls_peak_weight': 3.0,                                        # Weight multiplier inside the peak window
            'fit_diagnostic_plot': False,                                   # Whether to generate fit diagnostic plots
            'huber_delta': 2.5,                                             # Huber loss delta for robust fitting (in std units); set to None to disable robust fitting
            'irls_iters': 20,                                               # Number of IRLS iterations for robust fitting ; only used if huber_delta is set
            'nnls_last_event_tail_tau_s': 'best',                           # Last event tail downweight tau (s); None=off, 'auto'=ISI, 'best'=search for optimal, or float
            'nnls_two_pass_guard': False,                                    # If True, reject pass 2 when it worsens RMS; if False, always use pass 2 (smoothed fractions)

            # --- Time Windows (ISI-aware) ---
            'pre_zoom_s': pre_zoom_s,                                       # Pre-event snippet duration (s); controls how much data before each event is shown ; does not affect fitting
            'post_zoom_s': post_zoom_s,                                     # Post-event snippet duration (s); controls how much data after each event is shown ; does not affect fitting
            'f0_window_s': 1.0,                                             # F0 baseline window duration (s) ; controls how baseline F0 is computed for dF/F0 normalization
            
            # --- Peak Detection (ISI-aware) ---
            'peak_window_ms': peak_window_ms,                               # Peak detection window duration (ms) ; controls how peaks are identified within each event
            'peak_avg_points': 3,                                           # Number of points to average around peak for amplitude measurement
            'pre_peak_ms': 1.0,                                             # Pre-peak baseline window (ms) ; controls how local baseline before each peak is computed ;
            
            # --- Thresholding ---
            'measurement': 'NNLS',                                          # 'NNLS', 'SAVGOL', 'RAW' ; amplitude series used for p-values/classification
            'fail_method': 'SAVGOL',                                        # 'NNLS', 'SAVGOL', 'RAW' ; method used to build null/noise amplitudes for thresholding
            'threshold_mode': 'auto',                                       # 'auto', 'mad', 'sd' ; auto => mad for NNLS null, sd for SAVGOL/RAW null
            'null_N': 1.0,                                                  # Multiplier for null distribution to set threshold ; only used if threshold_mode is 'auto'
            'null_sim_max_points': 1000,                                    # Max points for null distribution simulation
            'null_min_post_zoom_s': 0.05,                                   # Minimum post-zoom duration (s) to use for null distribution simulation
            
            # --- Bleach Correction ---
            'bleach_huber_delta': 3.0,                                      # Huber loss delta for bleach fitting (in std units); set to None to disable robust fitting
            'bleach_tau_range_factor': (0.25, 4.0),                         # Range factor for bleach tau fitting ; multiplied by initial estimate to get min and max bounds
            'bleach_n_tau': 25,     
            'template_variant_select': 'soft',                                        # Number of tau candidates for bleach fitting
            
            # --- Plotting ---
            'plot': {
                'enabled': True,                                            # Master plot enable/disable
                'traces': ['raw', 'nnls'],                                  # 'raw', 'bleach_corrected', 'nnls', 'nnls_corr' ; which traces to plot
                'figsize': (10, 6),                                         # Figure size
                'show_decay': True,                                         # Show decay fits on average plot
                'show_onsets': True,                                        # Show detected onsets on recut snippets
                'trials': False,                                           # Whether to generate per-trial figures
                'baseline': False,                                         # Whether to show baseline F0 levels on traces
                'residuals': True,                                          # Whether to show residuals on average plot
                'nnls_residual': False,                                     # Whether to show NNLS residuals on average plot
                'nnls_n_minus_1': False,                                    # Whether to show NNLS n-1 fit on average plot
                'plot_peaks_details': True,                                 # Show peak detection details on traces
                'param_evolution': True,                                    # Show parameter evolution across events
            },
            
            # --- Template Variants ---
            'use_template_variants': True,                                  # Enable variant testing, where we try multiple tau combinations
            'template_variant_ratios': np.linspace(0.1, 0.9, 10),           # bi-exp and tri-exp
            'template_variant_superslow_fracs': np.linspace(0.1, 0.9, 11),  # tri-exp only
            'superslow_min_ratio': 1.0,                                     # Minimum ratio between slow and superslow taus for tri-exp variants
            'allow_tau_slow_override': True,                                # If True, allow tau_slow = tau_superslow in tri-exp models if it improves fit
            'force_tau_slow_override': False,                                # If True, force tau_slow = tau_superslow in tri-exp models ; unlike allow_tau_slow_override, this enforces the equality rather than just allowing it
            
            # --- Jitter Variants ---
            'jitter_variant_ms': np.arange(-2.0, 2.01, 0.5),                # Jitter variants to try (ms) ; set to None to disable jitter variants ; jitter means we shift event times by +/- jitter to test robustness
        },
    }


# =============================================================================
#                              INTERNAL SETUP
# =============================================================================
# (No need to edit below unless debugging)

try:
    _here = os.path.dirname(__file__)
except NameError:
    _here = os.getcwd()
REPO_ROOT = os.path.abspath(os.path.join(_here, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Feature_extraction.extract_metrics import extract_metrics
from dataset_tools import raw_loader

def _ensure_columns(df, ordered_cols):
    """Ensure all columns exist (fill missing with NaN) and reorder; extras follow at the end."""
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = np.nan
    rest = [c for c in df.columns if c not in ordered_cols]
    return df[ordered_cols + rest]


def _measurement_series_keys(measurement: str) -> dict:
    """Return result keys for the selected measurement series."""
    meas = str(measurement).strip().upper()
    if meas == 'SAVGOL':
        return {
            'label': 'SAVGOL',
            'avg_uncorr': 'amp_savgol',
            'avg_corr': 'amp_savgol_corr',
            'trial_uncorr': 'amp_savgol_unfloored',
            'trial_corr': 'amp_savgol_corr_unfloored',
        }
    if meas == 'RAW':
        return {
            'label': 'RAW',
            'avg_uncorr': 'amp_raw',
            'avg_corr': 'amp_raw_corr',
            'trial_uncorr': 'amp_raw_unfloored',
            'trial_corr': 'amp_raw_corr_unfloored',
        }
    return {
        'label': 'NNLS',
        'avg_uncorr': 'amp_nnls',
        'avg_corr': 'amp_nnls_corr',
        'trial_uncorr': 'amp_nnls_unfloored',
        'trial_corr': 'amp_nnls_corr_unfloored',
    }


def _compute_ppr_from_amplitudes(amps) -> np.ndarray:
    """Return AMPn / AMP1 using the provided amplitude vector."""
    arr = np.asarray(amps, float)
    if arr.size == 0:
        return arr
    a1 = float(arr[0])
    return (arr / a1) if np.isfinite(a1) and abs(a1) > 1e-12 else arr * np.nan


def _threshold_value(corr_arr, uncorr_arr, idx: int) -> float:
    """Return the conservative amplitude used for failure calls."""
    vals = []
    if idx < len(corr_arr):
        v = float(corr_arr[idx])
        if np.isfinite(v):
            vals.append(v)
    if idx < len(uncorr_arr):
        v = float(uncorr_arr[idx])
        if np.isfinite(v):
            vals.append(v)
    return float(min(vals)) if vals else np.nan


# --- Build file list to process ---
if FILES is not None:
    _file_list = list(FILES)
else:
    _file_list = [TARGET_LEGACY_ID]

_manifest = raw_loader.load_manifest(DATA_ROOT)

_errors = []
for _file_idx, TARGET_LEGACY_ID in enumerate(_file_list):
    CONDITION_HINT = CONDITION or None  # only used to disambiguate

    # --- Resolve the recording via the manifest ---
    try:
        _row = raw_loader.resolve_recording(
            _manifest, legacy_id=TARGET_LEGACY_ID, condition=CONDITION_HINT,
        )
    except KeyError:
        try:
            _row = raw_loader.resolve_recording(_manifest, uid=TARGET_LEGACY_ID)
        except KeyError as e:
            print(f"[SKIP] '{TARGET_LEGACY_ID}': {e}")
            _errors.append(TARGET_LEGACY_ID)
            continue

    CONDITION = str(_row["condition"])

    # --- Resolve parameters (use overrides if set, otherwise the manifest) ---
    _params = raw_loader.resolved_params(
        _row, override_isi=OVERRIDE_ISI, override_baseline=OVERRIDE_BASELINE,
        override_n_pulses=OVERRIDE_N_PULSES,
    )
    ISI = _params["isi"]
    START = _params["baseline_s"]
    N_PULSES = _params["n_pulses"]

    # --- Compute ISI-dependent parameters ---
    ISI_MS = ISI * 1000.0
    MARGIN_MS = 2.0  # Fixed margin before next event (ms)
    PEAK_WINDOW_MS = max(5.0, ISI_MS - MARGIN_MS)  # Use all data minus 2ms margin
    POST_ZOOM_S = 0.3  # Post-train window (s): controls plot zoom AND last-event fit window in sequential NNLS
    PRE_ZOOM_S = 0.20

    # --- Print configuration ---
    print("=" * 60)
    if len(_file_list) > 1:
        print(f"  [{_file_idx+1}/{len(_file_list)}]")
    print(f"  Condition:  {CONDITION}")
    print(f"  uid:        {_row['uid']}")
    print(f"  legacy_id:  {_row['legacy_id']}")
    print(f"  ISI:        {ISI_MS:.0f}ms ({1/ISI:.0f}Hz)")
    print(f"  Baseline:   {START:.3f}s")
    print(f"  N pulses:   {N_PULSES}")
    print(f"  Preset:     {PRESET_NAME}")
    print(f"  Peak win:   {PEAK_WINDOW_MS:.1f}ms | Post zoom: {POST_ZOOM_S:.3f}s")
    print("=" * 60)

    # --- Build presets and select ---
    options_presets = _build_options_presets(PEAK_WINDOW_MS, PRE_ZOOM_S, POST_ZOOM_S)
    if PRESET_NAME not in options_presets:
        raise ValueError(f"Unknown preset '{PRESET_NAME}'. Available: {list(options_presets)}")
    options = options_presets[PRESET_NAME]

    # --- Load data (release/raw/<uid>.csv via the manifest-resolved row) ---
    os.makedirs(OUT_DIR, exist_ok=True)
    _raw = raw_loader.load_raw_csv(DATA_ROOT, _row["uid"])
    _time = _raw.time_s
    _all_before_time = _raw.values
    _n_cols_total = 1 + _all_before_time.shape[1]

    # Auto-detect and skip the average column (penultimate = mean of preceding columns)
    _has_avg_col = False
    if _all_before_time.shape[1] >= 2:
        _candidate_avg = _all_before_time[:, -1]
        _preceding = _all_before_time[:, :-1]
        _computed_avg = np.nanmean(_preceding, axis=1)
        _finite = np.isfinite(_candidate_avg) & np.isfinite(_computed_avg)
        if _finite.sum() > 10:
            _corr = np.corrcoef(_candidate_avg[_finite], _computed_avg[_finite])[0, 1]
            if not math.isnan(_corr) and _corr > 0.99:
                _has_avg_col = True

    if _has_avg_col:
        _trials = _all_before_time[:, :-1]  # drop penultimate (average) column
        print(f"[info] Detected average column (penultimate) — excluded from trials")
    else:
        _trials = _all_before_time
        print(f"[info] No average column detected — using all data columns as trials")

    valid = np.isfinite(_time)
    time = _time[valid]
    trials = _trials[valid, :]

    print(f"[debug] Raw file has {_n_cols_total} total columns -> {trials.shape[1]} trial columns{' + 1 avg' if _has_avg_col else ''} + 1 time column")

    base = str(_row["legacy_id"])

    # =============================================================================
    #                              RUN ANALYSIS
    # =============================================================================
    res = extract_metrics(
        time, trials,
        train_start=START,
        isi=ISI,
        n_pulses=N_PULSES,
        options=options,
        filename=base
    )

    # =============================================================================
    #                              RESULTS OUTPUT
    # =============================================================================

    # --- Extract amplitudes and PPR ---
    meas_keys = _measurement_series_keys(options.get('measurement', 'NNLS'))
    amp_avg_raw = np.asarray(
        res['average'].get(meas_keys['avg_uncorr'], res['average'].get('amp_nnls')),
        float,
    )
    amp_avg = np.asarray(
        res['average'].get(meas_keys['avg_corr'], amp_avg_raw),
        float,
    )
    thr_arr = np.asarray(res.get('threshold_amp1', []), float)
    thr_arr = thr_arr[np.isfinite(thr_arr)]
    thr_median = float(np.nanmedian(thr_arr)) if thr_arr.size else np.nan
    ppr_avg = _compute_ppr_from_amplitudes(amp_avg)

    print("\n--- Results ---")
    print(f"Amplitudes ({meas_keys['label']} uncorrected):", amp_avg_raw)
    print(f"Amplitudes ({meas_keys['label']} corrected):", amp_avg)
    print(f"PPR ({meas_keys['label']} corrected):", ppr_avg)
    print("A1 thresholds:", res['threshold_amp1'])
    print("PPR floor (median thr):", res.get('median_threshold_floor', 'N/A'))
    print("A1 p-values:", res['pval_amp1'])

    # --- Build summary row ---
    row = {'measurement': meas_keys['label'], 'ID': base, 'NOISE_THR_MEDIAN': thr_median}
    for i, v in enumerate(amp_avg, 1):
        row[f'AMP{i}'] = float(v)
    for i, v in enumerate(amp_avg_raw, 1):
        row[f'AMP{i}_UNCORR'] = float(v)
    for i in range(2, len(ppr_avg) + 1):
        row[f'PPR{i}/1'] = float(ppr_avg[i - 1])

    # --- Per-trial A1 diagnostic table ---
    print(f"\n{'='*60}")
    print(f"  Per-trial A1 diagnostics ({len(res.get('per_trial', []))} trials)")
    print(f"{'='*60}")
    print(f"  {'Trial':>5}  {'A1 (corr)':>12}  {'Threshold':>12}  {'Status':>8}")
    print(f"  {'-'*5}  {'-'*12}  {'-'*12}  {'-'*8}")
    for _it, _rt in enumerate(res.get('per_trial', [])):
        _a1_corr = np.asarray(_rt.get(meas_keys['trial_corr'], _rt.get(meas_keys['avg_corr'])), float)
        _a1_uncorr = np.asarray(_rt.get(meas_keys['trial_uncorr'], _rt.get(meas_keys['avg_uncorr'])), float)
        _a1v = float(_a1_corr[0]) if _a1_corr.size else np.nan
        _a1_eval = _threshold_value(_a1_corr, _a1_uncorr, 0)
        _thrv = float(_rt.get('thr_shared', np.nan))
        _st = 'PASS' if (np.isfinite(_a1_eval) and np.isfinite(_thrv) and _a1_eval > _thrv) else 'FAIL'
        print(f"  {_it+1:>5}  {_a1v:>12.6f}  {_thrv:>12.6f}  {_st:>8}")
    _n_fail_diag = sum(1 for _rt in res.get('per_trial', [])
                       if _threshold_value(
                           np.asarray(_rt.get(meas_keys['trial_corr'], _rt.get(meas_keys['avg_corr'])), float),
                           np.asarray(_rt.get(meas_keys['trial_uncorr'], _rt.get(meas_keys['avg_uncorr'])), float),
                           0,
                       )
                       <= float(_rt.get('thr_shared', np.nan))
                       and np.isfinite(float(_rt.get('thr_shared', np.nan))))
    _n_total_diag = len(res.get('per_trial', []))
    print(f"  -> Failures: {_n_fail_diag}/{_n_total_diag} = {100*_n_fail_diag/_n_total_diag:.1f}%" if _n_total_diag else "  -> No trials")
    print(f"{'='*60}")

    # --- Per-trial failure counts ---
    per_trial_rows, per_trial_null_rows, fail_counts = [], [], {i: [0, 0] for i in range(1, 4)}
    for idx_trial, rtrial in enumerate(res.get('per_trial', [])):
        amp_trial = np.asarray(rtrial.get(meas_keys['trial_corr'], rtrial.get(meas_keys['avg_corr'])), float)
        amp_trial_uncorr = np.asarray(rtrial.get(meas_keys['trial_uncorr'], rtrial.get(meas_keys['avg_uncorr'])), float)
        thr = float(rtrial.get('thr_shared', np.nan))
        noise_level = float(rtrial.get('noise_level', np.nan))
        baseline_null_mean_including_zero = float(rtrial.get('baseline_null_mean_including_zero', np.nan))
        baseline_null_median_including_zero = float(rtrial.get('baseline_null_median_including_zero', np.nan))
        baseline_null_mean_excluding_zero = float(rtrial.get('baseline_null_mean_excluding_zero', np.nan))
        baseline_null_median_excluding_zero = float(rtrial.get('baseline_null_median_excluding_zero', np.nan))
        f0_value = float(rtrial.get('F0', np.nan))
        null_amps_nnls = np.asarray(rtrial.get('null_amps_nnls', []), float)
        null_amps_nnls = null_amps_nnls[np.isfinite(null_amps_nnls)]
        a1 = _threshold_value(amp_trial, amp_trial_uncorr, 0)
        status = 'NA'
        if np.isfinite(a1) and np.isfinite(thr):
            status = 'success' if a1 > thr else 'failure'
        trial_row = {
            'status': status,
            'file': base,
            'condition': CONDITION,
            'trial': idx_trial + 1,
            'trial_input_col_1based': int(rtrial.get('trial_input_col_1based', idx_trial + 1)),
            'F0': f0_value,
            'thr_shared': thr,
            'noise_level': noise_level,
            'baseline_null_mean_including_zero': baseline_null_mean_including_zero,
            'baseline_null_median_including_zero': baseline_null_median_including_zero,
            'baseline_null_mean_excluding_zero': baseline_null_mean_excluding_zero,
            'baseline_null_median_excluding_zero': baseline_null_median_excluding_zero,
        }
        for p in range(1, int(DEFAULT_N_PULSES) + 1):
            vc = float(amp_trial[p - 1]) if p <= amp_trial.size and np.isfinite(amp_trial[p - 1]) else np.nan
            vu = (
                float(amp_trial_uncorr[p - 1])
                if p <= amp_trial_uncorr.size and np.isfinite(amp_trial_uncorr[p - 1])
                else np.nan
            )
            trial_row[f'AMP{p}_CORR'] = vc
            trial_row[f'AMP{p}_UNCORR'] = vu
        # Backward-compatible legacy column: single-trial AMP1 should remain uncorrected.
        trial_row['AMP1'] = trial_row.get('AMP1_UNCORR', np.nan)
        per_trial_rows.append(trial_row)
        per_trial_null_rows.append({
            'condition': CONDITION,
            'file': base,
            'trial': idx_trial + 1,
            'trial_input_col_1based': int(rtrial.get('trial_input_col_1based', idx_trial + 1)),
            'status': status,
            'nnls_null_n': int(null_amps_nnls.size),
            'nnls_null_amps_json': json.dumps([float(v) for v in null_amps_nnls.tolist()]),
        })
        for p in range(1, min(3, amp_trial.size) + 1):
            val = _threshold_value(amp_trial, amp_trial_uncorr, p - 1)
            if np.isfinite(val) and np.isfinite(thr):
                fail_counts[p][1] += 1
                if val <= thr:
                    fail_counts[p][0] += 1
    for p in range(1, 4):
        n_fail, n_valid = fail_counts[p]
        if n_valid:
            row[f'%Fail{p}'] = round((n_fail / n_valid) * 100.0, 2)

    # --- Save to files ---
    df_rows = pd.DataFrame([row])
    ordered = ([f'AMP{i}' for i in range(1, 11)]
               + [f'AMP{i}_UNCORR' for i in range(1, 11)]
               + [f'PPR{i}/1' for i in range(2, 11)]
               + [f'%Fail{i}' for i in range(1, 4)]
               + ['NOISE_THR_MEDIAN'])
    df_rows = _ensure_columns(df_rows, ['ID', *ordered, 'measurement'])

    csv_out = os.path.join(OUT_DIR, f"{base}_summary.csv")
    df_rows.to_csv(csv_out, index=False)
    xl_out = os.path.splitext(csv_out)[0] + ".xlsx"
    df_rows.to_excel(xl_out, index=False)
    if per_trial_rows:
        df_trials = pd.DataFrame(per_trial_rows)
        trial_cols = (
            [
                'file', 'trial', 'trial_input_col_1based', 'status', 'F0',
                'thr_shared', 'noise_level',
                'baseline_null_mean_including_zero', 'baseline_null_median_including_zero',
                'baseline_null_mean_excluding_zero', 'baseline_null_median_excluding_zero',
            ]
            + [f'AMP{i}_CORR' for i in range(1, DEFAULT_N_PULSES + 1)]
            + [f'AMP{i}_UNCORR' for i in range(1, DEFAULT_N_PULSES + 1)]
            + ['AMP1']
        )
        df_trials = _ensure_columns(df_trials, trial_cols)
        df_trials.to_excel(os.path.splitext(xl_out)[0] + "_trials.xlsx", index=False)
    if per_trial_null_rows:
        df_trials_null = pd.DataFrame(per_trial_null_rows)
        null_cols = ['condition', 'file', 'trial', 'trial_input_col_1based', 'status', 'nnls_null_n', 'nnls_null_amps_json']
        df_trials_null = _ensure_columns(df_trials_null, null_cols)
        df_trials_null.to_excel(os.path.splitext(xl_out)[0] + "_trials_nnls_null.xlsx", index=False)

    # =============================================================================
    #                              PLOTTING
    # =============================================================================

    fig = res.get('figure')
    if fig is not None:
        try:
            fig.savefig(os.path.join(OUT_DIR, f"{base}_plot.png"), dpi=150)
        except Exception as e:
            print(f"[warn] Failed to save main figure: {e}")

    # Per-trial figures
    figs_trials = res.get('figures_trials') or []
    for i, ftri in enumerate(figs_trials, 1):
        try:
            outp = os.path.join(OUT_DIR, f"{base}_trialfig_{i:02d}.png")
            ftri.tight_layout()
            ftri.savefig(outp, dpi=120)
        except Exception as e:
            print(f"[warn] Failed to save trial figure {i:02d}: {e}")

    # Parameter evolution figure
    fig_param = res.get('figure_param_evolution')
    if fig_param is not None:
        try:
            fig_param.savefig(os.path.join(OUT_DIR, f"{base}_param_evolution.png"), dpi=150)
            print(f"[demo] Saved param evolution figure")
        except Exception as e:
            print(f"[demo] Error saving param evolution figure: {e}")

    plt.close('all')

    print()
    print("=" * 60)
    print(f"  DONE: {base}")
    print(f"  Output: {OUT_DIR}")
    print("=" * 60)

# --- Final summary ---
if len(_file_list) > 1:
    print(f"\n{'='*60}")
    print(f"  Processed {len(_file_list) - len(_errors)}/{len(_file_list)} files")
    if _errors:
        print(f"  Skipped ({len(_errors)}):")
        for e in _errors:
            print(f"    - {e}")
    print(f"{'='*60}")
