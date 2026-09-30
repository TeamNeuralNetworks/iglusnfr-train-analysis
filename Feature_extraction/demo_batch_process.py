"""
demo_batch_process.py - Batch analysis for iGluSnFR pulse trains

Quick reference for options (see Model_Calibration/event_models.py for details):
  - event_model: 'double_exp', 'iglusnfr', 'single_exp', 'cooperative', etc.
  - decay_progression_mode: 'fixed', 'linear', 'free_monotonic', 'none'
  - fit_source: 'global', 'average', 'individual'
  - nnls_weight_mode: 'uniform', 'linear', 'exponential', 'savgol', 'peak'
"""

import os, sys, glob, json, math, traceback, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# =============================================================================
#                         USER CONFIGURATION - EDIT HERE
# =============================================================================

# --- Data paths ---
DATA_ROOT = os.path.abspath(os.environ.get(
    "GLUSNFR_DATA_ROOT", os.path.join(os.path.expanduser("~"), "Desktop", "FINAL PUBLICATION TEST PROCESS")
))
OUT_DIR = os.path.join(DATA_ROOT, "FINALOUT_CLEAN_SAVGOL_FAILS_NEW_3")

# Clean, flat, Zenodo-ready output (the consolidated tidy tables land here).
RELEASE_DIR = os.path.join(DATA_ROOT, "release")

# Per-bouton manifest (release/boutons_manifest.csv, the metrics-free manifest
# written by build_manifest.py). When present it is the authoritative source of
# frequency / baseline / n_pulses; the lookup dicts below are a fallback.
# Generate it with: python dataset_tools/build_manifest.py --data-root <DATA_ROOT>
MANIFEST_PATH = os.path.join(RELEASE_DIR, "boutons_manifest.csv")

# The pipeline runs from the release layout: recordings in release/raw/<uid>.csv,
# driven by the manifest (release/boutons_manifest.csv, or release/boutons.csv as a
# fallback once metrics have already been merged into it).
RAW_DIR = os.path.join(RELEASE_DIR, "raw")
RELEASE_MANIFEST = os.path.join(RELEASE_DIR, "boutons.csv")

# --- Select conditions and files ---
# If CONDITIONS_TO_RUN is empty/None, the script will process all conditions
CONDITIONS_TO_RUN = []  # e.g., ["Theo_4_50Hz"]
FILE_GLOB = "*.xlsx"

# --- Select analysis preset ---
PRESET_NAME = 'iglusnfr_optimized'  # Options: 'iglusnfr_optimized', 'double_exp', 'single_exp_fixed_8ms'

# --- Manual overrides (set to None to use lookup tables) ---
OVERRIDE_ISI = None       # e.g., 0.02 for 50Hz, 0.05 for 20Hz
OVERRIDE_BASELINE = None  # e.g., 0.498 or 0.998
OVERRIDE_N_PULSES = None  # e.g., 10

# --- Plot output ---
SAVE_PLOTS = True
SHOW_PLOTS = False

# --- Consolidated tidy output ---
# When True, also write <OUT_DIR>/derived/{boutons,trials,null_amps,traces}.csv
# (the Zenodo-ready tidy tables, merged with release/boutons_manifest.csv). The legacy
# summary_* files are still written so an output-folder diff stays value-only.
WRITE_TIDY = True

# --- Parallel batch processing ---
PARALLEL_FILES = True
FILE_WORKERS = max(1, (os.cpu_count() or 1))

# =============================================================================
#                    MANIFEST (single source of truth)
# =============================================================================
# All experimental parameters come from the manifest (release/boutons.csv). These
# defaults are only an ultimate fallback if a manifest row is missing a value.

DEFAULT_ISI = 0.05       # 20Hz
DEFAULT_BASELINE = 0.998
DEFAULT_N_PULSES = 10


def _manifest_path():
    return MANIFEST_PATH if os.path.exists(MANIFEST_PATH) else RELEASE_MANIFEST


def _load_manifest_df():
    """Load the manifest DataFrame (release/boutons.csv), or None if absent."""
    path = _manifest_path()
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# Cached (condition, legacy_id) -> (freq_hz, baseline_s, n_pulses) for per-file
# parameter resolution (lazy, per-process so it survives the ProcessPoolExecutor).
_MANIFEST = {"loaded": False, "map": {}}


def _manifest_lookup():
    if not _MANIFEST["loaded"]:
        _MANIFEST["loaded"] = True
        path = _manifest_path()
        if os.path.exists(path):
            try:
                m = pd.read_csv(path)
                _MANIFEST["map"] = {
                    (str(r.condition), str(r.legacy_id)):
                        (float(r.frequency_hz), float(r.baseline_s), int(r.n_pulses))
                    for r in m.itertuples(index=False)
                }
            except Exception as e:
                print(f"[warn] manifest load failed ({MANIFEST_PATH}): {e}")
    return _MANIFEST["map"]

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
                # NOTE: two entries were removed here because neither was ever read.
                #  - 'tau_superslow' (0.035, 0.090): the parameter is called
                #    'tau_decay_superslow', so this bound has never been in force.
                #    Enabling it under the correct name DOES change results (on the
                #    bundled validation trace AMP10 shifts by 5.6% of AMP1 and the
                #    train R2 drops 0.998 -> 0.993), so it is left off to keep runs
                #    comparable with the published dataset. Uncomment to apply it.
                #  - 'amplitude_ratio' (0.0, 1.0): no registered model has a parameter
                #    of that name. The tri-exponential mixing weights are 'frac_fast'
                #    and 'frac_slow', and (0, 1) is their full range anyway.
                # 'tau_decay_superslow': (0.035, 0.090),
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
            'measurement': 'SAVGOL',                                          # 'NNLS', 'SAVGOL', 'RAW' ; amplitude series used for p-values/classification
            'fail_method': 'SAVGOL',                                          # 'NNLS', 'SAVGOL', 'RAW' ; method used to build null/noise amplitudes for thresholding
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
                'trials': False,                                            # Whether to generate per-trial figures
                'baseline': False,                                          # Whether to show baseline F0 levels on traces
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

if not SHOW_PLOTS and "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"
if "MPLBACKEND" in os.environ:
    import matplotlib
    matplotlib.use(os.environ["MPLBACKEND"], force=True)
import matplotlib.pyplot as plt

try:
    _here = os.path.dirname(__file__)
except NameError:
    _here = os.getcwd()
REPO_ROOT = os.path.abspath(os.path.join(_here, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Feature_extraction.extract_metrics import extract_metrics


def _safe_sheet_name(name: str) -> str:
    """Return a workbook-safe Excel sheet name (max 31 chars, no illegal chars)."""
    cleaned = "".join(c for c in name if c not in ":\\/?*[]")
    return (cleaned or "Sheet")[:31]


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


def _accumulate_result(result, failures, summary_rows, per_trial_rows,
                       per_trial_null_rows, traces_by_condition, max_pulses_seen):
    """Merge one file result into batch accumulators. Returns updated max_pulses_seen."""
    if result.get('error'):
        msg = result['error']
        print(msg)
        failures.append(msg)
        return max_pulses_seen
    summary_rows.append(result['row'])
    per_trial_rows.extend(result['per_trial_rows'])
    per_trial_null_rows.extend(result.get('per_trial_null_rows', []))
    max_pulses_seen = max(max_pulses_seen, result['max_pulses'])
    t_vec = result.get('trace_time_s')
    y_avg = result.get('trace_avg')
    if t_vec is not None and y_avg is not None:
        traces_by_condition.setdefault(
            result['row']['condition'], {}
        )[result['row']['ID']] = (t_vec, y_avg)
    return max_pulses_seen


def _save_figure(fig, outpath: str, *, dpi=None, label: str = "figure") -> bool:
    try:
        fig.savefig(outpath, dpi=dpi)
        if os.path.getsize(outpath) <= 0:
            print(f"[save] Empty file after save ({label}): {outpath}")
            return False
    except Exception as e:
        print(f"[save] Failed {label}: {outpath} ({e})")
        return False
    print(f"[save] {label} -> {outpath}")
    return True

def _process_one_file(task: tuple, *, show_plots: bool) -> dict:
    condition, xlsx_path, base = task
    try:
        if str(xlsx_path).lower().endswith(".csv"):
            # Release layout: raw/<uid>.csv (time_s, then trial_*/average in order).
            # float_precision='round_trip' is REQUIRED: the default C parser loses
            # ~1 ULP, which flips discrete template-variant choices in the sequential
            # NNLS for boundary boutons and perturbs amplitudes.
            df = pd.read_csv(xlsx_path, float_precision="round_trip")
            time_col = "time_s" if "time_s" in df.columns else df.columns[-1]
            other_cols = [c for c in df.columns if c != time_col]
            _time = pd.to_numeric(df[time_col], errors='coerce').to_numpy(float)
            _all_before_time = df[other_cols].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        else:
            df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
            _time = pd.to_numeric(df.iloc[:, -1], errors='coerce').to_numpy(float)
            _all_before_time = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    except Exception as e:
        return {'error': f"[skip] Failed to read {xlsx_path} -> {e}"}

    # Auto-detect and skip the average column (penultimate = mean of preceding columns).
    # Applied identically to xlsx and csv so both paths select exactly the same trials,
    # including quirky files where the correlation stays below 0.99 and the average
    # column is (deliberately) kept as a trial.
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
    _trials = _all_before_time[:, :-1] if _has_avg_col else _all_before_time

    valid = np.isfinite(_time)
    time = _time[valid]
    trials = _trials[valid, :]

    # --- Resolve parameters: manual override > manifest > condition lookup dict ---
    man = _manifest_lookup().get((condition, base))
    man_freq, man_baseline, man_npulses = man if man is not None else (None, None, None)

    if OVERRIDE_ISI is not None:
        isi = OVERRIDE_ISI
    elif man_freq is not None:
        isi = 1.0 / man_freq
    else:
        isi = DEFAULT_ISI

    if OVERRIDE_BASELINE is not None:
        start = OVERRIDE_BASELINE
    elif man_baseline is not None:
        start = man_baseline
    else:
        start = DEFAULT_BASELINE

    if OVERRIDE_N_PULSES is not None:
        n_pulses = OVERRIDE_N_PULSES
    elif man_npulses is not None:
        n_pulses = man_npulses
    else:
        n_pulses = DEFAULT_N_PULSES

    # --- Compute ISI-dependent parameters ---
    isi_ms = isi * 1000.0
    margin_ms = 2.0  # Fixed margin before next event (ms)
    peak_window_ms = max(5.0, isi_ms - margin_ms)  # Use all data minus 2ms margin
    post_zoom_s = 0.3  # Post-train window (s): controls plot zoom AND last-event fit window in sequential NNLS
    pre_zoom_s = 0.20

    # --- Print configuration ---
    print("=" * 60)
    print(f"  Condition:  {condition}")
    print(f"  File:       {os.path.basename(xlsx_path)}")
    print(f"  ISI:        {isi_ms:.0f}ms ({1/isi:.0f}Hz)")
    print(f"  Baseline:   {start:.3f}s")
    print(f"  N pulses:   {n_pulses}")
    print(f"  Preset:     {PRESET_NAME}")
    print(f"  Peak win:   {peak_window_ms:.1f}ms | Post zoom: {post_zoom_s:.3f}s")
    print("=" * 60)

    # --- Build presets and select ---
    options_presets = _build_options_presets(peak_window_ms, pre_zoom_s, post_zoom_s)
    options = options_presets[PRESET_NAME]

    # `base` (the legacy_id) is provided by the task so summary rows stay keyed by
    # the original recording id even though raw files are named <uid>.csv.
    res = extract_metrics(
        time, trials,
        train_start=start,
        isi=isi,
        n_pulses=n_pulses,
        options=options,
        filename=base
    )

    # =============================================================================
    #                              RESULTS OUTPUT
    # =============================================================================

    # --- Extract amplitudes and PPR ---
    meas_keys = _measurement_series_keys(options.get('measurement', 'NNLS'))
    amp_avg_uncorr = np.asarray(
        res['average'].get(meas_keys['avg_uncorr'], res['average'].get('amp_nnls')),
        float,
    )
    amp_avg = np.asarray(
        res['average'].get(meas_keys['avg_corr'], amp_avg_uncorr),
        float,
    )
    thr_arr = np.asarray(res.get('threshold_amp1', []), float)
    thr_arr = thr_arr[np.isfinite(thr_arr)]
    thr_median = float(np.nanmedian(thr_arr)) if thr_arr.size else np.nan
    ppr_avg = _compute_ppr_from_amplitudes(amp_avg)

    print("\n--- Results ---")
    print(f"Amplitudes ({meas_keys['label']} uncorrected):", amp_avg_uncorr)
    print(f"Amplitudes ({meas_keys['label']} corrected):", amp_avg)
    print(f"PPR ({meas_keys['label']} corrected):", ppr_avg)
    print("A1 thresholds:", res['threshold_amp1'])
    print("A1 p-values:", res['pval_amp1'])

    # --- Build summary row ---
    row = {
        'measurement': meas_keys['label'],
        'ID': base,
        'condition': condition,
        'NOISE_THR_MEDIAN': thr_median,
    }
    for i, v in enumerate(amp_avg, 1):
        row[f'AMP{i}'] = float(v)
    for i, v in enumerate(amp_avg_uncorr, 1):
        row[f'AMP{i}_UNCORR'] = float(v)
    for i in range(2, len(ppr_avg) + 1):
        row[f'PPR{i}/1'] = float(ppr_avg[i - 1])

    # --- Per-trial failure counts ---
    per_trial_rows = []
    per_trial_null_rows = []
    fail_counts = {i: [0, 0] for i in range(1, 4)}
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
            'condition': condition,
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
        for p in range(1, int(n_pulses) + 1):
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
            'condition': condition,
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

    # =============================================================================
    #                              PLOTTING
    # =============================================================================

    condition_out_dir = os.path.join(OUT_DIR, condition)
    os.makedirs(condition_out_dir, exist_ok=True)

    fig = res.get('figure')
    if fig is None:
        if SAVE_PLOTS:
            print(f"[warn] No main figure returned for {base}")
    elif SAVE_PLOTS:
        _save_figure(fig, os.path.join(condition_out_dir, f"{base}_plot.png"), dpi=150, label="plot")

    figs_trials = res.get('figures_trials') or []
    for i, ftri in enumerate(figs_trials, 1):
        if SAVE_PLOTS:
            try:
                ftri.tight_layout()
                _save_figure(ftri, os.path.join(condition_out_dir, f"{base}_trialfig_{i:02d}.png"),
                             dpi=120, label=f"trial {i:02d}")
            except Exception as e:
                print(f"[warn] Failed to save trial figure {i:02d}: {e}")

    fig_param = res.get('figure_param_evolution')
    if fig_param is not None and SAVE_PLOTS:
        _save_figure(fig_param, os.path.join(condition_out_dir, f"{base}_param_evolution.png"),
                     dpi=150, label="param evolution")

    if show_plots:
        try:
            plt.show()
        except Exception as e:
            print(f"[warn] plt.show() failed: {e}")
    else:
        plt.close('all')

    return {
        'row': row,
        'per_trial_rows': per_trial_rows,
        'per_trial_null_rows': per_trial_null_rows,
        'max_pulses': len(amp_avg),
        'trace_time_s': res.get('time_s'),
        'trace_avg': res.get('average', {}).get('y_avg'),
    }


# =============================================================================
#                              RUN ANALYSIS
# =============================================================================

def _json_safe(obj):
    """Recursively convert numpy/tuple/nan values into JSON-friendly forms."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return str(obj)


def _relative_to_data_root(path):
    """Express a path relative to DATA_ROOT, else fall back to its basename.

    Keeps the run log portable: an absolute path would pin it to one machine
    and leak the local directory layout.
    """
    try:
        rel = os.path.relpath(path, DATA_ROOT)
    except ValueError:  # different drive on Windows
        return os.path.basename(path)
    return os.path.basename(path) if rel.startswith(os.pardir) else rel


def _resolved_options_for_condition(condition):
    """Reconstruct the exact options dict used for a condition (ISI-aware).

    Parameters come from the manifest (any row of this condition shares the same
    frequency / baseline / n_pulses), falling back to the defaults / overrides.
    """
    freq = base = npul = None
    for (c, _legacy), (f, b, n) in _manifest_lookup().items():
        if c == condition:
            freq, base_val, npul = f, b, n
            base = base_val
            break
    isi = OVERRIDE_ISI if OVERRIDE_ISI is not None else (1.0 / freq if freq else DEFAULT_ISI)
    start = OVERRIDE_BASELINE if OVERRIDE_BASELINE is not None else (base if base is not None else DEFAULT_BASELINE)
    n_pulses = OVERRIDE_N_PULSES if OVERRIDE_N_PULSES is not None else (npul if npul is not None else DEFAULT_N_PULSES)
    isi_ms = isi * 1000.0
    margin_ms = 2.0
    peak_window_ms = max(5.0, isi_ms - margin_ms)
    post_zoom_s = 0.3
    pre_zoom_s = 0.20
    options = _build_options_presets(peak_window_ms, pre_zoom_s, post_zoom_s)[PRESET_NAME]
    return {
        'isi_s': isi, 'baseline_s': start, 'n_pulses': n_pulses,
        'peak_window_ms': peak_window_ms, 'pre_zoom_s': pre_zoom_s, 'post_zoom_s': post_zoom_s,
        'options': options,
    }


def _write_run_log(out_dir, conditions_run, tasks, failures):
    """Write a reproducibility log (settings + environment + date) to out_dir."""
    import datetime, platform, subprocess
    try:
        import importlib.metadata as _md
    except Exception:
        _md = None

    git = {}
    try:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        git['commit'] = subprocess.check_output(
            ['git', '-C', repo, 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(
            ['git', '-C', repo, 'status', '--porcelain'], stderr=subprocess.DEVNULL).decode().strip()
        git['dirty'] = bool(dirty)
    except Exception as e:
        git['error'] = str(e)

    pkgs = {}
    if _md is not None:
        for p in ['numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn',
                  'scikit-learn', 'statsmodels', 'openpyxl']:
            try:
                pkgs[p] = _md.version(p)
            except Exception:
                pkgs[p] = None

    log = {
        'export_datetime': datetime.datetime.now().astimezone().isoformat(),
        'script': os.path.basename(__file__),
        'preset_name': PRESET_NAME,
        'data_root': os.path.basename(os.path.normpath(DATA_ROOT)),
        'out_dir': _relative_to_data_root(out_dir),
        'conditions_run': list(conditions_run),
        'file_glob': FILE_GLOB,
        'n_files': len(tasks),
        'files': [os.path.basename(t[1]) for t in tasks],
        'n_failures': len(failures),
        'overrides': {
            'OVERRIDE_ISI': OVERRIDE_ISI,
            'OVERRIDE_BASELINE': OVERRIDE_BASELINE,
            'OVERRIDE_N_PULSES': OVERRIDE_N_PULSES,
        },
        'condition_lookup': {
            'DEFAULT_ISI': DEFAULT_ISI, 'DEFAULT_BASELINE': DEFAULT_BASELINE,
            'DEFAULT_N_PULSES': DEFAULT_N_PULSES,
            # Per-condition frequency/baseline/n_pulses come from the manifest,
            # and the values actually used are in resolved_options_by_condition.
            # Recorded relative to DATA_ROOT so the log stays portable and does
            # not embed a local absolute path.
            'manifest_path': _relative_to_data_root(MANIFEST_PATH),
        },
        'resolved_options_by_condition': {
            c: _resolved_options_for_condition(c) for c in conditions_run
        },
        'environment': {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
            'packages': pkgs,
        },
        'git': git,
    }
    path = os.path.join(out_dir, 'run_settings.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_json_safe(log), f, indent=2, allow_nan=True)
    print(f"[export] Saved run settings log to: {path}")


def run_batch():
    os.makedirs(OUT_DIR, exist_ok=True)

    man = _load_manifest_df()
    if man is None:
        print(f"[error] Manifest not found ({_manifest_path()}). "
              f"Run dataset_tools/build_manifest.py or point to a release/ folder.")
        return

    rows = man
    if CONDITIONS_TO_RUN:
        rows = rows[rows["condition"].isin(CONDITIONS_TO_RUN)]

    tasks = []
    failures = []
    for r in rows.itertuples(index=False):
        raw_path = os.path.join(RAW_DIR, f"{r.uid}.csv")
        if os.path.exists(raw_path):
            tasks.append((str(r.condition), raw_path, str(r.legacy_id)))
        else:
            msg = f"[skip] Missing raw file: {raw_path}"
            print(msg)
            failures.append(msg)

    summary_rows = []
    per_trial_rows = []
    per_trial_null_rows = []
    max_pulses_seen = 0
    traces_by_condition = {}

    use_parallel = bool(PARALLEL_FILES) and len(tasks) > 1
    if use_parallel and SHOW_PLOTS:
        print("[warn] SHOW_PLOTS=True disables parallel file processing. Falling back to sequential.")
        use_parallel = False

    _acc = (failures, summary_rows, per_trial_rows, per_trial_null_rows, traces_by_condition)
    if use_parallel:
        max_workers = min(FILE_WORKERS, len(tasks))
        print(f"[info] Parallel file processing: {max_workers} worker(s) for {len(tasks)} file(s).")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_one_file, task, show_plots=False) for task in tasks]
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                except Exception as e:
                    msg = f"[skip] Worker failure: {e}"
                    print(msg)
                    failures.append(msg)
                    continue
                max_pulses_seen = _accumulate_result(result, *_acc, max_pulses_seen)
    else:
        for task in tasks:
            result = _process_one_file(task, show_plots=SHOW_PLOTS)
            max_pulses_seen = _accumulate_result(result, *_acc, max_pulses_seen)

    # =============================================================================
    #                              SUMMARY OUTPUT
    # =============================================================================

    if failures:
        log_path = os.path.join(OUT_DIR, "log.txt")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("Failed extractions:\n")
                for msg in failures:
                    f.write(str(msg).rstrip() + "\n")
            print(f"[export] Saved failure log to: {log_path}")
        except Exception as e:
            print(f"[export] Failed to write failure log: {e}")

    if summary_rows:
        df_rows = pd.DataFrame(summary_rows)
        ordered = [f'AMP{i}' for i in range(1, max_pulses_seen + 1)] \
            + [f'AMP{i}_UNCORR' for i in range(1, max_pulses_seen + 1)] \
            + [f'PPR{i}/1' for i in range(2, max_pulses_seen + 1)] \
            + [f'%Fail{i}' for i in range(1, 4)] \
            + ['NOISE_THR_MEDIAN']
        df_rows = _ensure_columns(df_rows, ['condition', 'ID', *ordered, 'measurement'])

        csv_out = os.path.join(OUT_DIR, "summary.csv")
        df_rows.to_csv(csv_out, index=False)
        xl_out = os.path.splitext(csv_out)[0] + ".xlsx"
        with pd.ExcelWriter(xl_out) as writer:
            df_rows.to_excel(writer, sheet_name="All", index=False)
            for condition in sorted(set(df_rows['condition'])):
                df_rows[df_rows['condition'] == condition].to_excel(
                    writer, sheet_name=_safe_sheet_name(condition), index=False
                )

        if per_trial_rows:
            df_trials = pd.DataFrame(per_trial_rows)
            trial_cols = (
                [
                    'condition', 'file', 'trial', 'trial_input_col_1based', 'status',
                    'F0',
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
            null_cols = [
                'condition', 'file', 'trial', 'trial_input_col_1based', 'status',
                'nnls_null_n', 'nnls_null_amps_json',
            ]
            df_trials_null = _ensure_columns(df_trials_null, null_cols)
            df_trials_null.to_excel(os.path.splitext(xl_out)[0] + "_trials_nnls_null.xlsx", index=False)

        print(f"[export] Saved summary to: {csv_out}")
        print(f"[export] Saved summary workbook to: {xl_out}")
        if per_trial_rows:
            print(f"[export] Saved per-trial file to: {os.path.splitext(xl_out)[0] + '_trials.xlsx'}")
        if per_trial_null_rows:
            print(f"[export] Saved NNLS-null per-trial file to: {os.path.splitext(xl_out)[0] + '_trials_nnls_null.xlsx'}")
        if traces_by_condition:
            out_stem = os.path.join(OUT_DIR, "summary")
            traces_file = f"{out_stem}_traces.xlsx"
            times_file = f"{out_stem}_times.xlsx"
            with pd.ExcelWriter(traces_file) as trace_writer, pd.ExcelWriter(times_file) as time_writer:
                wrote_any = False
                for condition, id_traces in sorted(traces_by_condition.items()):
                    if not id_traces:
                        continue
                    trace_dict = {}
                    time_dict = {}
                    for bid, (t_vec, y_avg) in sorted(id_traces.items()):
                        trace_dict[bid] = pd.Series(np.asarray(y_avg, float))
                        time_dict[bid] = pd.Series(np.asarray(t_vec, float))
                    pd.DataFrame(trace_dict).to_excel(trace_writer, sheet_name=_safe_sheet_name(condition), index=False)
                    pd.DataFrame(time_dict).to_excel(time_writer, sheet_name=_safe_sheet_name(condition), index=False)
                    wrote_any = True
                if wrote_any:
                    print(f"[export] Saved summary traces to: {traces_file}")
                    print(f"[export] Saved summary times to: {times_file}")
    else:
        print("[export] No files processed; no summary written.")

    # --- Reproducibility log (settings + environment + export date) ---
    try:
        conditions_with_files = sorted(set(t[0] for t in tasks))
        _write_run_log(OUT_DIR, conditions_with_files, tasks, failures)
    except Exception as e:
        print(f"[export] Failed to write run settings log: {type(e).__name__}: {e}")
        traceback.print_exc()

    # --- Consolidated tidy tables (Zenodo-ready, flat) into RELEASE_DIR ---
    if WRITE_TIDY:
        try:
            _tools = os.path.join(REPO_ROOT, "dataset_tools")
            if _tools not in sys.path:
                sys.path.insert(0, _tools)
            from consolidate import consolidate
            consolidate(OUT_DIR, _manifest_path(), RELEASE_DIR)
            # keep the provenance log alongside the release tables
            import shutil
            _rs = os.path.join(OUT_DIR, "run_settings.json")
            if os.path.exists(_rs):
                shutil.copyfile(_rs, os.path.join(RELEASE_DIR, "run_settings.json"))
        except Exception as e:
            print(f"[export] Failed to write tidy tables: {e}")


if __name__ == "__main__":
    run_batch()
