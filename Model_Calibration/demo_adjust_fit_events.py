import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure the repository root is on sys.path when running from this subfolder
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Feature_extraction.extract_metrics import extract_metrics
from smoothing import build_median_recut_waveform
from dataset_tools import raw_loader

# Configuration parameters
TRAIN_START_S = 0.5
ISI_S = 0.05
N_PULSES = 10
PRE_MS = 5.0
POST_MS = 50.0
SAMPLE_HZ = 1000.0

# Processing settings
MAX_WORKERS = 24
EVENT_INDEX = None  # Set to 0 for event 1, 2 for event 3, etc. Use None for all events

# Input condition handling
# Recordings are loaded from the well-organized raw stage (release/raw/<uid>.csv +
# release/boutons_manifest.csv) via dataset_tools.raw_loader, filtered by condition -
# never from the private per-condition .xlsx tree.
# You can override the default condition in three ways (precedence high→low):
#  1. Pass a condition name as the first CLI argument when running this script
#        python demo_adjust_fit_events.py Theo_4Ca
#  2. Set environment variable GLUSNFR_IN_CONDITION
#        (Windows) set GLUSNFR_IN_CONDITION=Theo_4Ca
#  3. Set GLUSNFR_DATA_ROOT, or use the default local publication release root
DATA_ROOT = os.path.abspath(os.environ.get(
    "GLUSNFR_DATA_ROOT", os.path.join(os.path.expanduser("~"), "Desktop", "FINAL PUBLICATION TEST PROCESS")
))
DEFAULT_CONDITION = "Theo_1_5Ca"

def _resolve_condition_from_argv(argv) -> str | None:
    """Return the first positional CLI argument that isn't a flag.

    Skips any arguments starting with '-' (e.g., Jupyter's '--f=kernel.json').
    """
    for a in argv[1:]:  # skip script name
        if not a or a.startswith('-'):
            continue
        return a
    return None

def _resolve_condition(cli_arg: str | None) -> str:
    # Prefer an explicitly passed condition
    if cli_arg and not cli_arg.startswith('-'):
        return cli_arg
    # Scan remaining argv (handles Jupyter invocation)
    scan = _resolve_condition_from_argv(sys.argv)
    if scan:
        return scan
    # Environment variable override
    env_cond = os.environ.get("GLUSNFR_IN_CONDITION")
    if env_cond:
        return env_cond
    # Fallback default
    return DEFAULT_CONDITION


def _clean_trials(X: np.ndarray) -> np.ndarray:
    """Remove trials that are entirely NaN."""
    if X is None or X.size == 0:
        return X
    col_ok = ~np.all(~np.isfinite(X), axis=0)
    return X[:, col_ok]


def _file_to_resampled_trace(uid: str, t_grid: np.ndarray):
    """Process a single recording (release/raw/<uid>.csv) and return resampled trace."""
    try:
        raw = raw_loader.load_raw_csv(DATA_ROOT, uid)
        ok = np.isfinite(raw.time_s)
        t, X = raw.time_s[ok], raw.values[ok, :]
        X = _clean_trials(X)
        
        # Extract metrics from the data
        res = extract_metrics(
            t, X, train_start=TRAIN_START_S, isi=ISI_S, n_pulses=N_PULSES,
            options={'normalize_dff': False, 'bleach': False, 'plot': {'enabled': False}}
        )
        
        # Use processed trials from extract_metrics
        processed_trials = np.column_stack([trial_data['y_proc'] for trial_data in res['per_trial']])
        
        # Build median recut waveform
        stim = res['stim_times_s']
        if EVENT_INDEX is not None and len(stim) > EVENT_INDEX:
            stim_selected = [stim[EVENT_INDEX]]
            print(f"  Using event {EVENT_INDEX + 1} only for median calculation")
        else:
            stim_selected = stim
            print(f"  Using all {len(stim)} stimulus events (events 1-{len(stim)}) for median calculation")
        
        t_rel, med = build_median_recut_waveform(
            res['time_s'], processed_trials, np.asarray(stim_selected),
            pre_ms=PRE_MS, 
            post_ms=POST_MS, 
            peak_recenter=0
        )

        if t_rel is None or med is None:
            return None
        # Fill NaNs in the median waveform to avoid interp errors when resampling
        try:
            t_rel = np.asarray(t_rel, float)
            med = np.asarray(med, float)
            if np.any(~np.isfinite(med)):
                finite = np.isfinite(med)
                if finite.any():
                    med = np.interp(t_rel, t_rel[finite], med[finite])
                else:
                    return None
        except Exception:
            return None
            
        # Resample to grid
        r = np.full_like(t_grid, np.nan, dtype=float)
        m = (t_grid >= t_rel[0]) & (t_grid <= t_rel[-1])
        if np.any(m):
            r[m] = np.interp(t_grid[m], t_rel, med)
        return r
        
    except Exception as e:
        print(f"Error processing {uid}: {e}")
        return None


def process_folder(condition: str, max_workers: int = None):
    """Process all recordings for a condition and create summary plot."""

    # Create time grid for resampling
    t_grid = np.arange(
        -PRE_MS/1000.0,
        POST_MS/1000.0 + 1e-12,
        1.0/SAMPLE_HZ
    )

    # Find all recordings for this condition via the manifest
    manifest = raw_loader.load_manifest(DATA_ROOT)
    rows = raw_loader.iter_condition_rows(manifest, [condition])
    uids = list(rows["uid"])
    print(f"Found {len(uids)} recordings in manifest for condition '{condition}'")

    if not uids:
        print("No recordings found for this condition!")
        return None

    # Process files in parallel using threads
    traces_resampled = []
    workers = max_workers or MAX_WORKERS

    print(f"Processing files with {workers} workers...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all jobs
        future_to_file = {
            executor.submit(_file_to_resampled_trace, uid, t_grid): uid
            for uid in uids
        }

        # Collect results
        for future in as_completed(future_to_file):
            uid = future_to_file[future]
            result = future.result()

            if result is not None and np.any(np.isfinite(result)):
                traces_resampled.append(result)
                print(f"✓ {uid} -> Total: {len(traces_resampled)}")
            else:
                print(f"✗ Skipped {uid}")
    
    if not traces_resampled:
        print("No valid traces found!")
        return None
    
    # Calculate average across all files
    S = np.vstack(traces_resampled)
    avg_all = np.nanmean(S, axis=0)
    
    print(f"\nSuccessfully processed {len(traces_resampled)} files")
    print(f"Time grid: {len(t_grid)} points from {t_grid[0]*1000:.1f} to {t_grid[-1]*1000:.1f} ms")
    
    # Create summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot individual traces
    for i, trace in enumerate(traces_resampled):
        ax.plot(t_grid * 1000.0, trace, color='lightgray', alpha=0.5, linewidth=0.8)
    
    # Plot average
    ax.plot(t_grid * 1000.0, avg_all, color='red', linewidth=2.0, label=f'Average (N={len(traces_resampled)})')
    
    # Add stimulus time marker
    ax.axvline(0.0, color='black', linestyle='--', linewidth=1.0, label='Stimulus')
    
    # Formatting
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('ΔF (median)')
    title = f'Batch Processing Results: {len(traces_resampled)} files'
    if EVENT_INDEX is not None:
        title += f' - Event {EVENT_INDEX + 1} only'
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'traces': traces_resampled,
        'time_grid': t_grid,
        'average': avg_all,
        'n_files': len(traces_resampled),
        'figure': fig,
        'axis': ax
    }


if __name__ == "__main__":
    # Resolve input condition (CLI arg > env var > default)
    cli_cond = sys.argv[1] if len(sys.argv) > 1 else None
    IN_CONDITION = _resolve_condition(cli_cond)
    print(f"Using condition: {IN_CONDITION}")
    results = process_folder(IN_CONDITION, max_workers=MAX_WORKERS)
    
    if results:
        print(f"\nProcessing complete! Processed {results['n_files']} files.")
        print("Results saved to 'results' variable.")
    else:
        print("Processing failed - no valid data found.")
