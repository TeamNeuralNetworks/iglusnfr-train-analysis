import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Feature_extraction.extract_metrics import extract_metrics


def _default_data_root():
    candidates = (
        Path.home() / "Desktop" / "FINAL PUBLICATION TEST PROCESS",
        REPO_ROOT / "PPR_DATA_FINAL",
        Path.home() / "Desktop" / "Publication_test",
    )
    for candidate in candidates:
        if (candidate / "release" / "boutons.csv").is_file():
            return candidate
    return candidates[0]


DATA_ROOT = Path(os.environ.get("GLUSNFR_DATA_ROOT", str(_default_data_root())))
RELEASE_DIR = DATA_ROOT / "release"
MANIFEST = RELEASE_DIR / "boutons.csv"
PRESET_NAME = "iglusnfr_optimized"
DEFAULT_CONDITION = "Theo_1_5Ca"
DEFAULT_TARGET_FILE = "20200909_linescan1_20Hz_10pulses_1.5mMCa_bouton4_traces_converted.xlsx"
DEFAULT_TRIAL_INPUT_COL_1BASED = 1

DEFAULT_ISI = 0.05
DEFAULT_BASELINE = 0.998
DEFAULT_N_PULSES = 10

ISI_BY_CONDITION = {
    "Theo_1_5_50Hz": 0.02,
    "Theo_2_5_50Hz": 0.02,
    "Theo_4_50Hz": 0.02,
}

BASELINE_BY_CONDITION = {
    "Stability_Before_05": 0.498,
    "Stability_After_05": 0.498,
    "Theo_4Ca": 0.498,
    "Theo_1_5Ca": 0.498,
    "WT_Theo": 0.498,
    "Theo_1_5_50Hz": 0.498,
    "Theo_2_5_50Hz": 0.498,
    "Theo_4_50Hz": 0.498,
}


def build_options_presets(peak_window_ms, pre_zoom_s, post_zoom_s):
    return {
        "iglusnfr_optimized": {
            "normalize_dff": True,
            "bleach": True,
            "sg_window": 9,
            "sg_poly": 2,
            "fit_source": "global",
            "decay_progression_mode": "linear",
            "anchor_final_tau": False,
            "anchor_first_tau": False,
            "event_model": "iglusnfr_tri",
            "parameter_bounds": {
                "tau_decay_fast": (0.003, 0.008),
                "tau_decay_slow": (0.008, 0.035),
                "tau_superslow": (0.035, 0.090),
                "amplitude_ratio": (0.0, 1.0),
            },
            "early_events_only": 0,
            "recut_projection": "mean",
            "recut_oversample": 20,
            "recut_peak_recenter": 0,
            "recut_snippets": True,
            "onset_method": "baseline_threshold",
            "onset_baseline_threshold": 0.10,
            "amplitude_floor_to_noise": True,
            "average_amplitude_floor_to_noise": False,
            "average_null_N": None,
            "nnls_weight_mode": "savgol",
            "nnls_weight_tau_s": None,
            "nnls_peak_window_s": 0.010,
            "nnls_peak_weight": 3.0,
            "fit_diagnostic_plot": False,
            "huber_delta": 2.5,
            "irls_iters": 20,
            "nnls_last_event_tail_tau_s": "best",
            "pre_zoom_s": pre_zoom_s,
            "post_zoom_s": post_zoom_s,
            "f0_window_s": 1.0,
            "peak_window_ms": peak_window_ms,
            "peak_avg_points": 3,
            "pre_peak_ms": 1.0,
            "measurement": "NNLS",
            "fail_method": "SAVGOL",
            "threshold_mode": "auto",
            "null_N": 1.0,
            "null_sim_max_points": 1000,
            "null_min_post_zoom_s": 0.05,
            "bleach_huber_delta": 3.0,
            "bleach_tau_range_factor": (0.25, 4.0),
            "bleach_n_tau": 25,
            "template_variant_select": "soft",
            "plot": {
                "enabled": False,
                "traces": ["raw", "nnls"],
                "figsize": (10, 6),
                "show_decay": False,
                "show_onsets": False,
                "trials": False,
                "baseline": False,
                "residuals": False,
                "nnls_residual": False,
                "nnls_n_minus_1": False,
                "plot_peaks_details": False,
                "param_evolution": False,
            },
            "use_template_variants": True,
            "template_variant_ratios": np.linspace(0.1, 0.9, 10),
            "template_variant_superslow_fracs": np.linspace(0.1, 0.9, 11),
            "superslow_min_ratio": 1.0,
            "allow_tau_slow_override": True,
            "force_tau_slow_override": False,
            "jitter_variant_ms": np.linspace(-1.0, 1.0, 5),
        },
    }


def _strip_xlsx(name):
    """Drop a trailing .xlsx only (NOT Path.stem, which mangles dotted ids like 2.5mMCa)."""
    name = str(name)
    return name[:-5] if name.lower().endswith(".xlsx") else name


def resolve_raw(condition, target_file):
    """Resolve a bouton to its release raw csv + params via the manifest.
    Returns (raw_csv_path, freq_hz, baseline_s, n_pulses) or None."""
    stem = _strip_xlsx(target_file)
    try:
        m = pd.read_csv(MANIFEST)
    except Exception:
        return None
    norm = stem.replace("_traces_converted", "")
    key = m["legacy_id"].astype(str).str.replace("_traces_converted", "", regex=False)
    row = m[(m["condition"].astype(str) == str(condition)) & (key == norm)]
    if not len(row):
        return None
    r = row.iloc[0]
    return (RELEASE_DIR / "raw" / f"{r['uid']}.csv",
            float(r["frequency_hz"]), float(r["baseline_s"]), int(r["n_pulses"]))


def load_trials_table(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        # release layout: time_s, trial_*, average. round_trip avoids ULP drift.
        df = pd.read_csv(path, float_precision="round_trip")
        time_col = "time_s" if "time_s" in df.columns else df.columns[-1]
        other_cols = [c for c in df.columns if c != time_col]
        time_raw = pd.to_numeric(df[time_col], errors="coerce").to_numpy(float)
        all_before_time = df[other_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    else:
        df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
        time_raw = pd.to_numeric(df.iloc[:, -1], errors="coerce").to_numpy(float)
        all_before_time = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    has_avg_col = False
    if all_before_time.shape[1] >= 2:
        candidate_avg = all_before_time[:, -1]
        preceding = all_before_time[:, :-1]
        computed_avg = np.nanmean(preceding, axis=1)
        finite = np.isfinite(candidate_avg) & np.isfinite(computed_avg)
        if finite.sum() > 10:
            corr = np.corrcoef(candidate_avg[finite], computed_avg[finite])[0, 1]
            if corr > 0.99:
                has_avg_col = True

    trials = all_before_time[:, :-1] if has_avg_col else all_before_time
    valid = np.isfinite(time_raw)
    return time_raw[valid], trials[valid, :]


def sanitize_token(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def default_out_file(condition, target_file, trial_input_col_1based):
    stem = Path(target_file).stem
    name = (
        f"_tmp_nnls_overlay__{sanitize_token(condition)}__"
        f"{sanitize_token(stem)}__trial{int(trial_input_col_1based)}.npz"
    )
    return REPO_ROOT / name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default=DEFAULT_CONDITION)
    parser.add_argument("--file", dest="target_file", default=DEFAULT_TARGET_FILE)
    parser.add_argument("--trial", dest="trial_input_col_1based", type=int, default=DEFAULT_TRIAL_INPUT_COL_1BASED)
    parser.add_argument("--out", dest="out_file", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    condition = str(args.condition)
    target_file = str(args.target_file)
    trial_input_col_1based = int(args.trial_input_col_1based)
    out_file = Path(args.out_file) if args.out_file else default_out_file(condition, target_file, trial_input_col_1based)

    _rel = resolve_raw(condition, target_file)
    if _rel is not None:
        raw_path, _freq, train_start, n_pulses = _rel
        isi = 1.0 / _freq
    else:
        raw_path = DATA_ROOT / condition / target_file
        isi = ISI_BY_CONDITION.get(condition, DEFAULT_ISI)
        train_start = BASELINE_BY_CONDITION.get(condition, DEFAULT_BASELINE)
        n_pulses = DEFAULT_N_PULSES

    isi_ms = isi * 1000.0
    margin_ms = 2.0
    peak_window_ms = max(5.0, isi_ms - margin_ms)
    post_zoom_s = 0.3
    pre_zoom_s = 0.20

    options = build_options_presets(peak_window_ms, pre_zoom_s, post_zoom_s)[PRESET_NAME]

    time_s, trials = load_trials_table(raw_path)
    base = _strip_xlsx(target_file)

    res = extract_metrics(
        time_s,
        trials,
        train_start=train_start,
        isi=isi,
        n_pulses=n_pulses,
        options=options,
        filename=base,
    )

    selected_trial = None
    for row in res.get("per_trial", []):
        if int(row.get("trial_input_col_1based", -1)) == trial_input_col_1based:
            selected_trial = row
            break
    if selected_trial is None:
        raise RuntimeError(f"Trial {trial_input_col_1based} not found in extract_metrics output")

    variant_info = res.get("variant_info") or {}
    amplitudes_3d = variant_info.get("amplitudes_3d")
    template_variant_sums = variant_info.get("template_variant_sums")

    np.savez(
        out_file,
        condition=condition,
        target_file=target_file,
        file_stem=base,
        time_s=np.asarray(res["time_s"], float),
        trial_input_col_1based=trial_input_col_1based,
        train_start_s=float(train_start),
        isi_s=float(isi),
        tau_r_s=float(res.get("tau_r_s", np.nan)),
        tau_d_s=np.asarray(res.get("tau_d_s", []), float),
        selected_trial_yproc=np.asarray(selected_trial.get("y_proc", []), float),
        selected_trial_yhat=np.asarray(selected_trial["yhat"], float),
        selected_trial_amp_savgol=np.asarray(selected_trial.get("amp_savgol", []), float),
        selected_trial_amp_savgol_corr=np.asarray(selected_trial.get("amp_savgol_corr", []), float),
        selected_trial_thr_shared=float(selected_trial.get("thr_shared", np.nan)),
        selected_trial_noise_level=float(selected_trial.get("noise_level", np.nan)),
        selected_trial_null_amps_nnls=np.asarray(selected_trial.get("null_amps_nnls", []), float),
        all_trial_input_col_1based=np.asarray(
            [int(row.get("trial_input_col_1based", i + 1)) for i, row in enumerate(res.get("per_trial", []))],
            int,
        ),
        all_trial_yproc=np.asarray(
            [np.asarray(row.get("y_proc", []), float) for row in res.get("per_trial", [])],
            float,
        ),
        all_trial_yhat=np.asarray(
            [np.asarray(row.get("yhat", []), float) for row in res.get("per_trial", [])],
            float,
        ),
        all_trial_null_amps_nnls=np.asarray(
            [np.asarray(row.get("null_amps_nnls", []), float) for row in res.get("per_trial", [])],
            dtype=object,
        ),
        all_trial_thr_shared=np.asarray(
            [float(row.get("thr_shared", np.nan)) for row in res.get("per_trial", [])],
            float,
        ),
        all_trial_noise_level=np.asarray(
            [float(row.get("noise_level", np.nan)) for row in res.get("per_trial", [])],
            float,
        ),
        average_yhat=np.asarray(res["average"]["yhat_avg"], float),
        average_yproc=np.asarray(res["average"]["y_avg"], float),
        average_amp_savgol=np.asarray(res["average"].get("amp_savgol", []), float),
        average_amp_savgol_corr=np.asarray(res["average"].get("amp_savgol_corr", []), float),
        average_amp_nnls=np.asarray(res["average"].get("amp_nnls", []), float),
        average_amp_nnls_corr=np.asarray(res["average"]["amp_nnls_corr"], float),
        average_ppr_nnls_corr=np.asarray(res["average"]["ppr_nnls_corr"], float),
        recut_t_rel=np.asarray(res.get("recut_t_rel", []), float),
        recut_avg=np.asarray(res.get("recut_avg", []), float),
        threshold_amp1=np.asarray(res.get("threshold_amp1", []), float),
        median_threshold_floor=float(res.get("median_threshold_floor", np.nan)),
        variant_ratios=np.asarray(variant_info.get("variant_ratios", []), dtype=object),
        jitter_variant_ms=np.asarray(variant_info.get("jitter_variant_ms", []), float),
        dominant_template_ratio=np.asarray(variant_info.get("dominant_template_ratio", []), dtype=object),
        dominant_jitter_ms=np.asarray(variant_info.get("dominant_jitter_ms", []), float),
        template_variant_sums=np.asarray(template_variant_sums, float) if template_variant_sums is not None else np.asarray([], float),
        amplitudes_3d=np.asarray(amplitudes_3d, float) if amplitudes_3d is not None else np.asarray([], float),
    )

    print(f"saved={out_file}")
    print(f"condition={condition}")
    print(f"target_file={target_file}")
    print(f"trial_input_col_1based={trial_input_col_1based}")
    print(f"time_points={len(res['time_s'])}")


if __name__ == "__main__":
    main()
