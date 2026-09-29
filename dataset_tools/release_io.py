"""
release_io.py - Read the consolidated release/ folder and reshape it back into the
exact dataframes the Support_figure notebook used to load from the old summary_*.xlsx
files. This lets the notebook run off the new design while keeping every downstream
grouping/normalisation (which operates on `legacy_id`) untouched.

Design notes (validated against the pre-migration outputs):
  * Every table is re-keyed by `legacy_id` (== the identifiers the old files used),
    so BaseID/FiberID/PairBaseID normalisation downstream is byte-identical.
  * `target` is taken from the manifest (proven to match the old join exactly on the
    PCA cohort).
  * `sex` and `date` are taken from the manifest too (deliberately MORE complete than
    the old fragile string joins).

Usage in the notebook:
    import sys; sys.path.insert(0, str(REPO_ROOT / "dataset_tools"))
    import release_io as rio
    rio.RELEASE_DIR = BASE_DIR / "release"
    features = rio.features()
"""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

RELEASE_DIR = None  # set by the caller (e.g. BASE_DIR / "release")

_METRIC_PREFIXES = ("AMP", "PPR", "%Fail")
_METRIC_EXTRA = ("NOISE_THR_MEDIAN", "measurement")


def _dir():
    global RELEASE_DIR
    if RELEASE_DIR is not None and (Path(RELEASE_DIR) / "boutons.csv").is_file():
        return str(RELEASE_DIR)

    candidates = []
    env_root = os.environ.get("GLUSNFR_DATA_ROOT")
    if env_root:
        candidates.append(Path(env_root) / "release")
    candidates.extend([
        Path.home() / "Desktop" / "FINAL PUBLICATION TEST PROCESS" / "release",
        Path.cwd() / "PPR_DATA_FINAL" / "release",
        Path.home() / "Desktop" / "Publication_test" / "release",
    ])
    for candidate in candidates:
        if (candidate / "boutons.csv").is_file():
            RELEASE_DIR = candidate
            return str(candidate)

    raise FileNotFoundError("Could not find release/boutons.csv. Set GLUSNFR_DATA_ROOT to the publication data root.")


def manifest():
    return pd.read_csv(os.path.join(_dir(), "boutons.csv"))


def _uid_to_legacy():
    m = manifest()
    return dict(zip(m["uid"].astype(str), m["legacy_id"].astype(str)))


def features():
    """Old summary features: columns [condition, ID(=legacy_id), <metric columns>]."""
    m = manifest()
    metric_cols = [c for c in m.columns
                   if c.startswith(_METRIC_PREFIXES) or c in _METRIC_EXTRA]
    out = m[["condition", "legacy_id", *metric_cols]].rename(columns={"legacy_id": "ID"})
    return out.copy()


def trials():
    """Old summary_trials: per-(bouton,trial) with a 'file' column (= legacy_id)."""
    df = pd.read_csv(os.path.join(_dir(), "trials.csv"))
    df = df.rename(columns={"legacy_id": "file"})
    return df.drop(columns=[c for c in ["uid"] if c in df.columns])


def null_amps():
    """Old summary_trials_nnls_null: per-(bouton,trial), 'file' = legacy_id."""
    df = pd.read_csv(os.path.join(_dir(), "null_amps.csv"))
    df = df.rename(columns={"legacy_id": "file"})
    return df.drop(columns=[c for c in ["uid"] if c in df.columns])


def traces_by_condition():
    """Return (traces_dict, times_dict): {condition: wide DataFrame with one column
    per legacy_id}, matching the old summary_traces.xlsx / summary_times.xlsx sheets."""
    long = pd.read_csv(os.path.join(_dir(), "traces.csv"))
    u2l = _uid_to_legacy()
    long = long.copy()
    long["legacy_id"] = long["uid"].astype(str).map(u2l)
    traces, times = {}, {}
    for cond, g in long.groupby("condition", sort=False):
        tr, ti = {}, {}
        for legacy, gg in g.groupby("legacy_id", sort=False):
            tr[legacy] = pd.Series(gg["dff"].to_numpy(float))
            ti[legacy] = pd.Series(gg["time_s"].to_numpy(float))
        traces[cond] = pd.DataFrame(tr)
        times[cond] = pd.DataFrame(ti)
    return traces, times


def target_mapping():
    """Old Target_WT_pooled first two columns: [ID(=legacy_id), Target].
    One row per physical bouton: a legacy_id recorded in several conditions repeats
    in the manifest, so we de-duplicate on the normalised id to keep the historical
    map's 1-row-per-bouton shape (otherwise the notebook's merge multiplies rows)."""
    m = manifest()
    out = m[["legacy_id", "target"]].rename(columns={"legacy_id": "ID", "target": "Target"}).copy()
    _norm = out["ID"].astype(str).str.replace("_traces_converted", "", regex=False)
    out = out.loc[~_norm.duplicated()].reset_index(drop=True)
    return out


def sex_map():
    """Adopted (more complete) sex: [ID(=legacy_id, suffix stripped), Sexe]."""
    m = manifest()
    out = m[["legacy_id", "sex"]].copy()
    out["ID"] = out["legacy_id"].astype(str).str.replace("_traces_converted", "", regex=False)
    out = out.rename(columns={"sex": "Sexe"})[["ID", "Sexe"]]
    return out.dropna(subset=["ID"]).drop_duplicates(subset=["ID"])


def date_map():
    """Adopted (more complete) recording date: {stripped legacy_id -> 'yyyymmdd'}."""
    m = manifest()
    key = m["legacy_id"].astype(str).str.replace("_traces_converted", "", regex=False)
    return dict(zip(key, m["date"].astype("Int64").astype(str)))


def saturation():
    """Supplementary saturation experiment: (amps_df, traces_long_df)."""
    amps = pd.read_csv(os.path.join(_dir(), "saturation_amps.csv"))
    traces = pd.read_csv(os.path.join(_dir(), "saturation_traces.csv"))
    return amps, traces


def saturation_sheet(sheet_name):
    """A verbatim saturation sheet (e.g. '2.5mMCa_traces'), matching the old
    Saturation_data.xlsx layout, as release/saturation/<sheet>.csv."""
    return pd.read_csv(os.path.join(_dir(), "saturation", f"{sheet_name}.csv"))


# --- Per-bouton raw / F0 / reconstructed dF/F0 --------------------------------

def _legacy_to_uid():
    m = manifest()
    d = {}
    for c, l, u in m[["condition", "legacy_id", "uid"]].itertuples(index=False):
        d[(str(c), str(l))] = str(u)
        d[(str(c), str(l).replace("_traces_converted", ""))] = str(u)  # normalised key
    return d


def _uid_for(condition, legacy_id):
    return _legacy_to_uid().get((str(condition), str(legacy_id)))


def f0():
    """Per-(uid, trial) baseline F0 scalars."""
    return pd.read_csv(os.path.join(_dir(), "f0.csv"))


def raw_exists(condition, legacy_id):
    uid = _uid_for(condition, legacy_id)
    return uid is not None and os.path.exists(os.path.join(_dir(), "raw", uid + ".csv"))


def dff_traces(condition, legacy_id):
    """Per-trial dF/F0 traces for a bouton, reconstructed as (raw - F0)/F0.
    Returns a DataFrame with the historical 'Traces DF_F0' columns:
    'Time', one column per trial label, and 'Average' (or None if the bouton
    is missing)."""
    uid = _uid_for(condition, legacy_id)
    if uid is None:
        return None
    raw_path = os.path.join(_dir(), "raw", uid + ".csv")
    if not os.path.exists(raw_path):
        return None
    raw = pd.read_csv(raw_path, float_precision="round_trip")
    ft = f0(); ft = ft[ft["uid"] == uid]
    f0map = dict(zip(ft["trial_col"].astype(str), ft["F0"].astype(float)))
    trial_cols = [c for c in raw.columns if str(c).startswith("trial_")]
    out = {}
    if "time_s" in raw.columns:
        out["Time"] = raw["time_s"].to_numpy(float)
    labels = []
    for tc in trial_cols:
        label = str(tc)[len("trial_"):]
        F0v = f0map.get(str(tc))
        out[label] = (raw[tc].to_numpy(float) - F0v) / F0v if F0v else np.nan
        labels.append(label)
    df = pd.DataFrame(out)
    # 'Average' matches the historical sheet: (raw average - mean F0) / mean F0.
    if labels and "average" in raw.columns:
        mF0 = float(np.nanmean(list(f0map.values()))) if f0map else np.nan
        df["Average"] = (raw["average"].to_numpy(float) - mF0) / mF0
    elif labels:
        df["Average"] = df[labels].mean(axis=1)
    return df


def resolve_legacy(condition, bouton_id):
    """Return the legacy_id (raw file stem) for a bouton, matching with or without
    the _traces_converted suffix. None if unknown."""
    m = manifest()
    norm = str(bouton_id).replace("_traces_converted", "")
    key = m["legacy_id"].astype(str).str.replace("_traces_converted", "", regex=False)
    row = m[(m["condition"].astype(str) == str(condition)) & (key == norm)]
    return str(row.iloc[0]["legacy_id"]) if len(row) else None


def f0_mean(condition, legacy_id):
    """Mean baseline F0 across trials for a bouton (np.nan if unknown)."""
    uid = _uid_for(condition, legacy_id)
    if uid is None:
        return np.nan
    ft = f0(); vals = ft[ft["uid"] == uid]["F0"]
    return float(vals.mean()) if len(vals) else np.nan
