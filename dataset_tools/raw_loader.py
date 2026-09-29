"""raw_loader.py - Shared Stage-2 reader: release/raw/<uid>.csv + the metrics-free
manifest (release/boutons_manifest.csv). Used by the Feature_extraction/
Model_Calibration scripts instead of each one re-implementing manifest lookups and
raw-file loading.

Distinct from release_io.py: that module is the Stage-3 reader for the notebook (it
requires the fully curated release/ tables, metrics included). This module only
needs Stage 2 (raw traces + the metrics-free manifest), so it also works before any
extraction has been run.

Column semantics of release/raw/<uid>.csv (written by reorganize_raw.py): time_s,
then every other original column renamed trial_<label> except one literally named
"average" when present. Whether that "average" column should be dropped before
fitting is NOT decided here - callers differ (see demo_batch_process.py's
correlation-sniff, kept only in demo_batch_process.py and demo_single_file.py) and
this loader deliberately returns every non-time column unmodified so each caller can
keep applying its own existing logic.
"""
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def manifest_path(data_root: str) -> str:
    """release/boutons_manifest.csv if present, else release/boutons.csv (the
    metrics-enriched table also works as a manifest - it's a superset of the
    columns needed here)."""
    release_dir = os.path.join(data_root, "release")
    primary = os.path.join(release_dir, "boutons_manifest.csv")
    if os.path.exists(primary):
        return primary
    return os.path.join(release_dir, "boutons.csv")


def load_manifest(data_root: str) -> pd.DataFrame:
    path = manifest_path(data_root)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Manifest not found ({path}). Run "
            f"'python dataset_tools/build_manifest.py --data-root {data_root}' first."
        )
    return pd.read_csv(path)


def resolve_recording(manifest: pd.DataFrame, *, uid: str = None,
                       legacy_id: str = None, condition: str = None) -> pd.Series:
    """Return exactly one manifest row matching uid, or (legacy_id, condition)."""
    if uid is not None:
        rows = manifest[manifest["uid"] == uid]
        if len(rows) == 1:
            return rows.iloc[0]
        if len(rows) == 0:
            near = manifest.loc[
                manifest["uid"].astype(str).str.contains(str(uid), case=False, na=False),
                "uid",
            ].tolist()[:10]
            raise KeyError(f"No manifest row for uid={uid!r}. Near matches: {near}")
        raise KeyError(f"uid={uid!r} matched {len(rows)} manifest rows (expected 1)")

    if legacy_id is None:
        raise ValueError("Provide uid, or legacy_id (optionally with condition).")

    rows = manifest[manifest["legacy_id"] == legacy_id]
    if condition is not None:
        rows = rows[rows["condition"] == condition]
    if len(rows) == 1:
        return rows.iloc[0]
    if len(rows) == 0:
        near = manifest.loc[
            manifest["legacy_id"].astype(str).str.contains(str(legacy_id), case=False, na=False),
            "legacy_id",
        ].drop_duplicates().tolist()[:10]
        raise KeyError(
            f"No manifest row for legacy_id={legacy_id!r}"
            f"{f', condition={condition!r}' if condition else ''}. Near matches: {near}"
        )
    conditions = rows["condition"].tolist()
    raise KeyError(
        f"legacy_id={legacy_id!r} is ambiguous across conditions {conditions}; "
        f"pass condition= to disambiguate (matches consolidate.py's (condition, legacy_id) key)."
    )


class RawTrace:
    """time_s + every other release/raw/<uid>.csv column as loaded, unmodified."""

    def __init__(self, time_s: np.ndarray, columns: List[str], values: np.ndarray):
        self.time_s = time_s
        self.columns = columns          # names, in file order (may include "average")
        self.values = values            # (n_samples, len(columns))

    def has_column(self, name: str) -> bool:
        return name in self.columns


def load_raw_csv(data_root: str, uid: str) -> RawTrace:
    """Read release/raw/<uid>.csv. float_precision='round_trip' is REQUIRED: the
    default C parser loses ~1 ULP, which flips discrete template-variant choices in
    the sequential NNLS for boundary boutons and perturbs amplitudes (see the same
    note in Feature_extraction/demo_batch_process.py's CSV-reading branch)."""
    path = os.path.join(data_root, "release", "raw", f"{uid}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing organized-raw file: {path}")
    df = pd.read_csv(path, float_precision="round_trip")
    time_col = "time_s" if "time_s" in df.columns else df.columns[-1]
    other_cols = [c for c in df.columns if c != time_col]
    time_s = pd.to_numeric(df[time_col], errors="coerce").to_numpy(float)
    values = df[other_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return RawTrace(time_s=time_s, columns=other_cols, values=values)


def resolved_params(row: pd.Series, *, override_isi: Optional[float] = None,
                     override_baseline: Optional[float] = None,
                     override_n_pulses: Optional[int] = None,
                     default_isi: float = 0.05, default_baseline: float = 0.998,
                     default_n_pulses: int = 10) -> Dict:
    """override > manifest row > default. The manifest has no direct isi column -
    isi is derived from frequency_hz (1/freq), matching
    Feature_extraction/demo_batch_process.py's existing precedence exactly."""
    def _row_value(key):
        val = row.get(key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return val

    freq = _row_value("frequency_hz")
    isi = override_isi if override_isi is not None else (
        1.0 / float(freq) if freq else default_isi)
    baseline = override_baseline if override_baseline is not None else (
        _row_value("baseline_s") or default_baseline)
    n_pulses = override_n_pulses if override_n_pulses is not None else (
        _row_value("n_pulses") or default_n_pulses)

    return {
        "isi": float(isi),
        "baseline_s": float(baseline),
        "n_pulses": int(n_pulses),
        "condition": str(row.get("condition")),
    }


def iter_condition_rows(manifest: pd.DataFrame, conditions: Optional[List[str]]) -> pd.DataFrame:
    """Filter manifest rows by condition list (empty/None = all conditions).
    Replaces folder-glob-based condition iteration."""
    if not conditions:
        return manifest
    return manifest[manifest["condition"].isin(conditions)]
