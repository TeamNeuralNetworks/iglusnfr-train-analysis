"""
consolidate.py - Collapse the 6 legacy output files + the metadata manifest into
4 tidy CSVs keyed by the canonical `uid`, ready for the Zenodo `derived/` folder.

Inputs  (in <DATA_ROOT>):
    release/boutons_manifest.csv          (manifest; run build_manifest.py first)
    summary.csv                           (per-bouton metrics)
    summary_trials.xlsx                   (per-trial amplitudes / thresholds)
    summary_trials_nnls_null.xlsx         (per-trial null distribution)
    summary_traces.xlsx + summary_times.xlsx  (wide averaged traces + time)

Outputs (in <DATA_ROOT>/derived):
    boutons.csv     one row / bouton : manifest metadata + summary metrics
    trials.csv      one row / (bouton, trial)
    null_amps.csv   one row / (bouton, trial) with the null distribution
    traces.csv      long: one row / (bouton, sample) : time_s, dff

This is a non-destructive migration bridge: it converts the *existing* canonical
outputs, so the derived data is guaranteed identical to what is already published.

Usage:
    python consolidate.py --data-root "<path to PPR_DATA_FINAL>"
"""
import argparse
import os
import pandas as pd


def _load_map(manifest_path):
    man = pd.read_csv(manifest_path)
    key = man[["condition", "legacy_id", "uid"]].copy()
    return man, key


def _uid_map(key):
    """(condition, X) -> uid, where X may be either the legacy_id or the uid itself.
    Lets consolidation accept summaries keyed by either id (older runs used uid)."""
    m = {}
    for c, leg, u in key[["condition", "legacy_id", "uid"]].itertuples(index=False):
        m[(str(c), str(leg))] = u
        m[(str(c), str(u))] = u
    return m


def _to_uid(df, key, cond_col, id_col):
    """Attach a uid column by matching (condition, id) against legacy_id or uid."""
    m = _uid_map(key)
    df = df.copy()
    df["uid"] = [m.get((str(c), str(i))) for c, i in zip(df[cond_col], df[id_col])]
    missing = int(df["uid"].isna().sum())
    if missing:
        print(f"[warn] {missing} rows had no uid match on ({cond_col},{id_col})")
    return df


# Manifest metadata columns (everything that is NOT a computed metric). Selecting
# these keeps consolidation idempotent even when the manifest passed in is
# release/boutons.csv (which already carries metrics from a previous run) rather
# than the metrics-free release/boutons_manifest.csv.
META_COLS = ["uid", "base_uid", "condition", "legacy_id", "file", "date", "fibre",
             "section", "bouton", "frequency_hz", "ca_mM", "baseline_s", "n_pulses",
             "sex", "target"]


def build_boutons(summary_dir, man, key):
    meta = man[[c for c in META_COLS if c in man.columns]].copy()
    summ = pd.read_csv(os.path.join(summary_dir, "summary.csv"))
    summ = _to_uid(summ, key, "condition", "ID")
    metric_cols = [c for c in summ.columns if c not in ("condition", "ID", "uid", "measurement")]
    meas = summ[["uid", "measurement"]] if "measurement" in summ.columns else None
    out = meta.merge(summ[["uid", *metric_cols]], on="uid", how="left")
    if meas is not None:
        out = out.merge(meas, on="uid", how="left")
    # Preserve the summary's row order: downstream clustering/pooling is order-sensitive
    # (cluster label assignment), so the shipped table must match the canonical order.
    order = {u: i for i, u in enumerate(summ["uid"].astype(str))}
    out = (out.assign(_ord=out["uid"].astype(str).map(order))
              .sort_values("_ord", kind="stable")
              .drop(columns="_ord")
              .reset_index(drop=True))
    return out


def build_trials(summary_dir, key):
    tr = pd.read_excel(os.path.join(summary_dir, "summary_trials.xlsx"))
    tr = _to_uid(tr, key, "condition", "file").rename(columns={"file": "legacy_id"})
    front = [c for c in ["uid", "condition", "legacy_id", "trial"] if c in tr.columns]
    rest = [c for c in tr.columns if c not in front]
    return tr[front + rest]


def build_null(summary_dir, key):
    nl = pd.read_excel(os.path.join(summary_dir, "summary_trials_nnls_null.xlsx"))
    nl = _to_uid(nl, key, "condition", "file").rename(columns={"file": "legacy_id"})
    front = [c for c in ["uid", "condition", "legacy_id", "trial"] if c in nl.columns]
    rest = [c for c in nl.columns if c not in front]
    return nl[front + rest]


def build_traces(summary_dir, key):
    xl_tr = pd.ExcelFile(os.path.join(summary_dir, "summary_traces.xlsx"))
    xl_ti = pd.ExcelFile(os.path.join(summary_dir, "summary_times.xlsx"))
    kmap = _uid_map(key)
    frames = []
    for sheet in xl_tr.sheet_names:
        tr = xl_tr.parse(sheet)
        ti = xl_ti.parse(sheet) if sheet in xl_ti.sheet_names else None
        for col in tr.columns:
            uid = kmap.get((sheet, col))
            if uid is None:
                continue
            dff = pd.to_numeric(tr[col], errors="coerce").to_numpy()
            time = (pd.to_numeric(ti[col], errors="coerce").to_numpy()
                    if ti is not None and col in ti.columns else range(len(dff)))
            sub = pd.DataFrame({"uid": uid, "condition": sheet, "time_s": time, "dff": dff})
            sub = sub[sub["dff"].notna()]
            frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def consolidate(summary_dir, manifest_path, out_dir, verbose=True):
    """Build the 4 tidy CSVs from the summary_* files in `summary_dir` merged with
    the manifest, writing them into `out_dir`. Returns the output directory."""
    man, key = _load_map(manifest_path)
    os.makedirs(out_dir, exist_ok=True)
    tables = [
        ("boutons", build_boutons(summary_dir, man, key)),
        ("trials", build_trials(summary_dir, key)),
        ("null_amps", build_null(summary_dir, key)),
        ("traces", build_traces(summary_dir, key)),
    ]
    for name, df in tables:
        p = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(p, index=False)
        if verbose:
            print(f"[export] {name}.csv  rows={len(df):>7}  cols={df.shape[1]}  -> {p}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--summary-dir", default=None,
                    help="dir containing summary*.csv/xlsx (default: <data-root>)")
    ap.add_argument("--manifest", default=None,
                    help="manifest path (default: <data-root>/release/boutons_manifest.csv)")
    ap.add_argument("--out", default=None, help="output dir (default: <data-root>/derived)")
    args = ap.parse_args()

    summary_dir = args.summary_dir or args.data_root
    manifest = args.manifest or os.path.join(args.data_root, "release", "boutons_manifest.csv")
    out_dir = args.out or os.path.join(args.data_root, "derived")
    consolidate(summary_dir, manifest, out_dir)


if __name__ == "__main__":
    main()
