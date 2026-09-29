"""
build_manifest.py - Generate the master per-bouton metadata manifest.

Produces `<DATA_ROOT>/release/boutons_manifest.csv`: one row per raw recording, keyed
by a clean canonical id, carrying all identifiers, experimental parameters, and
biology. Lives inside `release/`, alongside `release/raw/` (written by
reorganize_raw.py) — together they are the "well-organized raw" stage that
Feature_extraction/Model_Calibration scripts load for extraction, kept separate from
the metrics-enriched `release/boutons.csv` produced later by consolidate.py.

This manifest is the single source of truth. It replaces ID_and_sex.csv,
Target_WT_pooled.xlsx, and the hardcoded ISI/baseline lookup dicts in
demo_batch_process.py. All external metadata is joined by parsing every id family
through parse_ids.parse_id() into a shared `base_uid`, so the messy legacy id
strings never have to be matched by hand again.

Usage:
    python build_manifest.py --data-root "<path to PPR_DATA_FINAL>"
    # raw .xlsx quarantined in their own subfolder, release/ built in the parent:
    python build_manifest.py --data-root "<parent>" --source-root "<parent>/MESSY RAW DATA"
"""
import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_ids import parse_id, base_uid, ALL_CONDITIONS

# --- Condition tables (mirror demo_batch_process.py at time of migration) ---
FREQ_HZ_BY_CONDITION = {c: (50 if c.endswith("_50Hz") else 20) for c in ALL_CONDITIONS}
DEFAULT_BASELINE = 0.998
BASELINE_BY_CONDITION = {
    "Stability_Before_05": 0.498, "Stability_After_05": 0.498,
    "Theo_4Ca": 0.498, "Theo_1_5Ca": 0.498, "WT_Theo": 0.498,
    "Theo_1_5_50Hz": 0.498, "Theo_2_5_50Hz": 0.498, "Theo_4_50Hz": 0.498,
}
DEFAULT_N_PULSES = 10
VALID_TARGETS = ["PC", "IN", "UN"]


def _base_uid_of(stem):
    """Canonical physical-bouton id for any raw/legacy id string (or None)."""
    if stem is None or (isinstance(stem, float)) or str(stem).strip() == "":
        return None
    return base_uid(parse_id(str(stem).strip()))


def build(data_root: str, source_root: str = None) -> pd.DataFrame:
    """Scan `source_root` (default: `data_root`) for the raw per-condition
    folders and annex files. `data_root` only matters for where the caller
    later writes `release/`; the manifest's `file` column stays relative to
    `source_root`, matching reorganize_raw.py's `--source-root` convention."""
    source_root = source_root or data_root
    rows = []
    for cond in ALL_CONDITIONS:
        folder = os.path.join(source_root, cond)
        if not os.path.isdir(folder):
            print(f"[warn] missing condition folder: {folder}")
            continue
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith(".xlsx"):
                continue
            stem = os.path.splitext(fn)[0]
            p = parse_id(stem)
            bu = base_uid(p)
            rows.append({
                "uid": f"{bu}__{cond}" if bu else None,
                "base_uid": bu,
                "condition": cond,
                "legacy_id": stem,
                "file": f"{cond}/{fn}",
                "date": p["date"],
                "fibre": p["fibre"],
                "section": p["section"],
                "bouton": p["bouton"],
                "frequency_hz": FREQ_HZ_BY_CONDITION[cond],
                "ca_mM": p["ca_mM"],
                "baseline_s": BASELINE_BY_CONDITION.get(cond, DEFAULT_BASELINE),
                "n_pulses": DEFAULT_N_PULSES,
            })
    man = pd.DataFrame(rows)

    # --- merge sex (keyed by condition + physical bouton) ---
    sex_path = os.path.join(source_root, "ID_and_sex.csv")
    if os.path.exists(sex_path):
        sex = pd.read_csv(sex_path)
        sex["base_uid"] = sex["ID"].map(_base_uid_of)
        sex = sex.dropna(subset=["base_uid"])[["condition", "base_uid", "sex"]]
        sex = sex.drop_duplicates(["condition", "base_uid"])
        man = man.merge(sex, on=["condition", "base_uid"], how="left")
    else:
        man["sex"] = pd.NA

    # --- merge target cell identity PC/IN/UN (keyed by physical bouton) ---
    tgt_path = os.path.join(source_root, "Target_WT_pooled.xlsx")
    if os.path.exists(tgt_path):
        tgt = pd.read_excel(tgt_path).iloc[:, :2].copy()
        tgt.columns = ["legacy", "target"]
        tgt["base_uid"] = tgt["legacy"].map(_base_uid_of)
        tgt["target"] = tgt["target"].astype(str).str.strip().str.upper()
        tgt["target"] = tgt["target"].where(tgt["target"].isin(VALID_TARGETS), "UN")
        tgt = tgt.dropna(subset=["base_uid"]).drop_duplicates("base_uid")[["base_uid", "target"]]
        man = man.merge(tgt, on="base_uid", how="left")
    else:
        man["target"] = pd.NA

    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--source-root", default=None,
                    help="where the raw .xlsx (and ID_and_sex.csv/Target_WT_pooled.xlsx "
                         "annexes) live (default: <data-root>)")
    ap.add_argument("--out", default=None, help="output root (default: <data-root>/release)")
    args = ap.parse_args()

    source_root = args.source_root or args.data_root
    man = build(args.data_root, source_root=source_root)
    out_dir = args.out or os.path.join(args.data_root, "release")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "boutons_manifest.csv")
    man.to_csv(out, index=False)

    stale = os.path.join(args.data_root, "metadata", "boutons.csv")
    if os.path.exists(stale):
        print(f"[info] Found an old-location manifest at {stale} (from a "
              f"pre-restructuring run). It is no longer read by this pipeline; "
              f"safe to delete.")

    print(f"[export] wrote {out}  ({len(man)} rows)")
    print(f"  unique uid: {man['uid'].nunique()}  | physical boutons: {man['base_uid'].nunique()}")
    print(f"  sex present: {man['sex'].notna().sum()} / {len(man)}")
    print(f"  ca_mM parsed: {man['ca_mM'].notna().sum()} / {len(man)}")
    print(f"  target present: {man['target'].notna().sum()} / {len(man)}  {man['target'].value_counts().to_dict()}")
    print(f"  freq counts: {man['frequency_hz'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
