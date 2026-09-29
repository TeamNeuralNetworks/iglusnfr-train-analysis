"""
reorganize_raw.py - Copy the raw per-bouton recordings into a flat Zenodo layout:

    raw/<uid>.csv   'Traces no Fback' (or sheet 0): raw fluorescence, standardised
                    columns time_s, trial_<label>..., average
    f0.csv          one row per (uid, trial): the scalar baseline F0.

The workbook's 'Traces DF_F0' sheet is NOT stored: it equals (raw - F0)/F0 exactly
(verified to ~1e-16), so ΔF/F0 is reconstructed on demand from raw + f0.

Non-destructive. Reads the metrics-free manifest written by build_manifest.py
(`<data-root>/release/boutons_manifest.csv`) to know which raw files to convert.

Usage:
    python reorganize_raw.py --data-root "<PPR_DATA_FINAL>"
    # raw .xlsx in a backup, manifest already consolidated with metrics:
    python reorganize_raw.py --data-root "<work>" --source-root "<...- Copy>" \
                             --manifest "<work>/release/boutons.csv" --out "<work>/release"
"""
import argparse
import os
import pandas as pd

RAW_SHEET_CANDIDATES = ("Traces no Fback",)  # else falls back to first sheet
F0_SHEET = "F0"


def _standardise_raw(df):
    rename, trial_cols = {}, []
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl == "time":
            rename[c] = "time_s"
        elif cl == "average":
            rename[c] = "average"
        else:
            new = f"trial_{c}"; rename[c] = new; trial_cols.append(new)
    df = df.rename(columns=rename)
    order = (["time_s"] if "time_s" in df.columns else []) + trial_cols + \
            (["average"] if "average" in df.columns else [])
    return df[order], trial_cols


def convert_one(src_xlsx, uid, out_root):
    xl = pd.ExcelFile(src_xlsx)
    sheets = xl.sheet_names
    raw_sheet = next((s for s in RAW_SHEET_CANDIDATES if s in sheets), sheets[0])
    std, trial_cols = _standardise_raw(xl.parse(raw_sheet))
    std.to_csv(os.path.join(out_root, "raw", f"{uid}.csv"), index=False)

    f0_rows = []
    if F0_SHEET in sheets:
        f0_vals = pd.to_numeric(xl.parse(F0_SHEET).iloc[0], errors="coerce").to_numpy()
        # F0 sheet = one value per trial, then a trailing workbook-average column.
        for pos, tcol in enumerate(trial_cols):
            if pos < len(f0_vals):
                f0_rows.append({"uid": uid, "trial_col": tcol, "F0": float(f0_vals[pos])})
    return f0_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--source-root", default=None,
                    help="where the raw .xlsx live (default: <data-root>)")
    ap.add_argument("--manifest", default=None,
                    help="manifest path (default: <data-root>/release/boutons_manifest.csv)")
    ap.add_argument("--out", default=None, help="output root (default: <data-root>/release)")
    args = ap.parse_args()

    source_root = args.source_root or args.data_root
    manifest = args.manifest or os.path.join(args.data_root, "release", "boutons_manifest.csv")
    out_root = args.out or os.path.join(args.data_root, "release")
    os.makedirs(os.path.join(out_root, "raw"), exist_ok=True)

    man = pd.read_csv(manifest)
    n_ok, n_fail, f0_all = 0, 0, []
    for r in man.itertuples(index=False):
        src = os.path.join(source_root, str(r.file).replace("/", os.sep))
        try:
            f0_all.extend(convert_one(src, r.uid, out_root))
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[fail] {r.uid}: {e}")
    pd.DataFrame(f0_all).to_csv(os.path.join(out_root, "f0.csv"), index=False)
    print(f"[done] raw={n_ok}  f0 rows={len(f0_all)}  (failures: {n_fail}) -> {out_root}")


if __name__ == "__main__":
    main()
