"""
parse_ids.py - Parse the heterogeneous raw filenames into structured fields and
propose a clean canonical bouton id. REVIEW TOOL (writes a preview CSV) - it does
not modify any data.

Two filename families are handled:
  legacy (Theo):  20200907_linescan1_20Hz_10pulses_4mMCa_bouton2_traces_converted
  new (Anthime):  241212_Fibre1_PortionB_Bouton_1_bis   /   250113_Fibre1_Bouton_1

Proposed canonical ids:
  base_uid = <yyyymmdd>_fibre<F>[_sec<S>]_bouton<M>     (the physical bouton)
  uid      = <base_uid>__<condition>                    (a single recording)

Usage:
    python parse_ids.py --data-root "<path to PPR_DATA_FINAL>"
"""
import argparse
import os
import re
import pandas as pd

ALL_CONDITIONS = [
    "Stability_Before", "Stability_After", "Stability_Before_05", "Stability_After_05",
    "Theo_4Ca", "Theo_1_5Ca", "WT_Theo", "WT_Theo_1scd", "WT_Anthime", "SynII",
    "Theo_1_5_50Hz", "Theo_2_5_50Hz", "Theo_4_50Hz",
]

_RE_DATE = re.compile(r"^(\d{8}|\d{6})_")
_RE_FIBRE = re.compile(r"(?:fibre|linescan)_?(\d+)", re.IGNORECASE)
_RE_SECTION = re.compile(r"portion([A-Za-z0-9]+)", re.IGNORECASE)
_RE_BOUTON = re.compile(r"bouton_?(\d+)", re.IGNORECASE)
_RE_FREQ = re.compile(r"(\d+)\s*Hz", re.IGNORECASE)
_RE_CA = re.compile(r"(\d+(?:[._]\d+)?)\s*mMCa", re.IGNORECASE)
_RE_PULSES = re.compile(r"(\d+)\s*pulses", re.IGNORECASE)
_RE_SET = re.compile(r"set(\d+)", re.IGNORECASE)


def parse_id(stem: str) -> dict:
    d = {"date": None, "fibre": None, "section": None, "bouton": None,
         "freq_hz": None, "ca_mM": None, "pulses": None, "set": None, "parse_ok": True}

    m = _RE_DATE.match(stem)
    if m:
        raw = m.group(1)
        d["date"] = raw if len(raw) == 8 else "20" + raw  # yymmdd -> yyyymmdd
    else:
        d["parse_ok"] = False

    m = _RE_FIBRE.search(stem)
    d["fibre"] = int(m.group(1)) if m else None
    if m is None:
        d["parse_ok"] = False

    m = _RE_SECTION.search(stem)
    d["section"] = m.group(1).upper() if m else None

    m = _RE_BOUTON.search(stem)
    d["bouton"] = int(m.group(1)) if m else None
    if m is None:
        d["parse_ok"] = False

    m = _RE_FREQ.search(stem)
    d["freq_hz"] = int(m.group(1)) if m else None
    m = _RE_CA.search(stem)
    d["ca_mM"] = float(m.group(1).replace("_", ".")) if m else None
    m = _RE_PULSES.search(stem)
    d["pulses"] = int(m.group(1)) if m else None
    m = _RE_SET.search(stem)
    d["set"] = int(m.group(1)) if m else None
    return d


def base_uid(p: dict) -> str:
    if not p["parse_ok"]:
        return None
    parts = [p["date"], f"fibre{p['fibre']}"]
    if p["section"]:
        parts.append(f"sec{p['section']}")
    parts.append(f"bouton{p['bouton']}")
    if p["set"]:
        parts.append(f"set{p['set']}")
    return "_".join(parts)


def build(data_root: str) -> pd.DataFrame:
    rows = []
    for cond in ALL_CONDITIONS:
        folder = os.path.join(data_root, cond)
        if not os.path.isdir(folder):
            continue
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith(".xlsx"):
                continue
            stem = os.path.splitext(fn)[0]
            p = parse_id(stem)
            bu = base_uid(p)
            rows.append({
                "condition": cond,
                "legacy_id": stem,
                **{k: p[k] for k in ["date", "fibre", "section", "bouton", "freq_hz", "ca_mM", "pulses", "set"]},
                "parse_ok": p["parse_ok"],
                "base_uid": bu,
                "uid": f"{bu}__{cond}" if bu else None,
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args()
    df = build(args.data_root)
    out = os.path.join(args.data_root, "metadata")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "_id_parse_preview.csv")
    df.to_csv(path, index=False)

    n = len(df)
    print(f"rows: {n}   parse_ok: {int(df['parse_ok'].sum())}/{n}")
    print(f"unparsed rows:")
    for _, r in df[~df["parse_ok"]].head(10).iterrows():
        print("   ", r["condition"], "|", r["legacy_id"])
    dup_uid = df["uid"].duplicated(keep=False) & df["uid"].notna()
    print(f"duplicate uid (recording key) collisions: {int(dup_uid.sum())}")
    for _, r in df[dup_uid].head(10).iterrows():
        print("   ", r["uid"], "<-", r["legacy_id"])
    # same physical bouton across conditions
    bu_multi = df.groupby("base_uid")["condition"].nunique()
    print(f"physical boutons (base_uid): {df['base_uid'].nunique()}  | appearing in >1 condition: {int((bu_multi > 1).sum())}")
    print(f"section present: {df['section'].notna().sum()}  | wrote preview: {path}")
    print("\n--- sample mapping (legacy_id -> uid) ---")
    for _, r in df.sample(min(12, n), random_state=0).sort_values('condition').iterrows():
        print(f"  [{r['condition']:16}] {str(r['legacy_id'])[:55]:55} -> {r['uid']}")


if __name__ == "__main__":
    main()
