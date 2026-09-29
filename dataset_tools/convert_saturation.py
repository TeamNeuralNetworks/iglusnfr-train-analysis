"""
convert_saturation.py - Convert the supplementary Saturation_data.xlsx (a separate
high-frequency saturation experiment) into tidy CSVs for the release folder:

    saturation_amps.csv    columns: ca_mM, button_id, AMP1..AMPk
    saturation_traces.csv  long:    ca_mM, button_id, time_s, dff

Usage:
    python convert_saturation.py --xlsx "<...>/Saturation_data.xlsx" --out "<release dir>"
"""
import argparse
import os
import re
import pandas as pd

_SUFFIX = re.compile(r"_dF_F0_(trace|time)$", re.IGNORECASE)


def convert(xlsx_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    xl = pd.ExcelFile(xlsx_path)

    # --- amps sheets -> one tidy table with a ca_mM column ---
    amp_frames = []
    for sheet in [s for s in xl.sheet_names if s.lower().endswith("_amps")]:
        ca = sheet.split("mMCa")[0].replace("_", ".")
        d = xl.parse(sheet)
        d.insert(0, "ca_mM", float(ca))
        amp_frames.append(d)
    amps = pd.concat(amp_frames, ignore_index=True) if amp_frames else pd.DataFrame()
    amps = amps.rename(columns={"ID": "button_id"})

    # --- trace sheets -> long (ca_mM, button_id, time_s, dff) ---
    long_frames = []
    for sheet in [s for s in xl.sheet_names if s.lower().endswith("_traces")]:
        ca = sheet.split("mMCa")[0].replace("_", ".")
        d = xl.parse(sheet)
        cols = [c for c in d.columns if str(c).startswith("2") or str(c).startswith("Unnamed")]
        # pair each button's _trace with its _time column
        buttons = {}
        for c in d.columns:
            m = _SUFFIX.search(str(c))
            if not m:
                continue
            base = _SUFFIX.sub("", str(c))
            buttons.setdefault(base, {})[m.group(1).lower()] = c
        for base, pair in buttons.items():
            if "trace" not in pair or "time" not in pair:
                continue
            dff = pd.to_numeric(d[pair["trace"]], errors="coerce")
            t = pd.to_numeric(d[pair["time"]], errors="coerce")
            sub = pd.DataFrame({"ca_mM": float(ca), "button_id": base, "time_s": t, "dff": dff})
            long_frames.append(sub[sub["dff"].notna()])
    traces = pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame()

    amps.to_csv(os.path.join(out_dir, "saturation_amps.csv"), index=False)
    traces.to_csv(os.path.join(out_dir, "saturation_traces.csv"), index=False)
    print(f"[export] saturation_amps.csv   rows={len(amps)}  cols={amps.shape[1]}")
    print(f"[export] saturation_traces.csv rows={len(traces)}  buttons={traces['button_id'].nunique() if len(traces) else 0}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    convert(args.xlsx, args.out)


if __name__ == "__main__":
    main()
