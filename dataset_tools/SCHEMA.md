# Dataset schema & Zenodo layout

This document defines the consolidated data format for the iGluSnFR release. It
replaces the previous set of parallel Excel/CSV files (summary, times, traces,
trials, null, plus the `ID_and_sex` / `Target_WT_pooled` annexes) with **one
metadata manifest + four tidy CSV tables**, all keyed by a single clean id.

## 1. Canonical bouton id

Raw filenames come in two families and carry parameters inside the string:

| family | example |
|---|---|
| legacy (Theo) | `20200907_linescan1_20Hz_10pulses_4mMCa_bouton2_traces_converted` |
| new (Anthime/SynII) | `241212_Fibre1_PortionB_Bouton_1_bis`, `250128_Fibre2_Bouton_1` |

`dataset_tools/parse_ids.py` parses both into structured fields and builds:

- **`base_uid`** — the *physical bouton*: `<yyyymmdd>_fibre<F>[_sec<S>]_bouton<M>[_set<K>]`
  e.g. `20241212_fibre1_secB_bouton1`
- **`uid`** — one *recording* = `<base_uid>__<condition>`
  e.g. `20241212_fibre1_secB_bouton1__Stability_After`

Dates are normalised to `yyyymmdd` (6-digit `yymmdd` → `20yymmdd`). All embedded
parameters (frequency, pulses, [Ca], `_traces_converted`, `_bis`, `_set1`) become
**columns**, never parsed from strings downstream. A bouton recorded in several
conditions (e.g. pre/post protocol, Ca series) shares one `base_uid`; join on
`base_uid` for cross-condition analysis and on `uid` for a single recording.

*Validation:* 427/427 filenames parse, 427 unique `uid` (0 collisions), 347
physical boutons (65 appear in >1 condition), 69 with a section flag.

## 2. Files

The published dataset is one **flat** `release/` folder, everything CSV (bar the
provenance JSON and this doc), raw recordings renamed to `<uid>.csv`:

```
release/
  boutons_manifest.csv  # 1 row / recording: identifiers + params + biology, no metrics
  boutons.csv           # 1 row / recording: metadata (sex, target, freq, Ca...) + all metrics
  trials.csv            # 1 row / (recording, trial)
  null_amps.csv         # 1 row / (recording, trial): null distribution
  traces.csv            # long: 1 row / (recording, sample): time_s, dff
  f0.csv                # 1 row / (recording, trial): pre-stimulus baseline fluorescence (F0)
  saturation_amps.csv   # supplementary saturation experiment (amplitudes)
  saturation_traces.csv # supplementary saturation experiment (long traces)
  run_settings.json     # settings + environment + export date (provenance)
  SCHEMA.md             # this file
  raw/
    <uid>.csv           # renamed raw recordings (time_s, trial_*, average)
```

The **input manifest** `release/boutons_manifest.csv` (identifiers + params +
biology, no metrics) is the source of truth every extraction/fitting script reads,
alongside `release/raw/` — together these two are the "well-organized raw" stage:
everything needed to re-run extraction, without the private per-condition `.xlsx`
tree. `release/boutons.csv` is that same manifest enriched with the metrics once
extraction has run. Both are shipped as part of the deposit (§4). The former
`ID_and_sex.csv` and `Target_WT_pooled.xlsx` are folded into the manifest (`sex`,
`target` columns) and are not shipped separately.

## 3. Column dictionary

All tables below are shipped as the flat `release/*.csv` files listed in §2.

### `release/boutons_manifest.csv` (manifest)
| column | meaning |
|---|---|
| `uid` | canonical recording id (primary key) |
| `base_uid` | canonical physical-bouton id (cross-condition key) |
| `condition` | experimental condition / source folder |
| `legacy_id` | original filename stem (provenance) |
| `file` | raw file path relative to data root (`<condition>/<name>.xlsx`) |
| `date` | recording date `yyyymmdd` |
| `fibre` | fibre / linescan number |
| `section` | portion flag (A/B/C/D) where present, else empty |
| `bouton` | bouton number |
| `frequency_hz` | stimulation frequency (20 or 50) — authoritative, from condition |
| `ca_mM` | extracellular [Ca] in mM where present in the filename |
| `baseline_s` | train-start / baseline time used by the pipeline |
| `n_pulses` | number of pulses in the train (10) |
| `sex` | animal sex (where annotated) |
| `target` | postsynaptic target identity: PC / IN / UN |

### `release/boutons.csv`
All manifest columns **plus** the per-bouton metrics (already merged — no manual join):
`AMP1..AMP10` (corrected ΔF/F0), `AMP1_UNCORR..AMP10_UNCORR`, `PPR2/1..PPR10/1`,
`%Fail1..%Fail3`, `NOISE_THR_MEDIAN`, `measurement` (SAVGOL).

### `release/trials.csv`  (grain: recording × trial)
`uid, condition, legacy_id, trial, trial_input_col_1based, status, F0, thr_shared,
noise_level, baseline_null_{mean,median}_{including,excluding}_zero,
AMP1_CORR..AMP10_CORR, AMP1_UNCORR..AMP10_UNCORR, AMP1`.

### `release/null_amps.csv`  (grain: recording × trial)
`uid, condition, legacy_id, trial, trial_input_col_1based, status, nnls_null_n,
nnls_null_amps_json` (the null amplitudes as a JSON list).

### `release/traces.csv`  (grain: recording × sample; long format)
`uid, condition, time_s, dff` — replaces the wide `summary_traces` + `summary_times`.

### `release/f0.csv`  (grain: recording × trial)
`uid, trial_col, F0` — pre-stimulus baseline fluorescence used to normalize each
trial's trace to ΔF/F0.

## 4. Zenodo deposit

Deposit the flat `release/` folder (Section 2) as the data. Code is deposited via a
tagged GitHub release (Zenodo–GitHub integration mints a separate DOI);
`run_settings.json` records the exact commit + environment that produced the tables.

Notes:
- Raw `.xlsx` → `.csv` keyed by `uid` (smaller, openable everywhere, no Excel engine).
- The large `.tif` stacks (Combined Stacks, all fits) are **not** part of this deposit
  (size, and not needed to reproduce the figures) — deposit separately if required.

## 5. Regenerating

```bash
# 1. manifest (input source of truth) from the raw folders + sex/target annexes
#    -> writes <DATA_ROOT>/release/boutons_manifest.csv
python dataset_tools/build_manifest.py   --data-root <DATA_ROOT>

# 2. renamed raw recordings (release/raw/<uid>.csv), f0.csv (baseline fluorescence),
#    and supplementary saturation tables
python dataset_tools/reorganize_raw.py   --data-root <DATA_ROOT> --out <DATA_ROOT>/release
python dataset_tools/convert_saturation.py --xlsx <DATA_ROOT>/Saturation_data.xlsx --out <DATA_ROOT>/release

# 3. extraction, reading only from release/raw/ + release/boutons_manifest.csv
#    (Feature_extraction/demo_batch_process.py, WRITE_TIDY=True) writes the four
#    tidy tables straight into release/ -- or convert existing summary_* outputs
#    directly:
python dataset_tools/consolidate.py --data-root <DATA_ROOT> \
       --summary-dir <dir with summary_*> --out <DATA_ROOT>/release
```
`consolidate.py` converts the *existing* canonical outputs, so the tables are
byte-identical to the published results (verified: max |Δ| = 9e-16 on the metrics).
A full pipeline run (`demo_batch_process.py`, `WRITE_TIDY=True`) writes the same
tables straight into `release/`, reading recordings from `release/raw/` rather than
the private `.xlsx` tree.
