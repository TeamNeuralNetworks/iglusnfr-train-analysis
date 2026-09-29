# Streamlined iGluSnFR Analysis (extract_metrics)

This guide mirrors the demo scripts with generic paths and options for the simplified API built around `extract_metrics`.

The function keeps the math equivalent to the main pipeline while exposing a small, explicit options dictionary and minimal plotting controls.

- Import path: `from extract_metrics import extract_metrics`
- Plot controls: `options['plot'] = {'enabled': True, 'traces': ['raw','savgol','nnls'], 'show_decay': True, 'trials': False, 'baseline': False}`
- When decay progression rules specify monotonic increase (default), τd is constrained non‑decreasing for stability.
- Core controls:
  - `measurement`: `'NNLS'|'SAVGOL'|'RAW'` (default `'NNLS'`) — p‑values use this amplitude series
  - `fail_method`: `'NNLS'|'SAVGOL'|'RAW'` (default: same as `measurement`) — controls null/threshold rule
    - `threshold_mode`: `'auto'|'mad'|'sd'` (default `'auto'`) — failure rates compare
      pulses 1–3 against a single baseline threshold; pulses 2/3 amplitudes are
      corrected for residual pre‑stim currents
    - `allow_shift`: bool (default True) — enable per‑pulse micro‑shifts
    - `event_model`: kernel used for fitting. Default `'double_exp'`. For iGluSnFR use `'iglusnfr'` (bi‑exp) or `'iglusnfr_tri'` (tri‑exp). In tri‑exp mode, recut fits are bi‑exp for fast/slow taus and superslow tau is estimated from the final event decay.
    - Extras for `'cooperative'`: `event_model_settings={'n_coop': 2.0}`
  - `fit_source`: `'global'|'average'|'individual'` (default `'global'`)
    - `global`: fit a **single recut event** (all trials) to get τr and decay(s); apply those τs to all events.
      Progression affects **ratios/fractions** (and anchors if enabled), not per‑event τ unless anchors are requested.
    - `average`: fit kinetics **per event on the average trace** then smooth via progression.
    - `individual`: fit kinetics **per trial** then aggregate (median) and smooth.
  - `decay_progression_mode`: `'fixed'|'free_monotonic'|'linear'` (default `'linear'`)
    - `fixed`: one τd for the whole train
    - `free_monotonic`: monotonic regression (default: non‑decreasing) between first and last event
    - `linear`: robust linear trend (slope constrained by progression rules)

---

## Decay Progression System

The decay progression system determines how decay time constants (τd) evolve across the stimulus train. This is critical for accurately modeling synaptic depression or facilitation where each subsequent event may have different kinetics.

---

## Tri‑exponential iGluSnFR model (`iglusnfr_tri`)

Tri‑exp mode behaves like the bi‑exp model for the recut fit (fast/slow taus), then adds a superslow component only during NNLS screening:

- **Fast/slow taus**: taken from the **global recut fit** and reused for all events.
- **Superslow tau**: estimated from the **last‑event decay** using a robust single‑exponential fit (50 ms max window).
  The fit is **amplitude‑weighted** (Savgol‑weighted if available, raw otherwise).
- **Template variants**: scan slow fraction (`template_variant_ratios`) and superslow fraction at the **final** event (`template_variant_superslow_fracs`).
  Superslow ramps monotonically from 0 to the chosen max across the train.
- **Safeguard**: if the estimated superslow tau is too close to the slow tau (ratio below `superslow_min_ratio`, default 1.0), superslow variants are disabled and the fit reduces to bi‑exp templates.
  If `allow_tau_slow_override=True` and the last‑event fit is **faster** than the recut slow tau, slow can be replaced; `force_tau_slow_override=True` always forces slow = superslow.

**Soft vs hard variant selection**
- `template_variant_select='soft'` (default): NNLS mixes multiple variants per event.
- `template_variant_select='hard'`: NNLS selects one dominant variant per event.
- The **parameter‑evolution plot** mirrors this:
  - Soft mode shows **weighted average fractions** (from NNLS variant weights).
  - Hard mode shows **single‑variant fractions**.

Deprecated settings: `template_variant_weights`, `triexp_weight_step`, and `triexp_weight_min` are no longer used; replace them with `template_variant_ratios` + `template_variant_superslow_fracs`.

### Overview

The system operates in **three stages**:

1. **Initial kinetics**:  
   - `global`: fit a **single recut event** to get τr and decay(s).  
   - `average`/`individual`: estimate τ per event (average trace or per‑trial).
2. **Anchoring (optional)**: apply `anchor_first_tau` / `anchor_final_tau`.
3. **Progression**: apply smoothing/regression based on `decay_progression_mode`.

### Key Options

```python
options = {
    'fit_source': 'global',              # How to estimate initial τd values
    'decay_progression_mode': 'linear',  # How to smooth/fit the progression
    'anchor_first_tau': False,           # Anchor to first event's τd
    'anchor_final_tau': True,            # Anchor to last event's τd (default)
}
```

---

### 1. Fit Source (`fit_source`)

Controls how initial τ values are estimated:

#### `'global'` (default, most robust)
- Fits a **single global recut event** by recutting and averaging all events from all trials.
- **τr and decay(s)** from the recut fit are reused for all events.
- Per‑event τ fitting is **skipped** in global mode unless anchors are explicitly enabled.
  - If `anchor_first_tau` and/or `anchor_final_tau` are enabled, only those events are fitted and used as anchor points.
- Progression in global mode **only acts on ratios/fractions** (slow/fast/superslow), not on τ values.

**Best for**: Most datasets, especially with noisy trials

#### `'average'`
- Fits each event **individually on the multi‑trial average trace**
- Progression applies to **per‑event τ estimates**

**Best for**: Clean average traces, when you want per-event detail without global anchoring

#### `'individual'`
- Fits each **trial independently**, then aggregates (median)
- Progression applies to aggregated per‑event τ estimates

**Best for**: Highly consistent trials, or when trial-to-trial variability is of interest

---

### Global recut fit (`fit_average_event`)

In `global` mode the recut average is fit once to obtain τr and decay(s). The fit process is:

1. **Grid search** on the recut average (robust, coarse).
2. **curve_fit refinement** using the grid result as a seed.
3. If curve_fit is **worse than the grid** (higher weighted SSE), the grid result is kept and you will see:
   `curve_fit worse than grid search, using grid search params`.

This message does **not** mean per‑event fitting is happening; it only reports that the global recut refinement was rejected.

Recut window for the global fit:
- `post_ms = min(50, isi_ms - 5)` when `isi_ms > 6`, otherwise `post_ms = 6`.

---

### 2. Decay Progression Mode (`decay_progression_mode`)

Controls how the per-event τd estimates are smoothed/fit.  
In `global` mode, τ values are fixed and progression applies **only to ratios/fractions**.

#### `'fixed'`
- Uses a **single τd for all events**
- Flat line in decay progression plot
- Ignores per-event variation

**Use when**: Events have identical kinetics, or to enforce uniform decay

#### `'linear'`
- **Robust linear regression** through per-event τd estimates
- Uses **IRLS (Iteratively Reweighted Least Squares)** with Huber weights
- Automatically downweights outliers (printed to console)
- Enforces non-negative slope (τd can only get slower or stay same)

**Use when**: Expect gradual, linear change across train (common for depression)

#### `'free_monotonic'`
- **Monotonic spline** (PCHIP interpolation) through per-event τd
- Enforces non-decreasing constraint (τd never gets faster)
- Smooth, flexible curve that follows data closely

**Use when**: Expect non-linear progression but want smooth, monotonic trend

---

### 3. Anchor Constraints

Fine-tune the progression by anchoring to specific events:

#### `anchor_final_tau` (default: `True`)
- **Anchors the last event's τd** as the maximum (slowest) decay time.
- For `linear`: Forces regression line through the final point.
- For `free_monotonic`: Forces spline endpoint to the final τd.
- In `global` mode, only the **last event** is fit for anchoring (no full per‑event fit).

**Recommended**: Keep enabled (default) for most robust results

#### `anchor_first_tau` (default: `False`)
- **Anchors the first event's τd** as the minimum (fastest) decay time
- Useful for enforcing a specific baseline decay
- For `linear`: Forces regression line through first point
- For `free_monotonic`: Forces spline start to first τd
 - In `global` mode, only the **first event** is fit for anchoring (no full per‑event fit)

**Use when**: You want to enforce a specific starting τd value

#### Both anchors enabled
- Forces progression **between first and last** τd values
- For `linear`: Direct line from event 1 to event 10
- For `free_monotonic`: Smooth curve constrained between endpoints
 - In `global` mode, only first/last events are fitted to define anchors; no full per‑event τ fit is done.

---

### Mode Combinations

#### Example 1: Default (Robust, Recommended)
```python
options = {
    'fit_source': 'global',
    'decay_progression_mode': 'linear',
    'anchor_first_tau': False,
    'anchor_final_tau': True,  # Anchor to most reliable estimate
}
```
- Global recut template (single τ)
- Ratios/fractions smoothed
- Final event anchored (most reliable)
- **Best for**: Most experiments, especially with noisy data

#### Example 2: Flexible, Smooth Progression
```python
options = {
    'fit_source': 'average',
    'decay_progression_mode': 'free_monotonic',
    'anchor_first_tau': False,
    'anchor_final_tau': True,
}
```
- Per-event fits on average trace
- Smooth monotonic spline
- Final event anchored
- **Best for**: Clean data with non-linear progression

#### Example 3: Fully Constrained
```python
options = {
    'fit_source': 'average',
    'decay_progression_mode': 'linear',
    'anchor_first_tau': True,
    'anchor_final_tau': True,
}
```
- Direct line from first to last event
- Ignores intermediate variation
- **Best for**: When you trust first and last estimates most

#### Example 4: Fixed Decay (No Progression)
```python
options = {
    'fit_source': 'global',
    'decay_progression_mode': 'fixed',
    # Anchor settings ignored for 'fixed' mode
}
```
- Single τd for entire train
- **Best for**: Events with identical kinetics

---

### Visualization: Decay Progression Plot

When `fit_diagnostic_plot=True`, the decay progression plot shows:

1. **Red dotted line with circles**: Initial per‑event τd estimates (not shown in `global` mode).
2. **Green solid line with squares**: Final progression fit.
3. **Vertical lines with markers**: Anchor points
   - **Green square** (`anchor_first_tau=True`): First event anchor
   - **Purple star** (`anchor_final_tau=True`): Final event anchor
4. **Horizontal dotted lines**: Show anchor τd values for reference

#### Interpreting the Plot

- **Gap between red and green**: How much smoothing/constraint was applied
- **Outliers**: Points far from green line; automatically downweighted in `linear` mode
- **Anchors**: Vertical lines show which events are constrained
- **Legend**: Shows actual τd values at anchor points

---

### Robust Linear Regression

The `linear` mode uses **IRLS with Huber weights** for outlier resistance:

- **MAD-based scale**: Robust to extreme outliers
- **Huber weighting**: Outliers (>2 MAD) are downweighted
- **Iterative refinement**: Converges in ~3-5 iterations
- **Automatic detection**: Prints outliers to console

Example output:
```
[robust fit] Detected outliers: event 4(τ=85.0ms,w=0.31), event 7(τ=92.1ms,w=0.28)
```

This means events 4 and 7 were automatically downweighted (weights 0.31 and 0.28 instead of 1.0) because they deviate significantly from the linear trend.

---

### Multi-Component Models

For models with multiple decay components (e.g., `iglusnfr` with `tau_decay_fast` and `tau_decay_slow`):

- `global` mode: **fixed decays** from the recut fit (fast/slow, plus superslow if tri‑exp).
  Progression acts on **fractions/ratios**, not on τ values.
- `average` / `individual`: progression is applied to the **per‑event τ estimates**.

---

### Last‑event decay fit (superslow estimate + overlay)

- The final‑event decay is fit with a **robust single‑exponential** model.
- Fit window is capped to **50 ms** after the decay start.
- Weights are **amplitude‑based** (Savgol‑weighted if available; raw otherwise).
- The orange dashed overlay is anchored to the **trace value at the last event** so it sits on the data.

---

### Baseline computation, normalization, and failure thresholds

The extractor keeps the same baseline logic as the full pipeline. The main
pieces (and the knobs you can adjust) are:

1. **Bleach removal (optional).** With `options['bleach']=True` the trace is
   first detrended by fitting a robust mono-exponential on quiet regions before
   and after the train. Disable this if bleaching has already been handled.
2. **Baseline window.** Samples strictly before `train_start` define the
   baseline mask. If fewer than five points exist (short pre-train segment),
   the earliest 10 % of the trace is used instead. This mask is shared by all
   downstream steps.
3. **F₀ estimation and ΔF/F normalization.** For each trial the median of the
   baseline samples becomes F₀. With `normalize_dff=True` the code reports
   ΔF/F₀; if a trial’s F₀ is effectively zero it falls back to simple
   baseline-subtraction so the series stays finite.
4. **Peak correction for overlapping decays.** Amplitudes are measured by
   subtracting the cumulative NNLS reconstruction from earlier pulses, so the
   “peak minus baseline” reflects only the current event even when decays
   overlap.
5. **Null distribution for failure calls.** The combination of
   `measurement`, `fail_method`, `threshold_mode`, and `null_N` controls the
   A1 threshold and p-values:
   - `measurement` selects which amplitude series is summarized and evaluated
     (`'NNLS'`, `'SAVGOL'`, or `'RAW'`).
   - `fail_method` chooses how the noise floor is estimated. `'NNLS'` (default)
     simulates single-pulse fits inside the final `f0_window_s` seconds before
     the train using the same kinetics, micro-shifts (`allow_shift`), and
     windowing as the real fit. `'SAVGOL'` or `'RAW'` instead measure windowed
     maxima on the smoothed or raw baseline, respectively.
   - `threshold_mode` sets the statistic: `'mad'` uses
     median + `null_N`·1.4826·MAD, `'sd'` uses mean + `null_N`·SD, and `'auto'`
     picks `'mad'` for NNLS-based nulls and `'sd'` for Savitzky-Golay or raw.
   - `null_N` (default 3.0) is the multiplier applied to the chosen spread.
   A single threshold computed from that null distribution is reused for pulse
   1 failures and the pulse 2/3 p-values. Reducing `f0_window_s` or disabling
   `allow_shift` tightens the null when pre-train baselines are short.
6. **Optional amplitude floor for PPR safety.** If
   `options['amplitude_floor_to_noise']=True`:
   - For each trial, after `thr1` and p-values are computed, **all pulse
     amplitudes** (`raw`, `savgol`, `nnls`, corrected and uncorrected) are
     floored as `amp := max(amp, thr1)`, then trial PPRs are computed from the
     floored amplitudes.
   - For the average trace, the same floor is applied using
     `median(threshold_amp1)` across trials, and average PPR is recomputed.
   - This is **not A1-only**: it applies to pulses 1..N.
   - It primarily stabilizes PPR denominators/ratios and can change exported
     AMP/PPR values; thresholds and p-values are computed before the floor.
   - For failure calls using `amp <= thr_shared` (as in the demo batch export),
     outcomes are unchanged by this floor.
   - In `demo_batch_process.py` and `demo_single_file.py`, this option is set
     to `True` in the `iglusnfr_optimized` preset.
7. **Baseline figures (optional).** Enable `options['plot']['baseline']=True`
   to save per-trial two-panel plots that show the baseline fit, null histogram
   and the stimulus train. This is useful when checking that the baseline mask
   and threshold rule match your expectations.

Related demo scripts in this folder (with concrete paths): `demo_single_file.py` and `demo_batch_process.py` (works for one or many folders).

---

## 1) Analyze a single file

Minimal usage (last column = time, other columns = trials):

```python
import numpy as np, pandas as pd
from extract_metrics import extract_metrics

xlsx_path = r"C:\\path\\to\\fiber.xlsx"
df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
# time in last column; trials in all columns except last
_time = pd.to_numeric(df.iloc[:, -1], errors='coerce').to_numpy(float)
_trials = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').to_numpy(float)
valid = np.isfinite(_time)
time = _time[valid]
trials = _trials[valid, :]

res = extract_metrics(
    time, trials,
    train_start=0.5,   # seconds
    isi=0.05,          # seconds
    n_pulses=10,
    options={
        'normalize_dff': True,
        'bleach': True,
        'measurement': 'NNLS',        # or 'SAVGOL', 'RAW'
        'fail_method': 'NNLS',        # default: same as measurement
        'threshold_mode': 'auto',     # or 'mad', 'sd'
        'allow_shift': True,
        'fit_source': 'global',
        'decay_progression_mode': 'linear',
        'plot': {
            'enabled': True,
            'traces': ['raw','savgol','nnls'],  # show all average overlays
            'show_decay': True,
            'trials': True,
            'baseline': True
        }
    }
)

# Inspect results
print("Averages (NNLS):", res['average']['amp_nnls'])
print("PPR (NNLS):", res['average']['ppr_nnls'])
print("A1 thresholds per trial:", res['threshold_amp1'])
print("A1 p-values per trial (using", 'NNLS', "):", res['pval_amp1'])

# Save plot if enabled
fig = res.get('figure')
if fig is not None:
    fig.savefig(r"C:\\path\\to\\fiber_plot.png", dpi=150)
```

Override the train start (seconds) for that file by passing `train_start=...`.

---

## 2) Analyze one folder

Process all `.xlsx` files in a directory, save one image per file and a summary CSV:

```python
import os, glob, zipfile, numpy as np, pandas as pd
from extract_metrics import extract_metrics

in_dir  = r"C:\\data\\groupA"
out_dir = r"C:\\out\\groupA"; os.makedirs(out_dir, exist_ok=True)
rows = []

def _is_valid_xlsx(path: str) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            return '[Content_Types].xml' in z.namelist()
    except Exception:
        return False

for xlsx_path in glob.glob(os.path.join(in_dir, "*.xlsx")):
    if not _is_valid_xlsx(xlsx_path):
        print(f"[skip] Not a valid .xlsx package: {xlsx_path}")
        continue
    try:
        df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
    except Exception as e:
        print(f"[skip] Failed to read Excel: {xlsx_path} -> {e}")
        continue
    t_raw = pd.to_numeric(df.iloc[:, -1], errors='coerce').to_numpy(float)
    X = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    ok = np.isfinite(t_raw)
    time = t_raw[ok]
    trials = X[ok, :]

    res = extract_metrics(
        time, trials,
        train_start=0.5, isi=0.05, n_pulses=10,
        options={'normalize_dff': True, 'bleach': True,
                 'fit_source': 'global', 'decay_progression_mode': 'linear',
                 'plot': {'enabled': True, 'traces': ['raw','savgol','nnls'], 'show_decay': True, 'trials': False}}
    )

    base = os.path.splitext(os.path.basename(xlsx_path))[0]
    if res.get('figure') is not None:
        res['figure'].savefig(os.path.join(out_dir, f"{base}.png"), dpi=150)

    row = {'file': base}
    amp = res['average']['amp_nnls']; ppr = res['average']['ppr_nnls']
    for i, v in enumerate(amp): row[f'amp_{i+1}'] = float(v)
    for i, v in enumerate(ppr): row[f'ppr_{i+1}'] = float(v)
    rows.append(row)

pd.DataFrame(rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
```

Tips:
- Vary `train_start` per directory: compute it from the folder name with a small map.
- Control which traces are drawn via `options['plot']['traces']`.

---

## 4) Multi‑folder → multi‑sheet Excel

Use the built‑in helper to export one sheet per folder with AMP and PPR columns (and `%Fail1..3` if available):

```python
from Feature_extraction.extract_metrics import export_folders_to_excel

folders = [r"C:\\data\\groupA", r"C:\\data\\groupB"]
export_folders_to_excel(
    folders,
    out_file=r"C:\\out\\ppr_results.xlsx",
    train_start=0.5,
    isi=0.05,
    n_pulses=10,
    options={'measurement': 'NNLS', 'fail_method': 'NNLS', 'threshold_mode': 'auto'}
)
```

---

## 3) Batch process multiple folders

Loop over several input folders; create one output subfolder per input. The last two below use a different `train_start` (0.5 s):

```python
import os, glob, zipfile, numpy as np, pandas as pd
from extract_metrics import extract_metrics

folders = [
    r"C:\\data\\groupA",
    r"C:\\data\\groupB",
    r"C:\\data\\groupA_05",
    r"C:\\data\\groupB_05",
]
root_out = r"C:\\out"; os.makedirs(root_out, exist_ok=True)

train_start_by_folder = {
    folders[2]: 0.5,
    folders[3]: 0.5,
}

def _is_valid_xlsx(path: str) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            return '[Content_Types].xml' in z.namelist()
    except Exception:
        return False

for in_dir in folders:
    out_dir = os.path.join(root_out, os.path.basename(in_dir))
    os.makedirs(out_dir, exist_ok=True)

    train_start = train_start_by_folder.get(in_dir, 1.0)
    isi = 0.05
    n_pulses = 10

    rows = []
    for xlsx_path in glob.glob(os.path.join(in_dir, "*.xlsx")):
        if not _is_valid_xlsx(xlsx_path):
            print(f"[skip] Not a valid .xlsx package: {xlsx_path}")
            continue
        try:
            df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
        except Exception as e:
            print(f"[skip] Failed to read Excel: {xlsx_path} -> {e}")
            continue
        t_raw = pd.to_numeric(df.iloc[:, -1], errors='coerce').to_numpy(float)
        X = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        ok = np.isfinite(t_raw)
        time = t_raw[ok]
        trials = X[ok, :]

        res = extract_metrics(
            time, trials,
            train_start=train_start, isi=isi, n_pulses=n_pulses,
            options={'normalize_dff': True, 'bleach': True,
                     'plot': {'enabled': True, 'traces': ['raw','savgol','nnls'], 'show_decay': True, 'trials': False}}
        )

        base = os.path.splitext(os.path.basename(xlsx_path))[0]
        if res.get('figure') is not None:
            res['figure'].savefig(os.path.join(out_dir, f"{base}.png"), dpi=150)

        row = {'file': base}
        amp = res['average']['amp_nnls']; ppr = res['average']['ppr_nnls']
        for i, v in enumerate(amp): row[f'amp_{i+1}'] = float(v)
        for i, v in enumerate(ppr): row[f'ppr_{i+1}'] = float(v)
        rows.append(row)

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
```

Headless batch (no windows) still writes figures if you call `figure.savefig(...)`. To suppress figure windows entirely, set `options['plot']['enabled'] = False`.

---

## 4) Adjust/overlay fitted model by recutting events (adjust_fit_model)

Use `adjust_fit_model.py` to re-cut windows around selected events (with optional peak alignment), average them, and overlay the current fitted model of the average trace.

CLI examples (paths are generic):

```bash
# Single folder, save figures
python adjust_fit_model.py C:\data\groupA \
  --train-start 0.5 --isi 0.05 --n-pulses 10 \
  --events 1-5,7 --align-by-peak \
  --pre-ms 2 --post-ms 200 \
  --out-dir C:\out --save

# Mix files and folders; disable ΔF/F0 and bleach if needed
python adjust_fit_model.py C:\data\groupA C:\data\fiber.xlsx \
  --train-start 1.0 --isi 0.05 --n-pulses 10 \
  --events all --pre-ms 2 --post-ms 200 \
  --no-dff --no-bleach --out-dir C:\out --save
```

Programmatic usage:

```python
from adjust_fit_model import process_file, process_folder, process_inputs

# Single file
process_file(
    r"C:\\data\\fiber.xlsx",
    train_start=0.5, isi=0.05, n_pulses=10,
    events_spec="1-5,7", peak_recenter=5,
    pre_ms=2.0, post_ms=200.0, out_dir=r"C:\\out", save=True
)

# Folder
process_folder(
    r"C:\\data\\groupA",
    train_start=1.0, isi=0.05, n_pulses=10,
    events_spec="all", peak_recenter=0,
    pre_ms=2.0, post_ms=200.0, out_dir=r"C:\\out", save=True
)
```

---

## Programmatic call (inside Python)

```python
from extract_metrics import extract_metrics

res = extract_metrics(
    time, trials,
    train_start=0.5, isi=0.05, n_pulses=10,
    options={
        # Only override what you need; the rest comes from DEFAULTS
        'normalize_dff': True,
        'bleach': True,
        'peak_window_ms': 25.0,
        'plot': {'enabled': True, 'traces': ['raw','savgol','nnls'], 'show_decay': True, 'trials': False, 'baseline': False}
    }
)
```

### Output structure
- `tau_r_s` (float): rise time (s)
- `tau_d_s` (array): per‑pulse decay times (s); constant in `global` mode unless anchors are enabled
- `stim_times_s` (array): stimulus times (s)
- `average` (dict): `y_avg`, `yhat_avg`, `amp_raw`, `amp_savgol`, `amp_nnls`, `ppr_nnls` (PPR = AMPn / AMP1)
- `per_trial` (list of dict): for each trial, the same amplitude/PPR triplets plus fitted `a_coeff`, `delta_s`, shared threshold (`thr_shared`), and `pval_amp1/2/3`
- `threshold_amp1` (array): MAD‑rule thresholds for pulse 1 per trial
- `pval_amp1` (array): empirical p‑values for pulse 1 per trial
- `figure` (matplotlib Figure or None): average trace with selected overlays

If `amplitude_floor_to_noise=True`, the amplitude arrays above are the
post-floor values used to compute PPR.

### Configuration via options
All defaults live in a single dictionary inside `extract_metrics.py` named `DEFAULTS`. You can override any of these keys in the `options` you pass to `extract_metrics`:

- Smoothing: `sg_window`, `sg_poly`
- Fit/plot window: `pre_zoom_s`, `post_zoom_s`
- Decay progression: `fit_source`, `decay_progression_mode`, `anchor_first_tau`, `anchor_final_tau`
- Peak window: `peak_window_ms`, `peak_avg_points`, `pre_peak_ms`
- Peak window controls determine how stimulus-locked maxima are located and
  quantified:
- `peak_window_ms` bounds how far after each stimulus `windowed_max`
    searches for the response (`[stimulus − pre_peak_ms, stimulus +
    peak_window_ms]`). Larger values permit slower peaks but can pull in
    unrelated fluctuations; smaller values tighten detection around rapid
    responses. The window is **clamped to ISI** so it never extends into the next pulse.
    The same window is used for null-sample amplitudes and bleach
    masking, so it shapes both detection sensitivity and thresholds.
  - `pre_peak_ms` extends the window before the stimulus, ensuring early-rising
    responses remain measurable. Increasing it can shorten the "quiet" segment
    reserved for bleach correction because the algorithm assumes anything in
    the window may belong to the evoked peak. It also **affects the local baseline**
    used for amplitude measurements.
  - `peak_avg_points` sets the symmetric sample count averaged around each
    detected maximum (±⌊N/2⌋). Higher values dampen noise and stabilize the
    reported amplitude, while lower values preserve temporal precision but are
    more sensitive to single-sample spikes. This averaging is applied to both
    real and null peaks so statistical thresholds stay consistent.
- Baseline/null: `f0_window_s`, `null_sim_max_points`, `null_min_post_zoom_s`, `null_N`
- Kinetics grids (ms): `kin_taur_grid_ms`, `kin_taud0_grid_ms`, `kin_slope_grid_ms`
  - If `kin_taur_grid_ms` / `kin_taud0_grid_ms` are **not provided**, they are auto‑built as
    **11‑point log‑spaced grids** within the active `parameter_bounds` (or defaults).
    Default bounds are τrise 0.5–3 ms and τfast 3–10 ms.
  - If you provide a grid, values **outside bounds are removed**; if nothing remains, the
    grid falls back to the auto‑built version.
- Robust NNLS + shifts: `huber_delta`, `irls_iters`, `delta_max_s`, `delta_step_s`, `shift_min_s`
- Bleach correction: `bleach_huber_delta`, `bleach_tau_range_factor`, `bleach_n_tau`
- Plot control: `plot = {'enabled': bool, 'traces': [...], 'show_decay': bool, 'trials': bool, 'baseline': bool, 'residual_buildup': bool, 'nnls_residual': bool, 'nnls_n_minus_1': bool}`

Notes:
- Provide `train_start`, `isi`, and `n_pulses` appropriate to each dataset.
- ΔF/F0 is used by default; disable by `options['normalize_dff'] = False` if you need raw ΔF.
- When progression rules specify monotonic increase (default for decay), τd is constrained non‑decreasing.
