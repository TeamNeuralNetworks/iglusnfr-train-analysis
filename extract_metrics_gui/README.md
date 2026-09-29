# Event fitting workbench

An interactive Tkinter front end for the `Feature_extraction.extract_metrics`
workflow. It exposes the whole model library and the option surface, and guides
the fit visually instead of asking for a set of numbers up front.

```powershell
python -m extract_metrics_gui.app
```

Headless check of everything the interface depends on (no window opened):

```powershell
python -m extract_metrics_gui.validate_one_file
```

## Two loops

The interface is built around the fact that the two things you want to tune have
very different costs.

**The inner loop** fits one model to the average event snippet, recut directly
from the traces. It takes about 10 ms, so it runs while you drag a slider. The
*Event fit* tab shows the snippets, their projection, the model overlay and a
residual strip, with R², RMS and AIC.

**The outer loop** is the full `extract_metrics` train extraction. It takes
several seconds to half a minute depending on the options, so it runs on a
worker thread with a progress indicator; the window stays responsive.

Dragging a slider only *evaluates* the model. Refitting happens when you press
**Fit event**, change model, apply a preset, or load a file.

## Bounds are the control that matters

`extract_metrics` derives the train kinetics from its own recut fit, constrained
by `parameter_bounds`. That dictionary — not the parameter values — is what
determines the result, so every parameter row carries editable bounds, and
**Send these bounds to the train fit** passes them through.

This is not a cosmetic setting. On the bundled validation trace:

| `tau_decay_fast` bound | fitted τd | AMP1 | train R² |
| --- | --- | --- | --- |
| (3, 10) ms — the built-in default | 10.0 ms (pinned) | 0.157826 | 0.822 |
| (3, 30) ms | 22.0 ms | 0.128035 | 0.997 |

The default upper bound clamps the decay and biases AMP1 by about 19 %. When a
fitted parameter lands on a bound the row shows **at min** / **at max** and the
event panel says so, because that is the signal that the bound, not the data, is
setting the answer.

Note that `event_model_settings` is deliberately not wired to a control: the
fitter overrides those values with its own recut fit, so a slider for them would
do nothing.

## What is exposed

- **Models** — all of them, read from `Model_Calibration.event_models` at import
  time via `registry.py`. Adding a model to that registry makes it appear here
  with its parameters, bounds and initial guesses; nothing is hard-coded. The
  menu labels each model with its parameter count and whether `extract_metrics`
  treats it as a τ-varying kernel or a fixed-shape template.
- **Options** — 43 documented options in `options.py`, grouped by what they
  affect, each with a tooltip. Advanced entries are hidden behind a toggle.
- **Presets** — *extract_metrics defaults*, *double-exp simple*, and
  *iGluSnFR optimised* (the preset from `Feature_extraction/demo_batch_process.py`).
  Applying a preset resets to schema defaults first, so it always describes a
  complete, reproducible state.

## Input formats

- **CSV / TSV / text** with a time column (`tim`, `time`, `time_s`, `t`, …, or
  the first increasing numeric column) plus one or more trace columns. A column
  named `average` is ignored, as is an unnamed trailing column that correlates
  above 0.995 with the mean of the others. This covers the release `raw/*.csv`
  layout.
- **Excel** (`.xlsx` / `.xlsm` / `.xls`), first sheet, needs `openpyxl`.
- **NumPy bundles** (`.npz`) as written by the pipeline: `time_s` plus
  `all_trial_yproc` or `selected_trial_yproc`. Train geometry is read from
  `train_start_s` / `isi_s` when present.

## Train detection

**Detect train from data** estimates start, ISI and pulse count from the average
trace. The derivative threshold is swept rather than fixed — a workable value
depends on the noise level — and the candidate whose onsets sit most regularly on
a constant-ISI grid wins. Off-grid detections are rejected, then the count is
extended along that grid so later pulses riding on an elevated baseline are still
counted; a slot counts when it both clears the expected noise maximum and stays
elevated on average.

It reports a confidence, and every field stays editable. On the bundled test set
it recovers the exact geometry of both real files and 35 of 36 synthetic trains,
with no false positives on pure noise.

## Layout

| Panel | Contents |
| --- | --- |
| Left | Data & train, event model with its parameter rows, grouped options |
| Event fit | Snippets, projection, model overlay, residual, R²/RMS/AIC |
| Train fit | Average trace, fitted model, residual, stimulus markers, train R² |
| Amplitudes | Per-pulse amplitude, ratio to pulse 1, and decay τ, on separate axes |
| Results | Per-pulse table and the metric list |

Amplitude and PPR get their own panels on purpose: they are proportional by
construction, so on twin axes one curve hides the other exactly.

**Export results** writes `*_pulses.csv`, `*_metrics.csv` and `*_config.json`.

## Taking the settings out of the GUI

**Generate script** writes a standalone, runnable `.py` that reproduces the
current state: every option written out literally and annotated from the same
schema that drives the tooltips, the bounds as a plain dict, a `run()` for one
file and a `run_folder()` for batch work. It reproduces the workbench result
exactly — same file, same options, same AMP1 to every digit.

## Loading a previous run

**Load config** accepts two shapes:

- the workbench's own `*_config.json` (model, parameters, bounds, locks, train
  geometry, options);
- `run_settings.json`, the reproducibility log written by
  `Feature_extraction/demo_batch_process.py`. Files covering several conditions
  prompt for which one to load, since ISI and baseline differ per condition.

A batch log records more options than the workbench has widgets for. Those are
**kept verbatim and forwarded to the fitter**, so loading a batch configuration
and re-running it reproduces that run rather than quietly reverting to defaults;
the status line lists which ones are being carried invisibly.

Import is lossless. Loading `sample_data/run_settings_example.json` and
rebuilding gives back the same 54 options with **no changed values** and a
`parameter_bounds` dict identical to the one in the file. The single exception
is `nnls_weight_tau_s`, dropped when it is `None` — which is also its default,
so the fitter behaves the same.

Only bounds you actually **edit** are sent. Rows left at their registry defaults
already show the limits the fit applies anyway, so forwarding them would add
constraints a loaded run never carried. Untick **Send edited bounds to the train
fit** to send none at all.

If you import an older log written before the preset was cleaned up, two keys
are handled specially and reported in the interface: `tau_superslow` is renamed
to `tau_decay_superslow`, and `amplitude_ratio` is dropped. See below.

### Two dead bounds in the batch preset

`demo_batch_process.py` carried two `parameter_bounds` entries that
`extract_metrics` never read, so neither has ever constrained anything:

- **`tau_superslow: (0.035, 0.090)`** — the parameter is `tau_decay_superslow`.
  Enabling it under the correct name *does* change results: on the bundled
  validation trace AMP10 moves by 5.6 % of AMP1 and the train R² drops from
  0.998 to 0.993. It is therefore left commented out, so runs stay comparable
  with the published dataset. Uncommenting one line applies it.
- **`amplitude_ratio: (0.0, 1.0)`** — no registered model has a parameter of that
  name. The tri-exponential mixing weights are `frac_fast` and `frac_slow`, and
  `(0, 1)` is their full range in any case. Removed.

Exported logs now contain only bounds the fitter honours, which is what makes
the import above lossless. The `iGluSnFR optimised` preset was aligned to match.

### Note on `run_settings.json`

If an older output folder has no `run_settings.json`, that is not a design
change. `_write_run_log` referenced two undefined names (`ISI_BY_CONDITION`,
`BASELINE_BY_CONDITION`) and raised `NameError` on every run since it was added;
the caller caught it and only printed a line, so the log was never written. Both
are fixed — the per-condition values it was reaching for are in
`resolved_options_by_condition`, and the manifest path is recorded instead — and
the handler now prints a traceback rather than swallowing the cause. Re-running
the batch produces the log; it cannot be recovered for past runs.

## Files

| File | Role |
| --- | --- |
| `app.py` | Tk interface, threading, panel updates |
| `core.py` | Inner-loop event fit, outer-loop train fit, figures, export |
| `registry.py` | Introspection over the model library |
| `options.py` | Option schema, help text and presets |
| `loaders.py` | Format handling and train detection |
| `scriptgen.py` | Renders a standalone runnable script from a configuration |
| `runsettings.py` | Reads workbench configs and batch `run_settings.json` |
| `widgets.py` | Tooltip, scroll frame, collapsible section, parameter row |
| `theme.py` | Light/dark palette for both Tk and matplotlib |
