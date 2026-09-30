"""
Adjust and visualize fit model by recutting, aligning, and averaging events.

Supports:
- Single recording (uid or legacy_id from release/boutons_manifest.csv)
- Single condition (process every recording in that condition)
- Multiple conditions/recordings (batch)

For each dataset, the script:
- Loads time and trials from release/raw/<uid>.csv (all non-time columns are
  trials, including any "average" column - matches the pipeline's prior xlsx
  behavior of treating every non-time column as a trial)
- Runs the streamlined pipeline (`extract_metrics`) to get the average trace
  and its fitted model (`yhat_avg`)
- Re-cuts windows around all or a subset of events (with optional peak
  realignment)
- Averages the recut segments (median across selected events)
- Overlays the current fitted model (recut in the same way) on top of the
  averaged waveform

Usage (CLI):
  python adjust_fit_model.py INPUT [INPUT ...] \
      --train-start 0.5 --isi 0.05 --n-pulses 10 \
      --events 1-5,7 --align-by-peak \
      --pre-ms 2 --post-ms 200 \
      --out-dir C:\\out --save

Notes:
- INPUT may be a uid, a legacy_id, or a condition name (looked up in
  release/boutons_manifest.csv). A condition name expands to every recording
  in that condition.
- The overlayed model is derived from the pipeline’s average-trace fit
  (`yhat_avg`) to reflect the current fitting model.
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# Ensure the repository root is on sys.path when running from this subfolder
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Feature_extraction.extract_metrics import extract_metrics
from smoothing import (
    build_median_recut_waveform,
    build_median_recut_figure,
    fit_template_decay,
)
from dataset_tools import raw_loader

DATA_ROOT = os.path.abspath(os.environ.get(
    "GLUSNFR_DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "PPR_DATA_FINAL")
))
RELEASE_DIR = os.path.join(DATA_ROOT, "release")


# -------------------------
# Helpers
# -------------------------

def _resolve_tokens_to_uids(tokens: List[str], manifest) -> List[str]:
    """Resolve CLI tokens (uid, legacy_id, or condition) to a flat, deduplicated
    list of uids, expanding a condition token to every recording in it."""
    uids: List[str] = []
    for tok in tokens:
        if (manifest["uid"] == tok).any():
            uids.append(tok)
            continue
        leg_rows = manifest[manifest["legacy_id"] == tok]
        if len(leg_rows) == 1:
            uids.append(str(leg_rows.iloc[0]["uid"]))
            continue
        if len(leg_rows) > 1:
            conditions = leg_rows["condition"].tolist()
            print(f"[skip] legacy_id={tok!r} is ambiguous across conditions "
                  f"{conditions}; pass the uid instead")
            continue
        cond_rows = manifest[manifest["condition"] == tok]
        if len(cond_rows) > 0:
            uids.extend(str(u) for u in cond_rows["uid"].tolist())
            continue
        print(f"[skip] Not a uid, legacy_id, or condition: {tok}")
    seen = set()
    out = []
    for u in uids:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _parse_events_spec(spec: Optional[str], n_pulses: int) -> List[int]:
    """Parse an events spec like "1,3-5,8" into 0-based pulse indices.

    Returns all pulses if spec is None/empty.
    Clamps to [0, n_pulses-1] and de-duplicates in ascending order.
    """
    if spec is None or str(spec).strip() == "" or str(spec).strip().lower() == "all":
        return list(range(n_pulses))
    out = []
    for chunk in str(spec).split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk:
            a, b = chunk.split('-', 1)
            try:
                i0 = int(a) - 1
                i1 = int(b) - 1
            except ValueError:
                continue
            if i0 > i1:
                i0, i1 = i1, i0
            out.extend(list(range(i0, i1 + 1)))
        else:
            try:
                out.append(int(chunk) - 1)
            except ValueError:
                continue
    # clamp and unique
    out = sorted({i for i in out if 0 <= i < int(n_pulses)})
    if not out:
        out = list(range(n_pulses))
    return out


DEFAULT_PEAK_RECENTER = 5


def _normalize_peak_recenter(value):
    """Coerce peak recenter options to an int or (pre, post) tuple."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return DEFAULT_PEAK_RECENTER if value else 0
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError('peak_recenter tuple/list must have length 2')
        lo, hi = value
        try:
            lo_i = int(float(lo))
            hi_i = int(float(hi))
        except (TypeError, ValueError) as exc:
            raise ValueError('peak_recenter limits must be numeric') from exc
        if lo_i < 0 or hi_i < 0:
            raise ValueError('peak_recenter limits must be non-negative')
        return (lo_i, hi_i)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {'', '0', 'none', 'false', 'off'}:
            return 0
        if ',' in s:
            parts = [p.strip() for p in s.split(',', 1)]
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError('peak_recenter string must be "pre,post"')
            try:
                lo_i = int(float(parts[0]))
                hi_i = int(float(parts[1]))
            except ValueError as exc:
                raise ValueError('peak_recenter components must be numeric') from exc
            if lo_i < 0 or hi_i < 0:
                raise ValueError('peak_recenter limits must be non-negative')
            return (lo_i, hi_i)
        try:
            val = int(float(s))
        except ValueError as exc:
            raise ValueError('peak_recenter must be an integer or "pre,post"') from exc
        if val < 0:
            raise ValueError('peak_recenter must be non-negative')
        return val
    try:
        val = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError('peak_recenter must be numeric or tuple') from exc
    if val < 0:
        raise ValueError('peak_recenter must be non-negative')
    return val


def _is_peak_recenter_active(value) -> bool:
    cfg = _normalize_peak_recenter(value)
    if isinstance(cfg, tuple):
        return any(v > 0 for v in cfg)
    return bool(cfg)


def _peak_recenter_label(value) -> str:
    cfg = _normalize_peak_recenter(value)
    if isinstance(cfg, tuple):
        if cfg[0] == 0 and cfg[1] == 0:
            return 'stim-aligned'
        return f'peak recenter [{cfg[0]}, {cfg[1]}]'
    if cfg:
        return f'peak recenter +/-{cfg}'
    return 'stim-aligned'


def _recut_median_from_series(
    t: np.ndarray,
    y: np.ndarray,
    stim_times: np.ndarray,
    event_idx: List[int],
    *,
    pre_ms: float,
    post_ms: float,
    peak_recenter: bool,
    peak_win_ms: float = 25.0,
    peak_search_pre_ms: float = 2.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Recut windows around selected events on a single series and return median.

    Uses the same utility as the smoothing module; passing a single-column series
    simply produces one snippet per selected event, whose median is returned.
    """
    if t is None or y is None or np.size(t) == 0 or np.size(y) == 0:
        return None, None
    stim_sel = np.asarray(stim_times, float)[event_idx]
    peak_cfg = _normalize_peak_recenter(peak_recenter)
    t_rel, med = build_median_recut_waveform(
        t, np.asarray(y, float)[:, None], stim_sel,
        pre_ms=float(pre_ms), post_ms=float(post_ms),
        peak_recenter=peak_cfg,
        peak_win_ms=float(peak_win_ms), peak_search_pre_ms=float(peak_search_pre_ms),
    )
    return t_rel, med


def _recut_median_from_trials(
    t: np.ndarray,
    Y_all: np.ndarray,
    stim_times: np.ndarray,
    event_idx: List[int],
    *,
    pre_ms: float,
    post_ms: float,
    peak_recenter: bool,
    peak_win_ms: float = 25.0,
    peak_search_pre_ms: float = 2.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Recut windows around selected events using ALL trials and return median.

    Produces one snippet per (event x trial) and returns the median across all
    snippets for a cleaner estimate of the event waveform.
    """
    if t is None or Y_all is None or np.size(t) == 0 or np.size(Y_all) == 0:
        return None, None
    stim_sel = np.asarray(stim_times, float)[event_idx]
    peak_cfg = _normalize_peak_recenter(peak_recenter)
    t_rel, med = build_median_recut_waveform(
        t, np.asarray(Y_all, float), stim_sel,
        pre_ms=float(pre_ms), post_ms=float(post_ms),
        peak_recenter=peak_cfg,
        peak_win_ms=float(peak_win_ms), peak_search_pre_ms=float(peak_search_pre_ms),
    )
    return t_rel, med


def _figure_title(base: str, events_spec: str, peak_recenter) -> str:
    ev = events_spec if events_spec else "all"
    align = _peak_recenter_label(peak_recenter)
    return f"{base} | events: {ev} | {align}"


def _resample_to_grid(t_src: np.ndarray, y_src: np.ndarray, t_ref: np.ndarray) -> np.ndarray:
    if t_src is None or y_src is None or t_ref is None:
        return None
    t_src = np.asarray(t_src, float)
    y_src = np.asarray(y_src, float)
    t_ref = np.asarray(t_ref, float)
    y_ref = np.full_like(t_ref, np.nan, dtype=float)
    m = (t_ref >= t_src[0]) & (t_ref <= t_src[-1])
    if np.any(m):
        y_ref[m] = np.interp(t_ref[m], t_src, y_src)
    return y_ref


# -------------------------
# Core per-file processing
# -------------------------

def process_file(
    uid: str,
    *,
    train_start: float,
    isi: float,
    n_pulses: int,
    events_spec: Optional[str] = None,
    peak_recenter=0,
    pre_ms: float = 2.0,
    post_ms: float = 200.0,
    normalize_dff: bool = True,
    bleach: bool = True,
    use_all_trials: bool = False,
    single_event_window: bool = False,
    guard_ms: float = 5.0,
    out_dir: Optional[str] = None,
    save: bool = False,
) -> Optional[str]:
    """Process a single recording (release/raw/<uid>.csv) and optionally save a
    figure. Returns path to saved figure if save=True and success, otherwise None.
    """
    try:
        manifest = raw_loader.load_manifest(DATA_ROOT)
        row = raw_loader.resolve_recording(manifest, uid=uid)
        raw = raw_loader.load_raw_csv(DATA_ROOT, uid)
    except Exception as e:
        print(f"[skip] Failed to load {uid}: {e}")
        return None
    ok = np.isfinite(raw.time_s)
    time, trials = raw.time_s[ok], raw.values[ok, :]

    peak_cfg = _normalize_peak_recenter(peak_recenter)

    # Run streamlined pipeline to get average trace and its model
    res = extract_metrics(
        time, trials,
        train_start=float(train_start), isi=float(isi), n_pulses=int(n_pulses),
        options={
            'normalize_dff': bool(normalize_dff),
            'bleach': bool(bleach),
            'plot': {'enabled': False}
        }
    )

    t = res['time_s']
    stim_times = res['stim_times_s']
    y_avg = res['average']['y_avg']
    yhat_avg = res['average']['yhat_avg']

    # Events selection
    idx = _parse_events_spec(events_spec, int(n_pulses))
    # Determine effective post window: optionally constrain to before next stimulus
    post_ms_eff = float(post_ms)
    if single_event_window and isi > 0:
        isi_ms = float(isi) * 1000.0
        post_ms_eff = max(0.0, min(float(post_ms), isi_ms - float(guard_ms)))

    # Recut source: either the average trace (default) or ALL trials for cleaner medians
    if use_all_trials:
        t_rel, med = _recut_median_from_trials(
            t, trials, stim_times, idx,
            pre_ms=pre_ms, post_ms=post_ms_eff, peak_recenter=peak_cfg,
        )
    else:
        t_rel, med = _recut_median_from_series(
            t, y_avg, stim_times, idx,
            pre_ms=pre_ms, post_ms=post_ms_eff, peak_recenter=peak_cfg,
        )
    _, med_model = _recut_median_from_series(
        t, yhat_avg, stim_times, idx,
        pre_ms=pre_ms, post_ms=post_ms_eff, peak_recenter=peak_cfg,
    )

    if t_rel is None or med is None:
        print(f"[warn] No recut median computed for {uid}")
        return None

    # Build figure with overlay
    base = str(row["legacy_id"])
    title = _figure_title(base, events_spec or "all", peak_cfg)
    fig = build_median_recut_figure(t_rel, med, fit_curve=med_model, title=title)

    if save and fig is not None:
        out_dir = out_dir or os.path.join(RELEASE_DIR, "adjust_fit_model_out")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{base}_adjust_fit.png")
        try:
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"[ok] Saved: {out_path}")
            return out_path
        except Exception as e:
            print(f"[warn] Failed to save figure for {uid}: {e}")
    else:
        try:
            plt.show(block=False); plt.pause(0.05)
        except Exception:
            pass
    return None


# -------------------------
# Folder / multi-folder drivers
# -------------------------

def process_folder(
    condition: str,
    *,
    train_start: float,
    isi: float,
    n_pulses: int,
    events_spec: Optional[str] = None,
    peak_recenter=0,
    pre_ms: float = 2.0,
    post_ms: float = 200.0,
    normalize_dff: bool = True,
    bleach: bool = True,
    use_all_trials: bool = False,
        single_event_window: bool = False,
        guard_ms: float = 5.0,
    out_dir: Optional[str] = None,
    aggregate: bool = False,
    aggregate_overlay: bool = False,
    sample_hz: float = 1000.0,
    save: bool = True,
) -> List[str]:
    """Process every recording in a condition; return list of saved figure paths."""
    out_dir = out_dir or os.path.join(RELEASE_DIR, "adjust_fit_model_out")
    peak_cfg = _normalize_peak_recenter(peak_recenter)
    manifest = raw_loader.load_manifest(DATA_ROOT)
    uids = raw_loader.iter_condition_rows(manifest, [condition])["uid"].tolist()
    paths = []
    agg_traces = []
    agg_trel = None
    for uid in uids:
        if not aggregate:
            out_path = process_file(
                uid,
                train_start=train_start, isi=isi, n_pulses=n_pulses,
                events_spec=events_spec, peak_recenter=peak_cfg,
                pre_ms=pre_ms, post_ms=post_ms,
                normalize_dff=normalize_dff, bleach=bleach,
                use_all_trials=use_all_trials,
                single_event_window=single_event_window, guard_ms=guard_ms,
                out_dir=out_dir, save=save,
            )
            if out_path:
                paths.append(out_path)
        else:
            # Aggregate: compute median recut per file (without drawing), then combine
            try:
                raw = raw_loader.load_raw_csv(DATA_ROOT, uid)
                ok = np.isfinite(raw.time_s)
                time, trials = raw.time_s[ok], raw.values[ok, :]
            except Exception:
                continue
            # basic extract for stim times
            res = extract_metrics(
                time, trials,
                train_start=float(train_start), isi=float(isi), n_pulses=int(n_pulses),
                options={'normalize_dff': bool(normalize_dff), 'bleach': bool(bleach), 'plot': {'enabled': False}},
            )
            t = res['time_s']; stim_times = res['stim_times_s']; y_avg = res['average']['y_avg']
            if use_all_trials:
                t_rel, med = _recut_median_from_trials(
                    t, trials, stim_times, _parse_events_spec(events_spec, int(n_pulses)),
                    pre_ms=pre_ms, post_ms=post_ms if not single_event_window else max(0.0, min(post_ms, isi*1000.0 - guard_ms)),
                    peak_recenter=peak_cfg,
                )
            else:
                t_rel, med = _recut_median_from_series(
                    t, y_avg, stim_times, _parse_events_spec(events_spec, int(n_pulses)),
                    pre_ms=pre_ms, post_ms=post_ms if not single_event_window else max(0.0, min(post_ms, isi*1000.0 - guard_ms)),
                    peak_recenter=peak_cfg,
                )
            if t_rel is None or med is None:
                continue
            if agg_trel is None:
                agg_trel = t_rel
            med_resampled = _resample_to_grid(t_rel, med, agg_trel)
            agg_traces.append(med_resampled)
    if aggregate and agg_traces:
        # Build a common 1 kHz grid and resample all traces
        pre_s = float(pre_ms) / 1000.0
        post_eff_ms = post_ms if not single_event_window else max(0.0, min(post_ms, isi * 1000.0 - guard_ms))
        post_s = float(post_eff_ms) / 1000.0
        dt = 1.0 / float(sample_hz)
        t_grid = np.arange(-pre_s, post_s + 1e-12, dt)
        traces_resampled = []
        for tr, t_rel in zip(agg_traces, [agg_trel] * len(agg_traces)):
            traces_resampled.append(_resample_to_grid(t_rel, tr, t_grid))
        S = np.vstack(traces_resampled)
        avg_all = np.nanmean(S, axis=0)

        # Plot overlays + average on top (no fit here)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        # overlay all
        for r in traces_resampled:
            ax.plot(t_grid * 1000.0, r, color='0.5', alpha=0.35, linewidth=1.0)
        # average on top
        ax.plot(t_grid * 1000.0, avg_all, color='tab:blue', linewidth=2.0, label='Average')
        ax.axvline(0.0, color='k', linestyle=':', linewidth=1.0)
        ax.set_xlabel('Time (ms)'); ax.set_ylabel('ΔF (median)')
        ax.set_title(_figure_title(condition, events_spec or 'all', peak_cfg))
        ax.legend(loc='best')

        # If overlay mode is requested, prefer showing without saving
        if aggregate_overlay or not save:
            try:
                plt.show(block=False); plt.pause(0.05)
            except Exception:
                pass
        else:
            try:
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{condition}_aggregate_overlay.png")
                fig.tight_layout(); fig.savefig(out_path, dpi=150)
                plt.close(fig)
                paths.append(out_path)
            except Exception:
                pass
    return paths


def process_inputs(
    inputs: List[str],
    *,
    train_start: float,
    isi: float,
    n_pulses: int,
    events_spec: Optional[str] = None,
    peak_recenter=0,
    pre_ms: float = 2.0,
    post_ms: float = 200.0,
    normalize_dff: bool = True,
    bleach: bool = True,
    use_all_trials: bool = False,
        single_event_window: bool = False,
        guard_ms: float = 5.0,
    out_dir: Optional[str] = None,
    aggregate: bool = False,
    aggregate_overlay: bool = False,
    sample_hz: float = 1000.0,
    save: bool = True,
) -> List[str]:
    """Process a list of inputs (uid, legacy_id, or condition tokens, looked up in
    release/boutons_manifest.csv). Returns saved figure paths."""
    peak_cfg = _normalize_peak_recenter(peak_recenter)
    manifest = raw_loader.load_manifest(DATA_ROOT)
    uids = _resolve_tokens_to_uids(inputs, manifest)
    saved = []
    if aggregate:
    # Aggregate across all resolved recordings
        agg_trel = None
        agg_traces = []
        label = inputs[0] if len(inputs) == 1 else "aggregate"
        def _collect_from_uid(uid: str):
            nonlocal agg_trel, agg_traces
            try:
                raw = raw_loader.load_raw_csv(DATA_ROOT, uid)
                ok = np.isfinite(raw.time_s)
                time, trials = raw.time_s[ok], raw.values[ok, :]
            except Exception:
                return
            res = extract_metrics(
                time, trials,
                train_start=float(train_start), isi=float(isi), n_pulses=int(n_pulses),
                options={'normalize_dff': bool(normalize_dff), 'bleach': bool(bleach), 'plot': {'enabled': False}},
            )
            t = res['time_s']; stim_times = res['stim_times_s']; y_avg = res['average']['y_avg']
            post_eff = post_ms if not single_event_window else max(0.0, min(post_ms, isi*1000.0 - guard_ms))
            if use_all_trials:
                t_rel, med = _recut_median_from_trials(
                    t, trials, stim_times, _parse_events_spec(events_spec, int(n_pulses)),
                    pre_ms=pre_ms, post_ms=post_eff, peak_recenter=peak_cfg,
                )
            else:
                t_rel, med = _recut_median_from_series(
                    t, y_avg, stim_times, _parse_events_spec(events_spec, int(n_pulses)),
                    pre_ms=pre_ms, post_ms=post_eff, peak_recenter=peak_cfg,
                )
            if t_rel is None or med is None:
                return
            if agg_trel is None:
                agg_trel = t_rel
            agg_traces.append(_resample_to_grid(t_rel, med, agg_trel))

        for uid in uids:
            _collect_from_uid(uid)
        if agg_traces and agg_trel is not None:
            # Build uniform grid at sample_hz and overlay all + mean
            pre_s = float(pre_ms) / 1000.0
            post_eff_ms = post_ms if not single_event_window else max(0.0, min(post_ms, isi * 1000.0 - guard_ms))
            post_s = float(post_eff_ms) / 1000.0
            dt = 1.0 / float(sample_hz)
            t_grid = np.arange(-pre_s, post_s + 1e-12, dt)
            traces_resampled = [
                _resample_to_grid(agg_trel, tr, t_grid) for tr in agg_traces
            ]
            S = np.vstack(traces_resampled)
            avg_all = np.nanmean(S, axis=0)

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            for r in traces_resampled:
                ax.plot(t_grid * 1000.0, r, color='0.5', alpha=0.35, linewidth=1.0)
            ax.plot(t_grid * 1000.0, avg_all, color='tab:blue', linewidth=2.0, label='Average')
            ax.axvline(0.0, color='k', linestyle=':', linewidth=1.0)
            ax.set_xlabel('Time (ms)'); ax.set_ylabel('ΔF (median)')
            ax.set_title(_figure_title(label, events_spec or 'all', peak_cfg))
            ax.legend(loc='best')

            if aggregate_overlay or not save:
                try:
                    plt.show(block=False); plt.pause(0.05)
                except Exception:
                    pass
            else:
                try:
                    out_dir_final = out_dir or os.path.join(RELEASE_DIR, "adjust_fit_model_out")
                    os.makedirs(out_dir_final, exist_ok=True)
                    out_path = os.path.join(out_dir_final, f"{label}_aggregate_overlay.png")
                    fig.tight_layout(); fig.savefig(out_path, dpi=150)
                    plt.close(fig)
                    saved.append(out_path)
                except Exception:
                    pass
        return saved
    for uid in uids:
        out_path = process_file(
            uid,
            train_start=train_start, isi=isi, n_pulses=n_pulses,
            events_spec=events_spec, peak_recenter=peak_cfg,
            pre_ms=pre_ms, post_ms=post_ms,
            normalize_dff=normalize_dff, bleach=bleach,
            use_all_trials=use_all_trials,
            single_event_window=single_event_window, guard_ms=guard_ms,
            out_dir=out_dir, save=save,
        )
        if out_path:
            saved.append(out_path)
    return saved


# -------------------------
# CLI
# -------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Recut/align events, average them, and overlay the current fitted model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("inputs", nargs="+", help="uid, legacy_id, and/or condition names (release/boutons_manifest.csv)")
    p.add_argument("--train-start", type=float, default=0.5, help="First stimulus time (s)")
    p.add_argument("--isi", type=float, default=0.05, help="Inter-stimulus interval (s)")
    p.add_argument("--n-pulses", type=int, default=10, help="Number of pulses in the train")
    p.add_argument("--events", type=str, default="all", help="Events to include, e.g. '1-5,7'")
    p.add_argument("--peak-recenter", type=str, default='0',
                   help="Peak realignment window in samples (int or 'pre,post'); 0 disables.")
    p.add_argument("--align-by-peak", action="store_true", dest="_legacy_align", help=argparse.SUPPRESS)
    p.add_argument("--pre-ms", type=float, default=2.0, help="Pre-stimulus window (ms)")
    p.add_argument("--post-ms", type=float, default=200.0, help="Post-stimulus window (ms)")
    p.add_argument("--no-dff", action="store_true", help="Disable ΔF/F0 normalization")
    p.add_argument("--no-bleach", action="store_true", help="Disable bleach correction")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory for figures (defaults to release/adjust_fit_model_out)")
    p.add_argument("--save", action="store_true", help="Save figures instead of showing interactively")
    p.add_argument("--use-all-trials", action="store_true", help="Recut using all trials (event x trial snippets) for a cleaner median")
    p.add_argument("--single-event-window", action="store_true", help="Truncate post window to before next stimulus (use guard-ms margin)")
    p.add_argument("--guard-ms", type=float, default=5.0, help="Safety margin (ms) before the next stimulus when using single-event-window")
    p.add_argument("--aggregate", action="store_true", help="Aggregate across all files/folders into a single figure")
    p.add_argument("--aggregate-overlay", action="store_true", help="Overlay all selected traces and draw the average on top; do not save unless --save is explicitly set")
    p.add_argument("--sample-hz", type=float, default=1000.0, help="Resampling rate for aggregation (Hz)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    ap = _build_argparser()
    args = ap.parse_args(argv)

    normalize_dff = not args.no_dff
    bleach = not args.no_bleach
    peak_cfg = _normalize_peak_recenter(args.peak_recenter)
    if getattr(args, '_legacy_align', False) and not _is_peak_recenter_active(peak_cfg):
        peak_cfg = DEFAULT_PEAK_RECENTER

    saved = process_inputs(
        args.inputs,
        train_start=args.train_start,
        isi=args.isi,
        n_pulses=args.n_pulses,
        events_spec=args.events,
        peak_recenter=peak_cfg,
        pre_ms=args.pre_ms,
        post_ms=args.post_ms,
        normalize_dff=normalize_dff,
        bleach=bleach,
    use_all_trials=args.use_all_trials,
    single_event_window=args.single_event_window,
    guard_ms=args.guard_ms,
        out_dir=args.out_dir,
    aggregate=args.aggregate,
    aggregate_overlay=args.aggregate_overlay,
    sample_hz=args.sample_hz,
        save=args.save,
    )

    if args.save and saved:
        print(f"Saved {len(saved)} figure(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

