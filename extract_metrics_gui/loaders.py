"""Trace loading and stimulus-train detection.

Handles the formats that actually turn up around this repository - CSV
(including the release ``raw/*.csv`` layout), Excel, and the ``.npz`` bundles
written by the extraction pipeline - and estimates the train geometry so the
user is not asked to type numbers they can measure from the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


CSV_SUFFIXES = (".csv", ".txt", ".tsv")
EXCEL_SUFFIXES = (".xlsx", ".xlsm", ".xls")
NPZ_SUFFIXES = (".npz",)

SUPPORTED_SUFFIXES = CSV_SUFFIXES + EXCEL_SUFFIXES + NPZ_SUFFIXES

TIME_CANDIDATES = ("tim", "time", "times", "time_s", "t", "seconds", "sec", "timestamp")
AVERAGE_NAMES = ("average", "avg", "mean")


@dataclass
class TraceData:
    time_s: np.ndarray
    trials: np.ndarray            # (n_samples, n_trials)
    time_column: str
    trial_columns: List[str]
    source: Path
    notes: List[str]
    train_start: Optional[float] = None   # filled from file metadata when present
    isi: Optional[float] = None
    n_pulses: Optional[int] = None

    @property
    def n_trials(self) -> int:
        return int(self.trials.shape[1])

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0]) if self.time_s.size else 0.0

    @property
    def sample_rate_hz(self) -> float:
        if self.time_s.size < 2:
            return float("nan")
        dt = float(np.median(np.diff(self.time_s)))
        return 1.0 / dt if dt > 0 else float("nan")

    def average(self) -> np.ndarray:
        return np.nanmean(self.trials, axis=1)


def _normalise(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def _detect_time_column(frame: pd.DataFrame, hint: str = "") -> str:
    lookup = {_normalise(c): str(c) for c in frame.columns}
    for candidate in ([hint] if hint else []) + list(TIME_CANDIDATES):
        key = _normalise(candidate)
        if key and key in lookup:
            return lookup[key]
    # Fall back to the first monotonically increasing numeric column.
    for col in frame.columns:
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
        finite = values[np.isfinite(values)]
        if finite.size >= 4 and np.all(np.diff(finite) >= 0):
            return str(col)
    raise ValueError(
        "No time column found. Name one 'time' or 'tim', or set the time-column hint."
    )


def _looks_like_average(candidate: np.ndarray, others: np.ndarray) -> bool:
    if others.shape[1] < 2:
        return False
    mean_trace = np.nanmean(others, axis=1)
    finite = np.isfinite(candidate) & np.isfinite(mean_trace)
    if finite.sum() <= 10:
        return False
    corr = np.corrcoef(candidate[finite], mean_trace[finite])[0, 1]
    return bool(np.isfinite(corr) and corr > 0.995)


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in EXCEL_SUFFIXES:
        try:
            return pd.read_excel(path, sheet_name=0)
        except ImportError as exc:
            raise ValueError(
                "Reading Excel needs the 'openpyxl' package (pip install openpyxl), "
                "or export the sheet to CSV."
            ) from exc
    sep = "\t" if suffix == ".tsv" else None
    return pd.read_csv(path, sep=sep, engine="python" if sep is None else "c")


def _load_table(path: Path, hint: str) -> TraceData:
    frame = _read_frame(path)
    if frame.shape[1] < 2:
        raise ValueError("Expected a time column plus at least one trace column.")

    notes: List[str] = []
    time_column = _detect_time_column(frame, hint)
    time_raw = pd.to_numeric(frame[time_column], errors="coerce").to_numpy(float)

    trial_columns = [str(c) for c in frame.columns if str(c) != time_column]
    # Drop an explicit average column (the release raw/*.csv layout has one).
    named_average = [c for c in trial_columns if _normalise(c) in AVERAGE_NAMES]
    if named_average and len(trial_columns) > len(named_average):
        for c in named_average:
            trial_columns.remove(c)
        notes.append(f"ignored average column '{named_average[0]}'")
    if not trial_columns:
        raise ValueError("No trace columns left after removing the time column.")

    trials = frame[trial_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if trials.ndim == 1:
        trials = trials[:, None]

    # Otherwise fall back to detecting an unnamed trailing average.
    if not named_average and trials.shape[1] >= 3 and _looks_like_average(trials[:, -1], trials[:, :-1]):
        notes.append(f"ignored trailing average-like column '{trial_columns[-1]}'")
        trials = trials[:, :-1]
        trial_columns = trial_columns[:-1]

    valid = np.isfinite(time_raw)
    time_s, trials = time_raw[valid], trials[valid, :]
    keep = np.any(np.isfinite(trials), axis=1)
    time_s, trials = time_s[keep], trials[keep, :]
    if time_s.size < 20:
        raise ValueError("Fewer than 20 usable samples after cleaning the file.")

    return TraceData(
        time_s=np.asarray(time_s, float), trials=np.asarray(trials, float),
        time_column=time_column, trial_columns=trial_columns, source=path, notes=notes,
    )


def _load_npz(path: Path) -> TraceData:
    bundle = np.load(path, allow_pickle=True)
    keys = set(bundle.files)

    time_key = next((k for k in ("time_s", "time", "t", "tim") if k in keys), None)
    if time_key is None:
        raise ValueError(f"No time array in {path.name}. Looked for time_s / time / t / tim.")
    time_s = np.asarray(bundle[time_key], float).ravel()

    notes: List[str] = []
    trials = None
    columns: List[str] = []
    for key in ("all_trial_yproc", "trials", "y_trials", "all_trial_y"):
        if key in keys:
            block = np.asarray(bundle[key], float)
            if block.ndim == 2:
                # Stored trial-major; extract_metrics wants samples as rows.
                trials = block.T if block.shape[1] == time_s.size else block
                columns = [f"trial_{i + 1}" for i in range(trials.shape[1])]
                notes.append(f"trials from '{key}'")
                break
    if trials is None:
        for key in ("selected_trial_yproc", "average_yproc", "y_avg"):
            if key in keys:
                trials = np.asarray(bundle[key], float).reshape(-1, 1)
                columns = [key]
                notes.append(f"single trace from '{key}'")
                break
    if trials is None:
        raise ValueError(
            f"No trace array in {path.name}. Looked for all_trial_yproc / trials / selected_trial_yproc."
        )
    if trials.shape[0] != time_s.size:
        raise ValueError(
            f"Trace length {trials.shape[0]} does not match the time vector ({time_s.size})."
        )

    def _scalar(key: str):
        if key not in keys:
            return None
        try:
            value = float(np.asarray(bundle[key]).ravel()[0])
            return value if np.isfinite(value) else None
        except Exception:
            return None

    train_start = _scalar("train_start_s")
    isi = _scalar("isi_s")
    n_pulses = None
    for key in ("average_amp_nnls", "average_amp_savgol", "tau_d_s"):
        if key in keys:
            n_pulses = int(np.asarray(bundle[key]).ravel().size)
            break
    if train_start is not None and isi is not None:
        notes.append("train geometry read from file metadata")

    return TraceData(
        time_s=time_s, trials=np.asarray(trials, float), time_column=time_key,
        trial_columns=columns, source=path, notes=notes,
        train_start=train_start, isi=isi, n_pulses=n_pulses,
    )


def load_traces(path: str, time_column_hint: str = "") -> TraceData:
    """Load a trace file, dispatching on extension."""
    target = Path(path)
    if not target.exists():
        raise ValueError(f"File not found: {target}")
    suffix = target.suffix.lower()
    if suffix in NPZ_SUFFIXES:
        return _load_npz(target)
    if suffix in EXCEL_SUFFIXES or suffix in CSV_SUFFIXES:
        return _load_table(target, time_column_hint)
    # Unknown extension: try the table reader, it copes with most text files.
    return _load_table(target, time_column_hint)


# --- stimulus-train detection ----------------------------------------------

@dataclass
class TrainGuess:
    train_start: float
    isi: float
    n_pulses: int
    confidence: float          # 0-1, from the regularity of the detected onsets
    message: str


def _onsets_at(t: np.ndarray, slope: np.ndarray, dt: float, k: float) -> np.ndarray:
    """Onset times where the derivative exceeds k robust deviations."""
    mad = float(np.nanmedian(np.abs(slope - np.nanmedian(slope))))
    if not np.isfinite(mad) or mad <= 0:
        return np.array([])
    above = slope > np.nanmedian(slope) + k * 1.4826 * mad
    if not np.any(above):
        return np.array([])
    edges = np.flatnonzero(np.diff(above.astype(int)) == 1) + 1
    if above[0]:
        edges = np.r_[0, edges]
    onsets: List[float] = []
    for start in edges:
        if onsets and t[start] - onsets[-1] < 3 * dt:
            continue
        onsets.append(float(t[start]))
    return np.asarray(onsets, float)


def _fit_grid(onsets: np.ndarray, *, rounds: int = 3, reject: float = 0.25):
    """Fit onset index -> time, dropping points that fall off the regular grid.

    Returns (train_start, isi, kept_onsets, jitter_s) or None.
    """
    kept = np.asarray(onsets, float)
    for _ in range(rounds):
        if kept.size < 2:
            return None
        idx = np.arange(kept.size, dtype=float)
        slope, intercept = np.polyfit(idx, kept, 1)
        if not np.isfinite(slope) or slope <= 0:
            return None
        residual = kept - (intercept + slope * idx)
        good = np.abs(residual) <= reject * slope
        if good.all():
            jitter = float(np.sqrt(np.mean(residual ** 2)))
            return float(intercept), float(slope), kept, jitter
        if good.sum() < 2:
            return None
        kept = kept[good]
    idx = np.arange(kept.size, dtype=float)
    slope, intercept = np.polyfit(idx, kept, 1)
    residual = kept - (intercept + slope * idx)
    return float(intercept), float(slope), kept, float(np.sqrt(np.mean(residual ** 2)))


# Responses rise within a few milliseconds of the stimulus, so the search slot
# is capped rather than scaled with the ISI: a window that grows with the ISI
# simply collects more noise samples, and its running maximum grows with it.
MAX_RISE_WINDOW_S = 0.05


def _slot_rise(t: np.ndarray, y: np.ndarray, start: float, isi: float,
               baseline: float = 0.0) -> Tuple[float, int, float]:
    """Measure one stimulus slot.

    Returns the peak-minus-onset excursion, the sample count, and the mean
    elevation above `baseline`. The elevation is what separates a real response
    from a lucky noise excursion: noise averages back to the baseline over the
    slot, a decaying response does not.
    """
    span = min(0.6 * isi, MAX_RISE_WINDOW_S)
    window = (t >= start) & (t <= start + span)
    count = int(window.sum())
    if count < 3:
        return float("nan"), count, float("nan")
    segment = y[window]
    rise = float(np.nanmax(segment) - segment[0])
    elevation = float(np.nanmean(segment) - baseline)
    return rise, count, elevation


def _noise_ceiling(noise: float, n_samples: int) -> float:
    """Expected largest excursion of `n_samples` noise draws.

    max|N(0, s)| over n samples grows like s * sqrt(2 ln n), so comparing a
    rise against a flat multiple of the noise misjudges long windows.
    """
    return float(noise) * float(np.sqrt(2.0 * np.log(max(n_samples, 3))))


def _baseline_stats(t: np.ndarray, y: np.ndarray, train_start: float) -> Tuple[float, float]:
    """Robust (level, noise) of the pre-train baseline."""
    pre = t < train_start
    if pre.sum() >= 8:
        baseline = y[pre]
        level = float(np.nanmedian(baseline))
        return level, 1.4826 * float(np.nanmedian(np.abs(baseline - level)))
    return float(np.nanmedian(y)), float(np.nanstd(y)) * 0.1


def _count_pulses_on_grid(t: np.ndarray, y: np.ndarray, train_start: float, isi: float,
                          *, found: int, level: float, noise: float,
                          max_pulses: int = 500, max_misses: int = 2) -> int:
    """Count stimulus slots that still respond, walking a regular ISI grid.

    A slot counts when it both rises above the expected noise maximum and stays
    elevated on average. Magnitude alone is not enough: over a long run of
    trailing noise some slot always produces a large excursion by chance, which
    makes the train look far longer than it is.
    """
    if not np.isfinite(noise) or noise <= 0:
        return found

    reference, ref_n, _ = _slot_rise(t, y, train_start, isi, level)
    floor = _noise_ceiling(noise, ref_n)
    if np.isfinite(reference):
        floor = max(floor, 0.15 * reference)
    # The mean of n noise samples has standard error noise/sqrt(n), so this is
    # several standard errors for any usable slot.
    elevation_floor = 0.5 * noise

    count, misses = 0, 0
    for pulse in range(max_pulses):
        start = train_start + pulse * isi
        if start > t[-1]:
            break
        rise, _, elevation = _slot_rise(t, y, start, isi, level)
        if not np.isfinite(rise):
            break
        if rise > floor and np.isfinite(elevation) and elevation > elevation_floor:
            count = pulse + 1
            misses = 0
        else:
            misses += 1
            if misses > max_misses:
                break
    return max(count, found)


def detect_train(time_s: np.ndarray, trials: np.ndarray, *,
                 min_pulses: int = 2) -> Optional[TrainGuess]:
    """Estimate train start, ISI and pulse count from the average trace.

    Onsets are clustered upward excursions of the derivative. The detection
    threshold is swept rather than fixed, because a workable value depends on
    how noisy the recording is; the candidate whose onsets sit most regularly
    on a constant-ISI grid wins. Off-grid detections are then rejected and the
    pulse count is extended along that grid, so later events riding on an
    elevated baseline are still counted.
    """
    t = np.asarray(time_s, float).ravel()
    y = np.nanmean(np.asarray(trials, float), axis=1) if trials.ndim == 2 else np.asarray(trials, float)
    if t.size < 32 or t.size != y.size:
        return None

    finite = np.isfinite(t) & np.isfinite(y)
    t, y = t[finite], y[finite]
    if t.size < 32:
        return None

    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return None

    # Light smoothing so single-sample noise does not create onsets.
    width = max(3, int(round(0.001 / dt)) | 1)
    kernel = np.ones(width) / width
    smooth = np.convolve(y - np.nanmedian(y[: max(8, t.size // 20)]), kernel, mode="same")
    slope = np.gradient(smooth, dt)

    best = None  # (score, train_start, isi, n_detected, jitter)
    for k in (8.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2.0):
        onsets = _onsets_at(t, slope, dt, k)
        if onsets.size < min_pulses:
            continue
        fitted = _fit_grid(onsets)
        if fitted is None:
            continue
        train_start, isi, kept, jitter = fitted
        if not np.isfinite(isi) or isi <= 3 * dt or kept.size < min_pulses:
            continue
        # Reject grids whose first slot carries no real response: those come
        # from thresholding a stretch of noise, not from a stimulus train.
        level, noise = _baseline_stats(t, y, train_start)
        reference, ref_n, _ = _slot_rise(t, y, train_start, isi, level)
        if not (np.isfinite(noise) and noise > 0 and np.isfinite(reference)
                and reference > 1.5 * _noise_ceiling(noise, ref_n)):
            continue
        # Prefer many onsets that sit tightly on the grid.
        regularity = 1.0 - min(jitter / (0.25 * isi), 1.0)
        score = kept.size * (0.25 + 0.75 * regularity)
        if best is None or score > best[0]:
            best = (score, train_start, isi, int(kept.size), jitter, level, noise)

    if best is None:
        return None
    _, train_start, isi, n_detected, jitter, level, noise = best

    # Derivative thresholding misses later pulses once they ride on an elevated,
    # slowly decaying baseline. Walk the regular grid forward instead and keep
    # counting while each slot still carries a rise above the baseline noise.
    n_pulses = _count_pulses_on_grid(t, y, train_start, isi, found=n_detected,
                                     level=level, noise=noise)
    regularity = float(np.clip(1.0 - (jitter / (0.25 * isi) if isi > 0 else 1.0), 0.0, 1.0))
    # Slots we had to infer from the grid are weaker evidence than detected
    # onsets, and a grid fitted through two points is barely evidence at all.
    coverage = float(np.clip(n_detected / max(n_pulses, 1), 0.0, 1.0))
    support = float(np.clip(n_detected / 4.0, 0.0, 1.0))
    confidence = float(np.clip(regularity * (0.5 + 0.5 * coverage) * support, 0.0, 1.0))

    return TrainGuess(
        train_start=train_start, isi=isi, n_pulses=n_pulses, confidence=confidence,
        message=(
            f"{n_pulses} pulses at {1.0 / isi:.1f} Hz "
            f"(ISI {isi * 1000:.1f} ms), start {train_start:.4f} s, jitter {jitter * 1000:.2f} ms"
        ),
    )


def describe(data: TraceData) -> str:
    """One-line dataset summary for the status area."""
    rate = data.sample_rate_hz
    rate_text = f"{rate:.0f} Hz" if np.isfinite(rate) else "irregular sampling"
    parts = [
        f"{data.n_trials} trace{'s' if data.n_trials != 1 else ''}",
        f"{data.time_s.size} samples",
        f"{data.duration_s:.3f} s",
        rate_text,
        f"time '{data.time_column}'",
    ]
    if data.notes:
        parts.extend(data.notes)
    return " · ".join(parts)
