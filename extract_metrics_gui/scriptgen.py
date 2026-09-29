"""Emit a standalone, runnable extract_metrics script from a GUI configuration.

The point is to leave the GUI with something you can keep: every option written
out literally and annotated with what it does, so the script is both a record of
the run and a starting point for batch processing.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from . import options as opt


# The module docstring is raw: it embeds a Windows source path, and a bare
# docstring would read "C:\\Users..." as a truncated \\U unicode escape.
HEADER = '''r"""extract_metrics run generated from the event fitting workbench.

Generated {stamp}
Source file  : {source}
Event model  : {model}
Train        : start {start} s, ISI {isi} s, {pulses} pulses

Run it with:
    python "{script_name}"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(r"{repo_root}")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Feature_extraction.extract_metrics import extract_metrics
from extract_metrics_gui.loaders import load_traces
'''


BODY = '''

# --------------------------------------------------------------------------
# What to fit
# --------------------------------------------------------------------------
TRACE_FILE = r"{source}"
TIME_COLUMN_HINT = {hint!r}

TRAIN_START = {start!r}   # seconds, first stimulus
ISI = {isi!r}             # seconds between stimuli
N_PULSES = {pulses!r}

EVENT_MODEL = {model!r}

# Bounds on the fitted kinetics. These constrain the values extract_metrics
# derives from its own recut fit, and are the setting that most changes the
# result -- a parameter pinned at a bound means the bound, not the data, is
# deciding the answer.
PARAMETER_BOUNDS = {bounds}

# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------
OPTIONS = {{
{options}
}}


def build_options() -> dict:
    options = dict(OPTIONS)
    options["event_model"] = EVENT_MODEL
    options["plot"] = {{"enabled": False}}   # set True for the fitter's own figures
    if PARAMETER_BOUNDS:
        options["parameter_bounds"] = {{k: tuple(v) for k, v in PARAMETER_BOUNDS.items()}}
    return options


def run(path: str = TRACE_FILE, *, train_start: float = TRAIN_START,
        isi: float = ISI, n_pulses: int = N_PULSES) -> dict:
    """Fit one file and return the raw extract_metrics result."""
    data = load_traces(path, TIME_COLUMN_HINT)
    return extract_metrics(
        time=data.time_s,
        trials=data.trials,
        train_start=float(train_start),
        isi=float(isi),
        n_pulses=int(n_pulses),
        options=build_options(),
        filename=Path(path).stem,
    )


def pulse_table(result: dict, n_pulses: int = N_PULSES) -> pd.DataFrame:
    """Per-pulse amplitudes for the configured measurement."""
    average = result["average"]
    measurement = str(OPTIONS.get("measurement", "NNLS")).upper()
    uncorrected_key, corrected_key = {{
        "SAVGOL": ("amp_savgol", "amp_savgol_corr"),
        "RAW": ("amp_raw", "amp_raw_corr"),
    }}.get(measurement, ("amp_nnls", "amp_nnls_corr"))

    def series(key):
        values = np.asarray(average.get(key, []), float).ravel()
        out = np.full(n_pulses, np.nan)
        out[: min(n_pulses, values.size)] = values[: min(n_pulses, values.size)]
        return out

    corrected = series(corrected_key)
    tau_d = np.asarray(result.get("tau_d_s", []), float).ravel()
    taus = np.full(n_pulses, np.nan)
    taus[: min(n_pulses, tau_d.size)] = tau_d[: min(n_pulses, tau_d.size)] * 1000.0
    return pd.DataFrame({{
        "pulse": np.arange(1, n_pulses + 1),
        "amp_corrected": corrected,
        "amp_uncorrected": series(uncorrected_key),
        "ppr": corrected / corrected[0] if np.isfinite(corrected[0]) and corrected[0] else np.nan,
        "tau_decay_ms": taus,
    }})


def main() -> int:
    result = run()
    table = pulse_table(result)
    print(table.to_string(index=False, float_format=lambda v: f"{{v:.6g}}"))
    print(f"\\ntau_rise = {{1000.0 * float(result.get('tau_r_s', float('nan'))):.4g}} ms")

    out = Path(TRACE_FILE).with_name(Path(TRACE_FILE).stem + "_pulses.csv")
    table.to_csv(out, index=False)
    print(f"wrote {{out}}")
    return 0


# --------------------------------------------------------------------------
# Batch over a folder: drop the guard below and adapt the pattern.
# --------------------------------------------------------------------------
def run_folder(folder: str, pattern: str = "*.csv") -> pd.DataFrame:
    rows = []
    for path in sorted(Path(folder).glob(pattern)):
        try:
            table = pulse_table(run(str(path)))
        except Exception as exc:
            print(f"[skip] {{path.name}}: {{type(exc).__name__}}: {{exc}}")
            continue
        table.insert(0, "file", path.stem)
        rows.append(table)
        print(f"[ok]   {{path.name}}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _literal(value: Any) -> str:
    if isinstance(value, tuple):
        return repr(list(value))
    return repr(value)


def _format_options(values: Mapping[str, Any]) -> str:
    """One option per line, grouped, each annotated from the schema."""
    lines: list[str] = []
    for group in opt.GROUPS:
        keys = [s.key for s in opt.specs_in_group(group) if s.key in values]
        if not keys:
            continue
        if lines:
            lines.append("")
        lines.append(f"    # --- {group} ---")
        width = max(len(f'"{k}":') for k in keys)
        for key in keys:
            spec = opt.BY_KEY[key]
            entry = f'    "{key}":'.ljust(width + 4) + f" {_literal(values[key])},"
            help_text = (spec.help or "").split(". ")[0].rstrip(".")
            lines.append(f"{entry}  # {help_text}" if help_text else entry)

    extra = [k for k in values if k not in opt.BY_KEY]
    if extra:
        lines.append("")
        lines.append("    # --- Not exposed in the workbench ---")
        for key in sorted(extra):
            lines.append(f'    "{key}": {_literal(values[key])},')
    return "\n".join(lines)


def _format_bounds(bounds: Mapping[str, Tuple[float, float]]) -> str:
    if not bounds:
        return "{}"
    lines = ["{"]
    width = max(len(f'"{k}":') for k in bounds)
    for key in sorted(bounds):
        low, high = bounds[key]
        lines.append(f'    "{key}":'.ljust(width + 5) + f" ({low!r}, {high!r}),")
    lines.append("}")
    return "\n".join(lines)


def build_script(config, *, repo_root: Optional[Path] = None,
                 script_name: str = "run_extract_metrics.py") -> str:
    """Render a standalone script reproducing `config` (a core.TrainConfig)."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    values = dict(config.options)
    values.pop("plot", None)
    values.pop("event_model", None)
    values.pop("parameter_bounds", None)

    header = HEADER.format(
        stamp=datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z"),
        source=config.source,
        model=config.event_model,
        start=f"{config.train_start:g}",
        isi=f"{config.isi:g}",
        pulses=int(config.n_pulses),
        repo_root=str(root),
        script_name=script_name,
    )
    body = BODY.format(
        source=config.source,
        hint=config.time_column_hint or "",
        start=float(config.train_start),
        isi=float(config.isi),
        pulses=int(config.n_pulses),
        model=str(config.event_model),
        bounds=_format_bounds(config.parameter_bounds),
        options=_format_options(values),
    )
    return header + body


def write_script(config, path: str, **kwargs) -> Path:
    target = Path(path)
    target.write_text(build_script(config, script_name=target.name, **kwargs), encoding="utf-8")
    return target
