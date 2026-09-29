"""Read configurations written by other parts of the pipeline.

Two shapes are understood:

* ``run_settings.json`` - the reproducibility log written by
  ``Feature_extraction/demo_batch_process.py``, which stores fully resolved
  options per condition;
* the workbench's own ``*_config.json``, written by Save config / Export.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import options as opt


GUI_CONFIG = "gui_config"
RUN_SETTINGS = "run_settings"

# Bound keys used by the batch preset that do not match any model parameter.
# `tau_superslow` never matched anything in extract_metrics, which reads
# `tau_decay_superslow`; it is mapped here so the value is not silently lost.
BOUND_ALIASES: Dict[str, str] = {
    "tau_superslow": "tau_decay_superslow",
    "tau_decay_super_slow": "tau_decay_superslow",
}

# Recorded by the batch script but not part of the workbench option surface.
NON_OPTION_KEYS = frozenset({"plot", "event_model", "parameter_bounds", "event_model_settings"})


@dataclass
class LoadedConfig:
    """A configuration normalised into the fields the GUI drives."""
    kind: str
    label: str
    train_start: Optional[float] = None
    isi: Optional[float] = None
    n_pulses: Optional[int] = None
    event_model: Optional[str] = None
    source: Optional[str] = None
    time_column_hint: str = ""
    options: Dict[str, Any] = field(default_factory=dict)
    # Valid extract_metrics options with no widget in the workbench. They are
    # kept verbatim and passed straight back to the fitter, so loading and
    # re-running a batch configuration does not quietly change it.
    extra_options: Dict[str, Any] = field(default_factory=dict)
    parameter_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    ignored: List[str] = field(default_factory=list)
    renamed: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        bits = [self.label]
        if self.event_model:
            bits.append(f"model {self.event_model}")
        if self.isi:
            bits.append(f"ISI {self.isi * 1000:.0f} ms")
        if self.n_pulses:
            bits.append(f"{self.n_pulses} pulses")
        bits.append(f"{len(self.options)} options")
        if self.extra_options:
            bits.append(f"{len(self.extra_options)} passed through")
        if self.parameter_bounds:
            bits.append(f"{len(self.parameter_bounds)} bounds")
        if self.renamed:
            bits.append("renamed " + ", ".join(self.renamed))
        return " · ".join(bits)


def sniff(payload: Dict[str, Any]) -> str:
    """Identify which of the two shapes a decoded JSON document has."""
    if "resolved_options_by_condition" in payload:
        return RUN_SETTINGS
    if "options" in payload and any(k in payload for k in ("train_start", "isi", "source")):
        return GUI_CONFIG
    raise ValueError(
        "Unrecognised JSON. Expected a workbench config (Save config) or a "
        "run_settings.json written by demo_batch_process.py."
    )


def conditions(payload: Dict[str, Any]) -> List[str]:
    """Condition names available in a run_settings document."""
    return sorted((payload.get("resolved_options_by_condition") or {}).keys())


def _number(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _coerce_options(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """Split options into (widget-backed, passed-through verbatim, names of the latter)."""
    kept: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}
    for key, value in raw.items():
        if key in NON_OPTION_KEYS:
            continue
        spec = opt.BY_KEY.get(key)
        if spec is None:
            extra[key] = value
            continue
        if spec.kind == "bool":
            kept[key] = bool(value)
        elif spec.kind == "choice":
            kept[key] = str(value)
        elif value is None:
            kept[key] = None
        else:
            try:
                kept[key] = spec.coerce(value)
            except (TypeError, ValueError):
                # A value the widget cannot represent still reaches the fitter.
                extra[key] = value
    return kept, extra, sorted(extra)


def _coerce_bounds(raw: Any) -> Tuple[Dict[str, Tuple[float, float]], List[str]]:
    bounds: Dict[str, Tuple[float, float]] = {}
    renamed: List[str] = []
    if not isinstance(raw, dict):
        return bounds, renamed
    for key, pair in raw.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        low, high = _number(pair[0]), _number(pair[1])
        if low is None or high is None:
            continue
        target = BOUND_ALIASES.get(key, key)
        if target != key:
            renamed.append(f"{key}→{target}")
        bounds[target] = (min(low, high), max(low, high))
    return bounds, renamed


def _from_run_settings(payload: Dict[str, Any], condition: Optional[str]) -> LoadedConfig:
    resolved = payload.get("resolved_options_by_condition") or {}
    available = sorted(resolved)
    if not available:
        raise ValueError("run_settings.json contains no resolved options.")
    chosen = condition if condition in resolved else available[0]

    entry = resolved[chosen] or {}
    raw_options = dict(entry.get("options") or {})
    options, extra, ignored = _coerce_options(raw_options)
    bounds, renamed = _coerce_bounds(raw_options.get("parameter_bounds"))

    # The batch script resolves these outside the options dict.
    for key in ("peak_window_ms", "pre_zoom_s", "post_zoom_s"):
        value = _number(entry.get(key))
        if value is not None:
            options[key] = value

    notes = []
    if payload.get("preset_name"):
        notes.append(f"preset '{payload['preset_name']}'")
    if payload.get("export_datetime"):
        notes.append(str(payload["export_datetime"])[:19])
    git = payload.get("git") or {}
    if git.get("commit"):
        notes.append(f"commit {str(git['commit'])[:8]}" + (" (dirty)" if git.get("dirty") else ""))
    if len(available) > 1:
        notes.append(f"{len(available)} conditions in file")

    return LoadedConfig(
        kind=RUN_SETTINGS,
        label=f"run_settings · {chosen}",
        train_start=_number(entry.get("baseline_s")),
        isi=_number(entry.get("isi_s")),
        n_pulses=int(entry["n_pulses"]) if _number(entry.get("n_pulses")) else None,
        event_model=str(raw_options.get("event_model")) if raw_options.get("event_model") else None,
        options=options, extra_options=extra, parameter_bounds=bounds,
        ignored=ignored, renamed=renamed, notes=notes,
    )


def _from_gui_config(payload: Dict[str, Any]) -> LoadedConfig:
    options, extra, ignored = _coerce_options(dict(payload.get("options") or {}))
    bounds, renamed = _coerce_bounds(payload.get("parameter_bounds"))
    return LoadedConfig(
        kind=GUI_CONFIG,
        label="workbench config",
        train_start=_number(payload.get("train_start")),
        isi=_number(payload.get("isi")),
        n_pulses=int(payload["n_pulses"]) if _number(payload.get("n_pulses")) else None,
        event_model=str(payload["event_model"]) if payload.get("event_model") else None,
        source=str(payload["source"]) if payload.get("source") else None,
        time_column_hint=str(payload.get("time_column_hint") or ""),
        options=options, extra_options=extra, parameter_bounds=bounds,
        model_parameters=dict(payload.get("model_parameters") or {}),
        ignored=ignored, renamed=renamed,
    )


def load(path: str, *, condition: Optional[str] = None) -> LoadedConfig:
    """Read either supported JSON shape into a `LoadedConfig`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object at the top level.")
    kind = sniff(payload)
    return _from_run_settings(payload, condition) if kind == RUN_SETTINGS else _from_gui_config(payload)


def peek(path: str) -> Tuple[str, List[str]]:
    """Return (kind, condition names) without fully converting the document."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    kind = sniff(payload)
    return kind, (conditions(payload) if kind == RUN_SETTINGS else [])
