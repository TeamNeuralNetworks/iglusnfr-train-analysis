"""Reusable Tk widgets: tooltips, scrollable panels, collapsible sections and
the parameter row that couples a value, its bounds and a live slider.
"""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Optional

from .theme import FONT, MONO


class Tooltip:
    """Lightweight hover tooltip; wraps long option descriptions."""

    def __init__(self, widget, text: str, pal: Dict[str, str], delay_ms: int = 450) -> None:
        self.widget = widget
        self.text = text
        self.pal = pal
        self.delay_ms = delay_ms
        self._after = None
        self._win = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self._after = self.widget.after(self.delay_ms, self._show)

    def _cancel(self) -> None:
        if self._after is not None:
            try:
                self.widget.after_cancel(self._after)
            except Exception:
                pass
            self._after = None

    def _show(self) -> None:
        if self._win is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 14
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._win = tk.Toplevel(self.widget)
        self._win.wm_overrideredirect(True)
        self._win.wm_geometry(f"+{x}+{y}")
        frame = tk.Frame(self._win, background=self.pal["border"], padx=1, pady=1)
        frame.pack()
        tk.Label(
            frame, text=self.text, justify="left", wraplength=330,
            background=self.pal["panel"], foreground=self.pal["text"],
            font=(FONT, 8), padx=7, pady=5,
        ).pack()

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._win is not None:
            self._win.destroy()
            self._win = None


class ScrollFrame(ttk.Frame):
    """Vertically scrollable container; `body` is the frame to fill."""

    def __init__(self, parent, pal: Dict[str, str], **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self._canvas = tk.Canvas(self, background=pal["bg"], highlightthickness=0, borderwidth=0)
        self._bar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self.body = ttk.Frame(self._canvas)
        self._window = self._canvas.create_window((0, 0), window=self.body, anchor="nw")

        self.body.bind("<Configure>", lambda _e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>", lambda e: self._canvas.itemconfigure(self._window, width=e.width))
        self._canvas.configure(yscrollcommand=self._bar.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        self._bar.pack(side="right", fill="y")

        for widget in (self._canvas, self.body):
            widget.bind("<Enter>", self._bind_wheel, add="+")
            widget.bind("<Leave>", self._unbind_wheel, add="+")

    def _bind_wheel(self, _event=None) -> None:
        self._canvas.bind_all("<MouseWheel>", self._on_wheel)

    def _unbind_wheel(self, _event=None) -> None:
        self._canvas.unbind_all("<MouseWheel>")

    def _on_wheel(self, event) -> None:
        self._canvas.yview_scroll(int(-event.delta / 120), "units")


class Section(ttk.Frame):
    """Collapsible titled section."""

    def __init__(self, parent, title: str, pal: Dict[str, str], *, open_: bool = True,
                 hint: str = "") -> None:
        super().__init__(parent)
        self.pal = pal
        self._open = bool(open_)

        header = ttk.Frame(self)
        header.pack(fill="x")
        self._toggle = ttk.Button(header, text="", style="Link.TButton", width=2, command=self.toggle)
        self._toggle.pack(side="left")
        label = ttk.Label(header, text=title, style="Section.TLabel")
        label.pack(side="left", padx=(2, 0))
        if hint:
            Tooltip(label, hint, pal)
        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=(3, 5))

        self.body = ttk.Frame(self)
        self._sync()

    def toggle(self) -> None:
        self._open = not self._open
        self._sync()

    def _sync(self) -> None:
        self._toggle.configure(text="▾" if self._open else "▸")
        if self._open:
            self.body.pack(fill="x", padx=(14, 0))
        else:
            self.body.forget()


def _is_log_scale(lo: float, hi: float) -> bool:
    return lo > 0 and hi > 0 and (hi / lo) >= 50.0


class ParamRow(ttk.Frame):
    """One model parameter: value entry, live slider, and editable bounds.

    Emits `on_change` (debounced) whenever the value or a bound is edited, and
    shows a badge when the current value sits on one of its bounds.
    """

    PIN_TOL = 0.02  # within 2% of the span counts as pinned

    def __init__(self, parent, pal: Dict[str, str], *, name: str, label: str, unit: str,
                 value: float, lo: float, hi: float, tooltip: str = "",
                 on_change: Optional[Callable[[], None]] = None,
                 lockable: bool = True, debounce_ms: int = 140) -> None:
        super().__init__(parent)
        self.pal = pal
        self.name = name
        self.unit = unit
        self._on_change = on_change
        self._debounce_ms = debounce_ms
        self._after = None
        self._syncing = False

        self.value_var = tk.StringVar(value=self._fmt(value))
        self.lo_var = tk.StringVar(value=self._fmt(lo))
        self.hi_var = tk.StringVar(value=self._fmt(hi))
        # Remembered so the app can tell an edited bound from the registry's own.
        self.default_bounds = (float(lo), float(hi))
        self.locked = tk.BooleanVar(value=False)
        self._slider_var = tk.DoubleVar(value=0.0)

        self.columnconfigure(1, weight=1)

        title = ttk.Label(self, text=f"{label}  ({unit})" if unit else label)
        title.grid(row=0, column=0, columnspan=2, sticky="w")
        if tooltip:
            Tooltip(title, tooltip, pal)

        self.pin_label = ttk.Label(self, text="", style="Warn.TLabel")
        self.pin_label.grid(row=0, column=2, sticky="e", padx=(4, 0))

        self.value_entry = ttk.Entry(self, textvariable=self.value_var, width=10, font=(MONO, 9))
        self.value_entry.grid(row=0, column=3, sticky="e", padx=(4, 0))
        self.value_entry.bind("<KeyRelease>", lambda _e: self._on_value_typed())
        self.value_entry.bind("<Return>", lambda _e: self._emit(force=True))

        if lockable:
            lock = ttk.Checkbutton(self, variable=self.locked, command=self._emit_now, width=2)
            lock.grid(row=0, column=4, sticky="e")
            Tooltip(lock, "Hold this parameter fixed at its current value during the fit.", pal)

        self.lo_entry = ttk.Entry(self, textvariable=self.lo_var, width=8, font=(MONO, 8))
        self.lo_entry.grid(row=1, column=0, sticky="w", pady=(2, 0))
        self.lo_entry.bind("<KeyRelease>", lambda _e: self._on_bounds_typed())

        self.slider = ttk.Scale(self, from_=0.0, to=1.0, orient="horizontal",
                                variable=self._slider_var, command=self._on_slider)
        self.slider.grid(row=1, column=1, columnspan=3, sticky="ew", padx=6, pady=(2, 0))

        self.hi_entry = ttk.Entry(self, textvariable=self.hi_var, width=8, font=(MONO, 8))
        self.hi_entry.grid(row=1, column=4, sticky="e", pady=(2, 0))
        self.hi_entry.bind("<KeyRelease>", lambda _e: self._on_bounds_typed())

        Tooltip(self.lo_entry, "Lower bound used when fitting this parameter.", pal)
        Tooltip(self.hi_entry, "Upper bound used when fitting this parameter.", pal)

        self._sync_slider()

    # -- formatting / parsing -------------------------------------------------
    @staticmethod
    def _fmt(v: float) -> str:
        if v is None or not math.isfinite(float(v)):
            return ""
        v = float(v)
        if v == 0:
            return "0"
        if 1e-3 <= abs(v) < 1e5:
            return f"{v:.4g}"
        return f"{v:.3e}"

    @staticmethod
    def _parse(text: str) -> Optional[float]:
        try:
            v = float(str(text).strip())
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None

    # -- public accessors -----------------------------------------------------
    @property
    def value(self) -> Optional[float]:
        return self._parse(self.value_var.get())

    @property
    def bounds(self):
        return self._parse(self.lo_var.get()), self._parse(self.hi_var.get())

    def set_value(self, v: float, *, silent: bool = True) -> None:
        self._syncing = True
        self.value_var.set(self._fmt(v))
        self._sync_slider()
        self._mark_pinned()
        self._syncing = False
        if not silent:
            self._emit_now()

    def set_bounds(self, lo: float, hi: float) -> None:
        self._syncing = True
        self.lo_var.set(self._fmt(lo))
        self.hi_var.set(self._fmt(hi))
        self._sync_slider()
        self._mark_pinned()
        self._syncing = False

    def valid(self) -> bool:
        v, (lo, hi) = self.value, self.bounds
        return v is not None and lo is not None and hi is not None and lo < hi

    def bounds_edited(self, rel_tol: float = 1e-9) -> bool:
        """True when either bound has been moved off the registry default."""
        lo, hi = self.bounds
        if lo is None or hi is None:
            return False
        return not (math.isclose(lo, self.default_bounds[0], rel_tol=rel_tol)
                    and math.isclose(hi, self.default_bounds[1], rel_tol=rel_tol))

    # -- internals ------------------------------------------------------------
    def _sync_slider(self) -> None:
        v, (lo, hi) = self.value, self.bounds
        if v is None or lo is None or hi is None or hi <= lo:
            return
        v = min(max(v, lo), hi)
        if _is_log_scale(lo, hi):
            frac = (math.log(max(v, lo)) - math.log(lo)) / (math.log(hi) - math.log(lo))
        else:
            frac = (v - lo) / (hi - lo)
        self._slider_var.set(min(max(frac, 0.0), 1.0))

    def _slider_to_value(self, frac: float) -> Optional[float]:
        lo, hi = self.bounds
        if lo is None or hi is None or hi <= lo:
            return None
        if _is_log_scale(lo, hi):
            return math.exp(math.log(lo) + frac * (math.log(hi) - math.log(lo)))
        return lo + frac * (hi - lo)

    def _on_slider(self, _value) -> None:
        if self._syncing:
            return
        v = self._slider_to_value(float(self._slider_var.get()))
        if v is None:
            return
        self._syncing = True
        self.value_var.set(self._fmt(v))
        self._syncing = False
        self._mark_pinned()
        self._emit()

    def _on_value_typed(self) -> None:
        if self._syncing:
            return
        ok = self.value is not None
        self.value_entry.configure(style="TEntry" if ok else "Bad.TEntry")
        if ok:
            self._syncing = True
            self._sync_slider()
            self._syncing = False
            self._mark_pinned()
            self._emit()

    def _on_bounds_typed(self) -> None:
        if self._syncing:
            return
        lo, hi = self.bounds
        ok = lo is not None and hi is not None and lo < hi
        style = "TEntry" if ok else "Bad.TEntry"
        self.lo_entry.configure(style=style)
        self.hi_entry.configure(style=style)
        if ok:
            self._syncing = True
            self._sync_slider()
            self._syncing = False
            self._mark_pinned()
            self._emit()

    def _mark_pinned(self) -> None:
        v, (lo, hi) = self.value, self.bounds
        if v is None or lo is None or hi is None or hi <= lo:
            self.pin_label.configure(text="")
            return
        span = hi - lo
        if v <= lo + self.PIN_TOL * span:
            self.pin_label.configure(text="at min", style="Warn.TLabel")
        elif v >= hi - self.PIN_TOL * span:
            self.pin_label.configure(text="at max", style="Warn.TLabel")
        else:
            self.pin_label.configure(text="")

    def is_pinned(self) -> bool:
        return bool(self.pin_label.cget("text"))

    def _emit(self, force: bool = False) -> None:
        if self._on_change is None:
            return
        if self._after is not None:
            try:
                self.after_cancel(self._after)
            except Exception:
                pass
        delay = 1 if force else self._debounce_ms
        self._after = self.after(delay, self._emit_now)

    def _emit_now(self) -> None:
        self._after = None
        if self._on_change is not None:
            self._on_change()


class OptionRow(ttk.Frame):
    """Generic labelled option control driven by an `options.OptionSpec`."""

    def __init__(self, parent, pal: Dict[str, str], spec, variable, *,
                 on_change: Optional[Callable[[], None]] = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self.variable = variable
        self.columnconfigure(0, weight=1)

        if spec.kind == "bool":
            widget = ttk.Checkbutton(self, text=spec.label, variable=variable, command=on_change)
            widget.grid(row=0, column=0, columnspan=2, sticky="w")
            target = widget
        else:
            label = ttk.Label(self, text=spec.label)
            label.grid(row=0, column=0, sticky="w")
            if spec.kind == "choice":
                widget = ttk.Combobox(self, textvariable=variable, values=list(spec.choices),
                                      state="readonly", width=15, font=(FONT, 9))
                widget.bind("<<ComboboxSelected>>", lambda _e: on_change and on_change())
            else:
                widget = ttk.Entry(self, textvariable=variable, width=12, font=(MONO, 9))
                widget.bind("<FocusOut>", lambda _e: on_change and on_change())
                widget.bind("<Return>", lambda _e: on_change and on_change())
            widget.grid(row=0, column=1, sticky="e", padx=(6, 0))
            target = label

        self.widget = widget
        if spec.help:
            Tooltip(target, spec.help, pal)
            if target is not widget:
                Tooltip(widget, spec.help, pal)


def labelled_value(parent, label: str, var: tk.StringVar, pal: Dict[str, str],
                   *, help_text: str = "", width: int = 10) -> ttk.Frame:
    """Compact `label [entry]` pair used in the data/train panel."""
    row = ttk.Frame(parent)
    row.columnconfigure(0, weight=1)
    text = ttk.Label(row, text=label)
    text.grid(row=0, column=0, sticky="w")
    entry = ttk.Entry(row, textvariable=var, width=width, font=(MONO, 9))
    entry.grid(row=0, column=1, sticky="e", padx=(6, 0))
    if help_text:
        Tooltip(text, help_text, pal)
    row.entry = entry
    return row
