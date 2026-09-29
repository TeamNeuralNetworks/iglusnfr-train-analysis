"""Interactive event-fitting workbench for extract_metrics.

Run from the repository root::

    python -m extract_metrics_gui.app
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python extract_metrics_gui/app.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "extract_metrics_gui"

import json
import queue
import threading
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from . import core, loaders, options as opt, registry, runsettings, scriptgen
from .theme import FONT, MONO, apply_ttk_style, palette, style_axes
from .widgets import OptionRow, ParamRow, ScrollFrame, Section, Tooltip, labelled_value


APP_TITLE = "extract_metrics — event fitting workbench"


class EventPanel:
    """Average-event figure updated in place, so sliders stay responsive."""

    def __init__(self, parent, pal: Dict[str, str]) -> None:
        self.pal = pal
        self.figure = Figure(figsize=(7.4, 5.2), dpi=100)
        self.figure.patch.set_facecolor(pal["bg"])
        grid = self.figure.add_gridspec(2, 1, height_ratios=[3.1, 1.0], hspace=0.09)
        self.ax = self.figure.add_subplot(grid[0])
        self.ax_res = self.figure.add_subplot(grid[1], sharex=self.ax)
        style_axes(self.ax, pal)
        style_axes(self.ax_res, pal)
        self.ax.set_ylabel("Event amplitude")
        self.ax.tick_params(labelbottom=False)
        self.ax_res.set_ylabel("Residual")
        self.ax_res.set_xlabel("Time from stimulus (ms)")

        self.snippets = LineCollection([], colors=pal["snippet"], linewidths=0.5, alpha=0.28, zorder=1)
        self.ax.add_collection(self.snippets)
        (self.avg_line,) = self.ax.plot([], [], color=pal["trace"], lw=2.1, label="average", zorder=3)
        (self.model_line,) = self.ax.plot([], [], color=pal["model"], lw=2.0, ls="--",
                                          label="model", zorder=4)
        self.ax.axvline(0.0, color=pal["text_dim"], ls=":", lw=0.9, alpha=0.7)
        (self.res_line,) = self.ax_res.plot([], [], color=pal["residual"], lw=1.0)
        self.ax_res.axhline(0.0, color=pal["text_dim"], ls=":", lw=0.9, alpha=0.7)
        self.legend = self.ax.legend(frameon=False, loc="upper left", fontsize=8,
                                     labelcolor=pal["text_dim"])

        self.stats = self.ax.text(
            0.985, 0.96, "", transform=self.ax.transAxes, ha="right", va="top",
            fontsize=8, color=pal["text_dim"], family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=pal["panel"],
                      edgecolor=pal["border"], alpha=0.9),
        )
        self.warning = self.ax.text(
            0.985, 0.03, "", transform=self.ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=pal["warn"],
        )
        self.placeholder = self.ax.text(
            0.5, 0.5, "Open a trace file to begin.", transform=self.ax.transAxes,
            ha="center", va="center", fontsize=10, color=pal["text_dim"],
        )
        self.figure.subplots_adjust(left=0.115, right=0.98, top=0.97, bottom=0.115)

        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, parent, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")
        self._style_toolbar()
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _style_toolbar(self) -> None:
        try:
            self.toolbar.configure(background=self.pal["bg"])
            for child in self.toolbar.winfo_children():
                try:
                    child.configure(background=self.pal["bg"])
                except tk.TclError:
                    pass
        except tk.TclError:
            pass

    def _on_scroll(self, event) -> None:
        ax = event.inaxes
        if ax is None or event.xdata is None:
            return
        scale = 0.82 if event.button == "up" else 1.22
        x0, x1 = ax.get_xlim()
        ax.set_xlim(event.xdata - (event.xdata - x0) * scale,
                    event.xdata + (x1 - event.xdata) * scale)
        if event.key in ("control", "ctrl") and event.ydata is not None:
            y0, y1 = ax.get_ylim()
            ax.set_ylim(event.ydata - (event.ydata - y0) * scale,
                        event.ydata + (y1 - event.ydata) * scale)
        self.canvas.draw_idle()

    def update(self, event: Optional[core.AverageEvent], fit: Optional[core.EventFit],
               *, show_snippets: bool = True, keep_view: bool = False) -> None:
        xlim = self.ax.get_xlim() if keep_view else None

        if event is None or not event.ok:
            self.placeholder.set_visible(True)
            self.snippets.set_segments([])
            for line in (self.avg_line, self.model_line, self.res_line):
                line.set_data([], [])
            self.stats.set_text("")
            self.warning.set_text("")
            self.canvas.draw_idle()
            return

        self.placeholder.set_visible(False)
        t = np.asarray(event.t_ms, float)

        if show_snippets and event.snippets.size:
            step = max(1, event.snippets.shape[0] // 120)
            self.snippets.set_segments([
                np.column_stack([t, snippet]) for snippet in event.snippets[::step]
            ])
        else:
            self.snippets.set_segments([])

        self.avg_line.set_data(t, event.average)
        self.avg_line.set_label(f"{event.projection} of {event.n_used}")

        if fit is not None:
            self.model_line.set_data(fit.t_ms, fit.yhat)
            self.model_line.set_label(f"{fit.model}")
            self.res_line.set_data(fit.t_ms, fit.residual)
            lines = [f"R2   {fit.r2:.4f}", f"RMS  {fit.rms:.4g}"]
            if np.isfinite(fit.aic):
                lines.append(f"AIC  {fit.aic:.1f}")
            self.stats.set_text("\n".join(lines))
            notes = []
            if fit.pinned:
                notes.append("at bounds: " + ", ".join(fit.pinned))
            if not fit.converged:
                notes.append("fit did not converge")
            self.warning.set_text("   ".join(notes))
            self.warning.set_color(self.pal["bad"] if not fit.converged else self.pal["warn"])
        else:
            self.model_line.set_data([], [])
            self.res_line.set_data([], [])
            self.stats.set_text("")
            self.warning.set_text("")

        self.legend = self.ax.legend(frameon=False, loc="upper left", fontsize=8,
                                     labelcolor=self.pal["text_dim"])
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax_res.relim()
        self.ax_res.autoscale_view()
        self.ax.set_xlim(xlim if xlim else (float(t.min()), float(t.max())))
        self.canvas.draw_idle()


class FigurePanel:
    """Static figure holder for the panels that only change once per train fit."""

    def __init__(self, parent, pal: Dict[str, str], builder) -> None:
        self.parent = parent
        self.pal = pal
        self.builder = builder
        self.canvas = None
        self.toolbar = None
        self.render(None)

    def render(self, payload) -> None:
        if self.toolbar is not None:
            self.toolbar.destroy()
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        figure = self.builder(payload, self.pal)
        self.canvas = FigureCanvasTkAgg(figure, master=self.parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.parent, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")
        try:
            self.toolbar.configure(background=self.pal["bg"])
            for child in self.toolbar.winfo_children():
                try:
                    child.configure(background=self.pal["bg"])
                except tk.TclError:
                    pass
        except tk.TclError:
            pass
        self.canvas.draw_idle()


class Workbench(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        width = min(1600, max(1180, int(self.winfo_screenwidth() * 0.9)))
        height = min(1020, max(680, int(self.winfo_screenheight() * 0.88)))
        self.geometry(f"{width}x{height}")
        self.minsize(1100, 660)

        self.theme_name = tk.StringVar(value="light")
        self.pal = palette(self.theme_name.get())
        self.style = ttk.Style(self)
        apply_ttk_style(self.style, self.pal)
        self.configure(bg=self.pal["bg"])

        self.data: Optional[loaders.TraceData] = None
        self.event: Optional[core.AverageEvent] = None
        self.event_fit: Optional[core.EventFit] = None
        self.result: Optional[core.TrainResult] = None
        self.param_rows: Dict[str, ParamRow] = {}
        self.option_vars: Dict[str, tk.Variable] = {}
        self._unmatched_bounds: List[str] = []
        # Options from a loaded run that the workbench has no widget for. They
        # are forwarded to extract_metrics untouched so loading a batch
        # configuration and re-running it reproduces that run.
        self.passthrough_options: Dict[str, Any] = {}
        self._worker: Optional[threading.Thread] = None
        self._queue: "queue.Queue[tuple]" = queue.Queue()
        self._event_job = None

        self._init_vars()
        self._build_ui()
        self._rebuild_param_rows()
        self.after(120, self._poll_worker)

    # -- state ---------------------------------------------------------------
    def _init_vars(self) -> None:
        self.file_path = tk.StringVar()
        self.time_hint = tk.StringVar(value="")
        self.train_start = tk.StringVar(value="0.5")
        self.isi = tk.StringVar(value="0.05")
        self.n_pulses = tk.StringVar(value="10")
        self.model_name = tk.StringVar(value=registry.label_for("double_exp"))
        self.preset_name = tk.StringVar(value=opt.preset_names()[0])
        self.show_snippets = tk.BooleanVar(value=True)
        self.show_advanced = tk.BooleanVar(value=False)
        self.push_bounds = tk.BooleanVar(value=True)
        self.auto_fit = tk.BooleanVar(value=True)
        self.dataset_info = tk.StringVar(value="No file loaded.")
        self.train_info = tk.StringVar(value="")
        self.status = tk.StringVar(value="Open a CSV, Excel or .npz trace file to begin.")
        self.fit_quality = tk.StringVar(value="")

        for spec in opt.SPECS:
            if spec.kind == "bool":
                var: tk.Variable = tk.BooleanVar(value=bool(spec.default))
            else:
                var = tk.StringVar(value=spec.to_text(spec.default))
            self.option_vars[spec.key] = var

    # -- layout --------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=(12, 10, 12, 8))
        outer.pack(fill="both", expand=True)

        self._build_header(outer)

        body = ttk.PanedWindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True, pady=(8, 0))

        left = ttk.Frame(body, width=360)
        right = ttk.Frame(body)
        body.add(left, weight=0)
        body.add(right, weight=1)

        self._build_controls(left)
        self._build_panels(right)
        self._build_statusbar(outer)

    def _build_header(self, parent) -> None:
        header = ttk.Frame(parent)
        header.pack(fill="x")

        title = ttk.Label(header, text="Event fitting workbench", style="Head.TLabel")
        title.pack(side="left")

        ttk.Button(header, text="Theme", style="Link.TButton",
                   command=self._toggle_theme).pack(side="right")
        script_button = ttk.Button(header, text="Generate script", style="Link.TButton",
                                   command=self._generate_script)
        script_button.pack(side="right", padx=(0, 4))
        Tooltip(script_button, "Write a standalone Python script that reproduces the current "
                               "settings, with every option written out and annotated. It also "
                               "carries a run_folder() helper for batch processing.", self.pal)
        load_button = ttk.Button(header, text="Load config", style="Link.TButton",
                                 command=self._load_config)
        load_button.pack(side="right", padx=(0, 4))
        Tooltip(load_button, "Load a workbench config, or a run_settings.json written by "
                             "demo_batch_process.py.", self.pal)
        ttk.Button(header, text="Save config", style="Link.TButton",
                   command=self._save_config).pack(side="right", padx=(0, 4))

        preset_box = ttk.Frame(header)
        preset_box.pack(side="right", padx=(0, 14))
        ttk.Label(preset_box, text="Preset", style="Dim.TLabel").pack(side="left", padx=(0, 4))
        combo = ttk.Combobox(preset_box, textvariable=self.preset_name, state="readonly",
                             values=opt.preset_names(), width=22, font=(FONT, 9))
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_preset())
        Tooltip(combo, "Load a documented set of options. 'iGluSnFR optimised' mirrors the preset "
                       "used by Feature_extraction/demo_batch_process.py.", self.pal)

        file_row = ttk.Frame(parent)
        file_row.pack(fill="x", pady=(9, 0))
        entry = ttk.Entry(file_row, textvariable=self.file_path, font=(MONO, 9))
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda _e: self._load_file(self.file_path.get()))
        ttk.Button(file_row, text="Open…", command=self._browse).pack(side="left", padx=(6, 0))

    def _build_controls(self, parent) -> None:
        scroller = ScrollFrame(parent, self.pal)
        scroller.pack(fill="both", expand=True)
        body = scroller.body

        # --- data & train ---------------------------------------------------
        data_section = Section(body, "Data & stimulus train", self.pal,
                              hint="Where the traces come from and how the pulse train is laid out.")
        data_section.pack(fill="x", pady=(0, 10))
        box = data_section.body

        labelled_value(box, "Time column hint", self.time_hint, self.pal,
                       help_text="Name of the time column. Blank auto-detects 'tim', 'time', "
                                 "'time_s' or the first increasing numeric column.").pack(fill="x", pady=1)
        ttk.Label(box, textvariable=self.dataset_info, style="Dim.TLabel",
                  wraplength=310, justify="left").pack(anchor="w", pady=(4, 6))

        for label, var, help_text in (
            ("Train start (s)", self.train_start, "Time of the first stimulus."),
            ("ISI (s)", self.isi, "Interval between successive stimuli."),
            ("Pulses", self.n_pulses, "Number of stimuli in the train."),
        ):
            row = labelled_value(box, label, var, self.pal, help_text=help_text)
            row.pack(fill="x", pady=1)
            row.entry.bind("<KeyRelease>", lambda _e: self._schedule_event_refresh())

        detect = ttk.Button(box, text="Detect train from data", command=self._detect_train)
        detect.pack(fill="x", pady=(7, 2))
        Tooltip(detect, "Estimate start, ISI and pulse count from upward excursions of the "
                        "averaged trace, then refine them on a regular grid.", self.pal)
        ttk.Label(box, textvariable=self.train_info, style="Dim.TLabel",
                  wraplength=310, justify="left").pack(anchor="w")

        # --- event model ----------------------------------------------------
        model_section = Section(body, "Event model", self.pal,
                               hint="Every model registered in Model_Calibration/event_models.py.")
        model_section.pack(fill="x", pady=(0, 10))
        box = model_section.body

        combo = ttk.Combobox(box, textvariable=self.model_name, state="readonly",
                             values=[registry.label_for(n) for n in registry.model_names()],
                             font=(FONT, 9))
        combo.pack(fill="x")
        combo.bind("<<ComboboxSelected>>", lambda _e: self._on_model_changed())
        self.model_combo = combo

        self.model_summary = ttk.Label(box, text="", style="Dim.TLabel",
                                       wraplength=310, justify="left")
        self.model_summary.pack(anchor="w", pady=(4, 6))

        actions = ttk.Frame(box)
        actions.pack(fill="x", pady=(0, 6))
        ttk.Button(actions, text="Fit event", style="Accent.TButton",
                   command=self._fit_event_now).pack(side="left")
        ttk.Button(actions, text="Reset", command=self._reset_params).pack(side="left", padx=(6, 0))
        ttk.Label(actions, textvariable=self.fit_quality, style="Dim.TLabel").pack(side="right")

        self.params_host = ttk.Frame(box)
        self.params_host.pack(fill="x")

        push = ttk.Checkbutton(box, text="Send edited bounds to the train fit",
                               variable=self.push_bounds)
        push.pack(anchor="w", pady=(8, 0))
        Tooltip(push, "Passes the bounds you have changed as extract_metrics 'parameter_bounds'. "
                      "This is the control that actually constrains the train kinetics. Rows left "
                      "at their registry defaults are not sent, since the fit already applies "
                      "those limits.", self.pal)
        ttk.Checkbutton(box, text="Refit automatically on change", variable=self.auto_fit
                        ).pack(anchor="w")
        ttk.Checkbutton(box, text="Show individual snippets", variable=self.show_snippets,
                        command=lambda: self._refresh_event(refit=False)).pack(anchor="w")

        # --- options --------------------------------------------------------
        options_section = Section(body, "Fitting options", self.pal,
                                 hint="The extract_metrics option surface, grouped by what it affects.")
        options_section.pack(fill="x", pady=(0, 10))
        box = options_section.body

        ttk.Checkbutton(box, text="Show advanced options", variable=self.show_advanced,
                        command=self._rebuild_option_rows).pack(anchor="w", pady=(0, 6))
        self.options_host = ttk.Frame(box)
        self.options_host.pack(fill="x")
        self._rebuild_option_rows()

    def _build_panels(self, parent) -> None:
        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True)

        event_tab = ttk.Frame(notebook)
        train_tab = ttk.Frame(notebook)
        amp_tab = ttk.Frame(notebook)
        table_tab = ttk.Frame(notebook)
        notebook.add(event_tab, text="Event fit")
        notebook.add(train_tab, text="Train fit")
        notebook.add(amp_tab, text="Amplitudes")
        notebook.add(table_tab, text="Results")

        self.event_panel = EventPanel(event_tab, self.pal)
        self.train_panel = FigurePanel(train_tab, self.pal, core.build_train_figure)
        self.amp_panel = FigurePanel(amp_tab, self.pal, core.build_amplitude_figure)
        self._build_tables(table_tab)

    def _build_tables(self, parent) -> None:
        paned = ttk.PanedWindow(parent, orient="vertical")
        paned.pack(fill="both", expand=True, padx=2, pady=2)

        top = ttk.Frame(paned)
        bottom = ttk.Frame(paned)
        paned.add(top, weight=3)
        paned.add(bottom, weight=2)

        columns = ("pulse", "amp_corrected", "amp_uncorrected", "ppr", "tau_decay_ms")
        headings = ("Pulse", "Amp (corrected)", "Amp (raw)", "Ratio to P1", "Tau decay (ms)")
        self.pulse_table = ttk.Treeview(top, columns=columns, show="headings")
        for column, heading in zip(columns, headings):
            self.pulse_table.heading(column, text=heading)
            self.pulse_table.column(column, width=90 if column == "pulse" else 130,
                                    anchor="center" if column == "pulse" else "e")
        self.pulse_table.pack(fill="both", expand=True)

        self.metric_table = ttk.Treeview(bottom, columns=("metric", "value"), show="headings")
        self.metric_table.heading("metric", text="Metric")
        self.metric_table.heading("value", text="Value")
        self.metric_table.column("metric", width=220, anchor="w")
        self.metric_table.column("value", width=200, anchor="e")
        self.metric_table.pack(fill="both", expand=True)

    def _build_statusbar(self, parent) -> None:
        bar = ttk.Frame(parent)
        bar.pack(fill="x", pady=(8, 0))

        ttk.Label(bar, textvariable=self.status, style="Dim.TLabel").pack(side="left")

        self.export_button = ttk.Button(bar, text="Export results", command=self._export,
                                        state="disabled")
        self.export_button.pack(side="right")
        self.run_button = ttk.Button(bar, text="Run train fit", style="Accent.TButton",
                                     command=self._run_train, state="disabled")
        self.run_button.pack(side="right", padx=(0, 6))
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=120)
        self.progress.pack(side="right", padx=(0, 10))

    # -- model / parameter panel ---------------------------------------------
    def _current_model(self) -> str:
        return registry.name_from_label(self.model_name.get()) or "double_exp"

    def _on_model_changed(self) -> None:
        self._rebuild_param_rows()
        self._refresh_event(refit=self.auto_fit.get())

    def _rebuild_param_rows(self) -> None:
        for child in self.params_host.winfo_children():
            child.destroy()
        self.param_rows.clear()

        name = self._current_model()
        try:
            info = registry.model(name)
        except Exception as exc:
            ttk.Label(self.params_host, text=f"Unknown model: {exc}", style="Bad.TLabel").pack(anchor="w")
            return

        self.model_summary.configure(
            text=f"{info.summary}\ncomplexity {info.complexity} · "
                 f"{'tau-varying kernel' if info.tau_varying else 'fixed-shape template'}"
        )

        seed = self.event_fit.params if (self.event_fit and self.event_fit.model == info.name) else {}
        if not seed and self.event is not None and self.event.ok:
            finite = np.isfinite(self.event.average)
            guess = info.initial_guess(self.event.average[finite], self.event.t_ms[finite])
            seed = dict(zip(info.params, guess))
        if not seed:
            seed = dict(zip(info.params, info.display_defaults()))

        for param in info.params:
            value = float(seed.get(param, 0.0))
            lo, hi = info.slider_bounds(param, value)
            row = ParamRow(
                self.params_host, self.pal, name=param, label=param,
                unit=registry.param_unit(param), value=value, lo=lo, hi=hi,
                tooltip=registry.param_help(param), on_change=self._schedule_event_refresh,
            )
            row.pack(fill="x", pady=(4, 0))
            self.param_rows[param] = row

    def _reset_params(self) -> None:
        self.event_fit = None
        self._rebuild_param_rows()
        self._refresh_event(refit=True)

    def _param_state(self, *, edited_bounds_only: bool = False):
        start, bounds, locked = {}, {}, []
        for name, row in self.param_rows.items():
            if row.value is not None:
                start[name] = float(row.value)
            lo, hi = row.bounds
            if lo is not None and hi is not None and lo < hi:
                if not edited_bounds_only or row.bounds_edited():
                    bounds[name] = (float(lo), float(hi))
            if row.locked.get():
                locked.append(name)
        return start, bounds, tuple(locked)

    # -- inner loop -----------------------------------------------------------
    def _schedule_event_refresh(self) -> None:
        if self._event_job is not None:
            try:
                self.after_cancel(self._event_job)
            except Exception:
                pass
        self._event_job = self.after(130, lambda: self._refresh_event(refit=False))

    def _train_geometry(self):
        try:
            return (float(self.train_start.get()), float(self.isi.get()),
                    int(float(self.n_pulses.get())))
        except (TypeError, ValueError):
            return None

    def _rebuild_event(self) -> None:
        geometry = self._train_geometry()
        if self.data is None or geometry is None:
            self.event = None
            return
        start, isi, pulses = geometry
        if isi <= 0 or pulses < 1:
            self.event = None
            return
        try:
            oversample = int(float(self.option_vars["recut_oversample"].get() or 8))
        except (TypeError, ValueError):
            oversample = 8
        projection = str(self.option_vars["recut_projection"].get() or "median")
        self.event = core.build_average_event(
            self.data, train_start=start, isi=isi, n_pulses=pulses,
            oversample=max(1, min(oversample, 32)), projection=projection,
        )

    def _refresh_event(self, *, refit: bool) -> None:
        self._event_job = None
        self._rebuild_event()

        if self.event is None or not self.event.ok:
            self.event_fit = None
            self.fit_quality.set("")
            self.event_panel.update(None, None)
            return

        start, bounds, locked = self._param_state()
        try:
            self.event_fit = core.fit_event(
                self.event, self._current_model(), start=start, bounds=bounds,
                locked=locked, refit=refit,
            )
        except Exception as exc:
            self.event_fit = None
            self.fit_quality.set("fit failed")
            self.status.set(f"Event fit failed: {exc}")
            self.event_panel.update(self.event, None, show_snippets=self.show_snippets.get())
            return

        if refit:
            for name, value in self.event_fit.params.items():
                row = self.param_rows.get(name)
                if row is not None and not row.locked.get():
                    row.set_value(value)

        fit = self.event_fit
        quality = f"R2 {fit.r2:.4f}"
        if fit.pinned:
            quality += f"  ·  {len(fit.pinned)} at bound"
        self.fit_quality.set(quality)
        self.event_panel.update(self.event, fit, show_snippets=self.show_snippets.get(),
                                keep_view=not refit)

        if fit.pinned:
            self.status.set(
                "Fitted. " + ", ".join(fit.pinned) + " sat on a bound — widen it before trusting the fit."
            )
        elif refit:
            self.status.set(f"Event fit converged (R2 = {fit.r2:.4f}).")

    def _fit_event_now(self) -> None:
        self._refresh_event(refit=True)

    # -- file handling --------------------------------------------------------
    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Open trace file",
            filetypes=[
                ("All supported", "*.csv *.txt *.tsv *.xlsx *.xlsm *.xls *.npz"),
                ("CSV / text", "*.csv *.txt *.tsv"),
                ("Excel", "*.xlsx *.xlsm *.xls"),
                ("NumPy bundle", "*.npz"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str) -> None:
        if not str(path).strip():
            return
        try:
            self.data = loaders.load_traces(str(path).strip(), self.time_hint.get().strip())
        except Exception as exc:
            self.data = None
            self.dataset_info.set("No file loaded.")
            self.status.set("Could not read the file.")
            messagebox.showerror(APP_TITLE, str(exc))
            return

        self.file_path.set(str(self.data.source))
        self.dataset_info.set(loaders.describe(self.data))
        self.result = None
        self.export_button.configure(state="disabled")
        self.run_button.configure(state="normal")
        self.train_panel.render(None)
        self.amp_panel.render(None)
        self._clear_tables()

        if self.data.train_start is not None and self.data.isi is not None:
            self.train_start.set(f"{self.data.train_start:g}")
            self.isi.set(f"{self.data.isi:g}")
            if self.data.n_pulses:
                self.n_pulses.set(str(int(self.data.n_pulses)))
            self.train_info.set("Train geometry read from the file.")
        else:
            self._detect_train(quiet=True)

        self.status.set(f"Loaded {Path(self.data.source).name}.")
        self._refresh_event(refit=True)

    def _detect_train(self, *, quiet: bool = False) -> None:
        if self.data is None:
            if not quiet:
                messagebox.showinfo(APP_TITLE, "Open a trace file first.")
            return
        guess = loaders.detect_train(self.data.time_s, self.data.trials)
        if guess is None:
            self.train_info.set("Could not detect a train; enter the values manually.")
            return
        self.train_start.set(f"{guess.train_start:.6g}")
        self.isi.set(f"{guess.isi:.6g}")
        self.n_pulses.set(str(guess.n_pulses))
        confidence = "high" if guess.confidence > 0.8 else "medium" if guess.confidence > 0.5 else "low"
        self.train_info.set(f"{guess.message} — {confidence} confidence")
        if not quiet:
            self._refresh_event(refit=True)

    # -- options --------------------------------------------------------------
    def _rebuild_option_rows(self) -> None:
        for child in self.options_host.winfo_children():
            child.destroy()
        advanced = self.show_advanced.get()
        for group in opt.GROUPS:
            specs = [s for s in opt.specs_in_group(group) if advanced or not s.advanced]
            if not specs:
                continue
            section = Section(self.options_host, group, self.pal, open_=group in ("Kinetics", "Preprocessing"))
            section.pack(fill="x", pady=(0, 6))
            for spec in specs:
                row = OptionRow(section.body, self.pal, spec, self.option_vars[spec.key],
                                on_change=self._on_option_changed)
                row.pack(fill="x", pady=1)

    def _on_option_changed(self) -> None:
        # Only the recut options change the inner-loop event; the rest apply at run time.
        self._schedule_event_refresh()

    def _apply_preset(self) -> None:
        preset = opt.PRESETS.get(self.preset_name.get(), {})
        # Reset to the schema defaults first: a preset must describe a complete,
        # reproducible state rather than layering onto whatever came before —
        # including any options inherited from a previously loaded run.
        self.passthrough_options = {}
        self._unmatched_bounds = []
        self.train_info.set("")
        for spec in opt.SPECS:
            if spec.kind == "bool":
                self.option_vars[spec.key].set(bool(spec.default))
            else:
                self.option_vars[spec.key].set(spec.to_text(spec.default))
        for key, value in preset.items():
            if key == "event_model":
                try:
                    self.model_name.set(registry.label_for(str(value)))
                except Exception:
                    continue
                self._rebuild_param_rows()
            elif key == "parameter_bounds":
                # Presets are written with canonical keys (tau_decay_fast, ...);
                # map them onto whatever the selected model calls its parameters.
                for canonical, (lo, hi) in value.items():
                    for param, row in self.param_rows.items():
                        if param == canonical or registry.canonical_bound_key(param) == canonical:
                            row.set_bounds(float(lo), float(hi))
            elif key in self.option_vars:
                spec = opt.BY_KEY[key]
                if spec.kind == "bool":
                    self.option_vars[key].set(bool(value))
                else:
                    self.option_vars[key].set(spec.to_text(value))
        self._rebuild_option_rows()
        self.status.set(f"Applied preset '{self.preset_name.get()}'.")
        self._refresh_event(refit=True)

    def _collect_options(self) -> Dict[str, Any]:
        values: Dict[str, Any] = {}
        problems: List[str] = []
        for spec in opt.SPECS:
            raw = self.option_vars[spec.key].get()
            try:
                coerced = spec.coerce(raw)
            except (TypeError, ValueError):
                problems.append(f"{spec.label}: '{raw}'")
                continue
            if coerced is None and spec.kind != "bool":
                continue
            values[spec.key] = coerced
        if problems:
            raise ValueError("Could not read these options:\n  " + "\n  ".join(problems))
        # Widget values win; anything only the loaded run knew about is kept.
        merged = dict(self.passthrough_options)
        merged.update(values)
        return merged

    def _build_config(self) -> core.TrainConfig:
        if self.data is None:
            raise ValueError("Open a trace file first.")
        geometry = self._train_geometry()
        if geometry is None:
            raise ValueError("Train start, ISI and pulse count must all be numbers.")
        start, isi, pulses = geometry
        if isi <= 0:
            raise ValueError("ISI must be greater than zero.")
        if pulses < 1:
            raise ValueError("The train needs at least one pulse.")

        parameter_bounds: Dict[str, Any] = {}
        if self.push_bounds.get():
            # Only bounds moved off the registry default are sent. Untouched
            # rows already show the model's own limits, which the fit applies
            # regardless, so forwarding them would add constraints a loaded run
            # never had without changing what the fitter does.
            _, bounds, _ = self._param_state(edited_bounds_only=True)
            parameter_bounds = registry.build_parameter_bounds(self._current_model(), bounds)

        return core.TrainConfig(
            source=str(self.data.source), train_start=start, isi=isi, n_pulses=pulses,
            event_model=self._current_model(), options=self._collect_options(),
            parameter_bounds=parameter_bounds, time_column_hint=self.time_hint.get().strip(),
        )

    # -- outer loop -----------------------------------------------------------
    def _run_train(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        try:
            config = self._build_config()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return

        data = self.data
        self.run_button.configure(state="disabled", text="Running…")
        self.export_button.configure(state="disabled")
        self.progress.start(12)
        self.status.set(f"Running extract_metrics with '{config.event_model}'…")

        def work() -> None:
            try:
                result = core.run_train_fit(
                    data, config, progress=lambda msg: self._queue.put(("status", msg))
                )
                self._queue.put(("done", result))
            except Exception as exc:
                self._queue.put(("error", (exc, traceback.format_exc())))

        self._worker = threading.Thread(target=work, name="train-fit", daemon=True)
        self._worker.start()

    def _poll_worker(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "status":
                    self.status.set(str(payload))
                elif kind == "done":
                    self._finish_run(payload)
                elif kind == "error":
                    self._fail_run(*payload)
        except queue.Empty:
            pass
        self.after(120, self._poll_worker)

    def _finish_run(self, result: core.TrainResult) -> None:
        self.progress.stop()
        self.run_button.configure(state="normal", text="Run train fit")
        self.export_button.configure(state="normal")
        self.result = result
        result.event_fit = self.event_fit

        self.train_panel.render(result)
        self.amp_panel.render(result)
        self._fill_tables(result)

        r2 = result.train_r2
        parts = [f"AMP1 = {result.amp1:.6g}"]
        if np.isfinite(r2):
            parts.append(f"train R2 = {r2:.4f}")
        if result.config.parameter_bounds:
            parts.append(f"{len(result.config.parameter_bounds)} bound(s) applied")
        self.status.set("Train fit complete.  " + "  ·  ".join(parts))

    def _fail_run(self, exc: Exception, detail: str) -> None:
        self.progress.stop()
        self.run_button.configure(state="normal", text="Run train fit")
        self.status.set("Train fit failed.")
        print(detail, file=sys.stderr)
        messagebox.showerror(APP_TITLE, f"{type(exc).__name__}: {exc}")

    def _clear_tables(self) -> None:
        for table in (self.pulse_table, self.metric_table):
            for item in table.get_children():
                table.delete(item)

    def _fill_tables(self, result: core.TrainResult) -> None:
        self._clear_tables()

        def fmt(value: float) -> str:
            return f"{value:.6g}" if np.isfinite(value) else "—"

        for row in result.pulses.itertuples(index=False):
            self.pulse_table.insert("", "end", values=(
                int(row.pulse), fmt(float(row.amp_corrected)), fmt(float(row.amp_uncorrected)),
                fmt(float(row.ppr)), fmt(float(row.tau_decay_ms)),
            ))
        for row in result.metrics.itertuples(index=False):
            self.metric_table.insert("", "end", values=(row.metric, row.value))

    # -- persistence ----------------------------------------------------------
    def _export(self) -> None:
        if self.result is None:
            return
        directory = filedialog.askdirectory(title="Choose an export folder")
        if not directory:
            return
        try:
            written = core.export_results(self.result, directory)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        self.status.set(f"Exported {len(written)} files to {directory}")

    def _save_config(self) -> None:
        try:
            config = self._build_config()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        target = filedialog.asksaveasfilename(
            title="Save configuration", defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile=f"{Path(config.source).stem}_config.json",
        )
        if not target:
            return
        payload = json.loads(config.to_json())
        payload["model_parameters"] = {
            name: {"value": row.value, "bounds": list(row.bounds), "locked": bool(row.locked.get())}
            for name, row in self.param_rows.items()
        }
        Path(target).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        self.status.set(f"Saved configuration to {target}")

    def _generate_script(self) -> None:
        try:
            config = self._build_config()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        target = filedialog.asksaveasfilename(
            title="Save runnable script", defaultextension=".py",
            filetypes=[("Python script", "*.py")],
            initialfile=f"run_{Path(config.source).stem}.py",
        )
        if not target:
            return
        try:
            written = scriptgen.write_script(config, target)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"{type(exc).__name__}: {exc}")
            return
        self.status.set(f"Wrote runnable script to {written}")

    def _choose_condition(self, names: List[str]) -> Optional[str]:
        """Modal picker for run_settings files covering several conditions."""
        dialog = tk.Toplevel(self)
        dialog.title("Choose a condition")
        dialog.configure(bg=self.pal["bg"])
        dialog.transient(self)
        dialog.resizable(False, False)

        frame = ttk.Frame(dialog, padding=14)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="This run covers several conditions.\n"
                              "Load the settings resolved for:").pack(anchor="w")

        choice = tk.StringVar(value=names[0])
        combo = ttk.Combobox(frame, textvariable=choice, values=names, state="readonly",
                             width=34, font=(FONT, 9))
        combo.pack(fill="x", pady=(8, 12))

        result: Dict[str, Optional[str]] = {"value": None}

        def accept() -> None:
            result["value"] = choice.get()
            dialog.destroy()

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Load", style="Accent.TButton", command=accept).pack(side="right")
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right", padx=(0, 6))

        dialog.bind("<Return>", lambda _e: accept())
        dialog.bind("<Escape>", lambda _e: dialog.destroy())
        dialog.update_idletasks()
        dialog.geometry(f"+{self.winfo_rootx() + 180}+{self.winfo_rooty() + 160}")
        combo.focus_set()
        dialog.grab_set()
        self.wait_window(dialog)
        return result["value"]

    def _load_config(self) -> None:
        source = filedialog.askopenfilename(
            title="Load configuration or run_settings.json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not source:
            return
        try:
            kind, names = runsettings.peek(source)
            condition = None
            if kind == runsettings.RUN_SETTINGS and len(names) > 1:
                condition = self._choose_condition(names)
                if condition is None:
                    return
            loaded = runsettings.load(source, condition=condition)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"{type(exc).__name__}: {exc}")
            return

        self._apply_loaded_config(loaded)

        message = f"Loaded {loaded.summary()}"
        if loaded.notes:
            message += "  ·  " + " · ".join(loaded.notes)
        self.status.set(message)
        notes = []
        if loaded.ignored:
            notes.append("Forwarded to the fitter but not shown here: " + ", ".join(loaded.ignored))
        if self._unmatched_bounds:
            notes.append("Dropped, not a parameter of "
                         f"{self._current_model()}: " + ", ".join(self._unmatched_bounds))
        self.train_info.set("  |  ".join(notes))

    def _apply_loaded_config(self, loaded: runsettings.LoadedConfig) -> None:
        """Push a parsed configuration into the widgets."""
        self.passthrough_options = dict(loaded.extra_options)
        for spec in opt.SPECS:
            if spec.kind == "bool":
                self.option_vars[spec.key].set(bool(spec.default))
            else:
                self.option_vars[spec.key].set(spec.to_text(spec.default))
        for key, value in loaded.options.items():
            spec = opt.BY_KEY.get(key)
            if spec is None:
                continue
            if spec.kind == "bool":
                self.option_vars[key].set(bool(value))
            else:
                self.option_vars[key].set(spec.to_text(value))

        if loaded.event_model:
            try:
                self.model_name.set(registry.label_for(loaded.event_model))
            except Exception:
                pass
        if loaded.time_column_hint:
            self.time_hint.set(loaded.time_column_hint)
        for value, var in ((loaded.train_start, self.train_start), (loaded.isi, self.isi),
                           (loaded.n_pulses, self.n_pulses)):
            if value is not None:
                var.set(f"{value:g}")

        if loaded.source and Path(loaded.source).exists():
            self._load_file(loaded.source)

        self._rebuild_option_rows()
        self._rebuild_param_rows()

        # Bounds may be keyed by model parameter or by canonical kinetics name.
        # Anything matching no parameter of the selected model is dropped, so
        # say so rather than letting it disappear on the next save.
        self._unmatched_bounds = []
        for key, (low, high) in loaded.parameter_bounds.items():
            matched = False
            for param, row in self.param_rows.items():
                if param == key or registry.canonical_bound_key(param) == key:
                    row.set_bounds(low, high)
                    matched = True
            if not matched:
                self._unmatched_bounds.append(key)
        for name, spec in loaded.model_parameters.items():
            row = self.param_rows.get(name)
            if row is None:
                continue
            bounds = spec.get("bounds") or []
            if len(bounds) == 2 and all(b is not None for b in bounds):
                row.set_bounds(float(bounds[0]), float(bounds[1]))
            value = spec.get("value")
            if value is not None:
                row.set_value(float(value))
            row.locked.set(bool(spec.get("locked", False)))

        self._refresh_event(refit=False)

    # -- theme ----------------------------------------------------------------
    def _toggle_theme(self) -> None:
        self.theme_name.set("dark" if self.theme_name.get() == "light" else "light")
        self.pal = palette(self.theme_name.get())
        apply_ttk_style(self.style, self.pal)
        self.configure(bg=self.pal["bg"])

        state = {
            "file": self.file_path.get(),
            "model": self.model_name.get(),
            "params": {n: (r.value, r.bounds, r.locked.get()) for n, r in self.param_rows.items()},
        }
        for child in self.winfo_children():
            child.destroy()
        self.param_rows.clear()
        self._build_ui()
        self.model_name.set(state["model"])
        self._rebuild_param_rows()
        for name, (value, bounds, locked) in state["params"].items():
            row = self.param_rows.get(name)
            if row is None:
                continue
            if bounds[0] is not None and bounds[1] is not None:
                row.set_bounds(bounds[0], bounds[1])
            if value is not None:
                row.set_value(value)
            row.locked.set(locked)
        if self.data is not None:
            self.run_button.configure(state="normal")
            self._refresh_event(refit=False)
        if self.result is not None:
            self.train_panel.render(self.result)
            self.amp_panel.render(self.result)
            self._fill_tables(self.result)
            self.export_button.configure(state="normal")


def main() -> None:
    Workbench().mainloop()


if __name__ == "__main__":
    main()
