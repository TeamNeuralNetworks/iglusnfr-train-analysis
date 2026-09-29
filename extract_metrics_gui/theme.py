"""Palette and ttk/matplotlib styling for the fitting GUI.

Two restrained palettes (light/dark) that drive both the Tk widgets and the
matplotlib panels, so the figure never looks pasted onto the window.
"""

from __future__ import annotations

from typing import Dict


LIGHT: Dict[str, str] = {
    "name": "light",
    "bg": "#faf9f7",
    "panel": "#ffffff",
    "panel_alt": "#f2f0ec",
    "border": "#dcd8d0",
    "text": "#22262b",
    "text_dim": "#6b7280",
    "accent": "#b45309",
    "accent_text": "#ffffff",
    "trace": "#1f4e79",
    "model": "#d97706",
    "residual": "#64748b",
    "good": "#0f766e",
    "warn": "#b45309",
    "bad": "#b91c1c",
    "snippet": "#9aa4b2",
    "grid": "#c9ccd1",
}

DARK: Dict[str, str] = {
    "name": "dark",
    "bg": "#1c1f24",
    "panel": "#24282e",
    "panel_alt": "#2b3037",
    "border": "#3a4048",
    "text": "#e6e8ea",
    "text_dim": "#98a0ab",
    "accent": "#e08c2a",
    "accent_text": "#1c1f24",
    "trace": "#69b7ff",
    "model": "#f0a94c",
    "residual": "#9aa4b2",
    "good": "#34d399",
    "warn": "#fbbf24",
    "bad": "#f87171",
    "snippet": "#5d6673",
    "grid": "#3d434b",
}

FONT = "Segoe UI"
MONO = "Consolas"


def palette(name: str) -> Dict[str, str]:
    return DARK if str(name).lower() == "dark" else LIGHT


def apply_ttk_style(style, pal: Dict[str, str]) -> None:
    """Configure the shared ttk styles for one palette."""
    style.theme_use("clam")

    style.configure(".", background=pal["bg"], foreground=pal["text"], font=(FONT, 9))
    style.configure("TFrame", background=pal["bg"])
    style.configure("Panel.TFrame", background=pal["panel"])
    style.configure("TLabel", background=pal["bg"], foreground=pal["text"])
    style.configure("Dim.TLabel", background=pal["bg"], foreground=pal["text_dim"], font=(FONT, 8))
    style.configure("Head.TLabel", background=pal["bg"], foreground=pal["text"], font=(FONT, 13, "bold"))
    style.configure("Section.TLabel", background=pal["bg"], foreground=pal["text"], font=(FONT, 9, "bold"))
    style.configure("Mono.TLabel", background=pal["bg"], foreground=pal["text"], font=(MONO, 9))
    style.configure("Good.TLabel", background=pal["bg"], foreground=pal["good"], font=(FONT, 8, "bold"))
    style.configure("Warn.TLabel", background=pal["bg"], foreground=pal["warn"], font=(FONT, 8, "bold"))
    style.configure("Bad.TLabel", background=pal["bg"], foreground=pal["bad"], font=(FONT, 8, "bold"))

    style.configure(
        "TButton", background=pal["panel_alt"], foreground=pal["text"],
        bordercolor=pal["border"], focuscolor=pal["bg"], padding=(9, 5), font=(FONT, 9),
    )
    style.map(
        "TButton",
        background=[("active", pal["border"]), ("disabled", pal["panel_alt"])],
        foreground=[("disabled", pal["text_dim"])],
    )
    style.configure(
        "Accent.TButton", background=pal["accent"], foreground=pal["accent_text"],
        bordercolor=pal["accent"], padding=(11, 6), font=(FONT, 9, "bold"),
    )
    style.map(
        "Accent.TButton",
        background=[("active", pal["warn"]), ("disabled", pal["panel_alt"])],
        foreground=[("disabled", pal["text_dim"])],
    )
    style.configure(
        "Link.TButton", background=pal["bg"], foreground=pal["text_dim"],
        bordercolor=pal["bg"], relief="flat", padding=(3, 1), font=(FONT, 8),
    )
    style.map("Link.TButton", background=[("active", pal["panel_alt"])], foreground=[("active", pal["text"])])

    style.configure(
        "TEntry", fieldbackground=pal["panel"], foreground=pal["text"],
        bordercolor=pal["border"], insertcolor=pal["text"], padding=3,
    )
    style.configure(
        "Bad.TEntry", fieldbackground=pal["panel"], foreground=pal["bad"], bordercolor=pal["bad"], padding=3,
    )
    style.configure(
        "TCombobox", fieldbackground=pal["panel"], background=pal["panel"],
        foreground=pal["text"], bordercolor=pal["border"], arrowcolor=pal["text_dim"], padding=3,
    )
    style.map("TCombobox", fieldbackground=[("readonly", pal["panel"])], foreground=[("readonly", pal["text"])])

    style.configure(
        "TCheckbutton", background=pal["bg"], foreground=pal["text"],
        indicatorcolor=pal["panel"], focuscolor=pal["bg"],
    )
    style.map("TCheckbutton", background=[("active", pal["bg"])], indicatorcolor=[("selected", pal["accent"])])

    style.configure(
        "Horizontal.TScale", background=pal["bg"], troughcolor=pal["panel_alt"], bordercolor=pal["border"],
    )
    style.configure(
        "Treeview", background=pal["panel"], fieldbackground=pal["panel"],
        foreground=pal["text"], bordercolor=pal["border"], rowheight=21, font=(MONO, 9),
    )
    style.configure(
        "Treeview.Heading", background=pal["panel_alt"], foreground=pal["text"], font=(FONT, 9, "bold"),
    )
    style.map("Treeview", background=[("selected", pal["accent"])], foreground=[("selected", pal["accent_text"])])

    style.configure("TNotebook", background=pal["bg"], bordercolor=pal["border"])
    style.configure(
        "TNotebook.Tab", background=pal["panel_alt"], foreground=pal["text_dim"], padding=(12, 6), font=(FONT, 9),
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", pal["bg"])],
        foreground=[("selected", pal["text"])],
    )

    style.configure(
        "TPanedwindow", background=pal["bg"])
    style.configure("Sash", sashthickness=6, gripcount=0, background=pal["border"])
    style.configure(
        "Vertical.TScrollbar", background=pal["panel_alt"], troughcolor=pal["bg"],
        bordercolor=pal["bg"], arrowcolor=pal["text_dim"],
    )
    style.configure(
        "Horizontal.TProgressbar", background=pal["accent"], troughcolor=pal["panel_alt"], bordercolor=pal["border"],
    )
    style.configure("TSeparator", background=pal["border"])


def style_axes(ax, pal: Dict[str, str]) -> None:
    """Apply the palette to one matplotlib axes."""
    ax.set_facecolor(pal["panel"])
    ax.tick_params(colors=pal["text_dim"], labelsize=8, length=3)
    ax.xaxis.label.set_color(pal["text_dim"])
    ax.yaxis.label.set_color(pal["text_dim"])
    ax.xaxis.label.set_fontsize(9)
    ax.yaxis.label.set_fontsize(9)
    ax.title.set_color(pal["text"])
    for side, visible in (("top", False), ("right", False), ("left", True), ("bottom", True)):
        ax.spines[side].set_visible(visible)
        if visible:
            ax.spines[side].set_color(pal["border"])
    ax.grid(True, color=pal["grid"], alpha=0.35, linewidth=0.6)
    ax.set_axisbelow(True)
