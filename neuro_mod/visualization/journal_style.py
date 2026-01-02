"""Matplotlib journal-style defaults for physics and neuroscience plots."""

from __future__ import annotations


JOURNAL_FIGURE_SIZES = {
    "single": (3.4, 2.5),
    "double": (7.0, 3.5),
    "wide": (7.0, 2.5),
}


def apply_journal_style() -> None:
    """Apply journal-style matplotlib defaults."""
    import matplotlib as mpl
    from matplotlib import cycler
    from matplotlib.colors import to_rgba

    base_cycle = mpl.rcParams.get("axes.prop_cycle", None)
    if base_cycle is None:
        base_cycle = mpl.rcParamsDefault["axes.prop_cycle"]

    base_colors = base_cycle.by_key().get("color", [])
    if not base_colors:
        base_colors = ["#1f77b4"]
    alpha = 0.5
    alpha_cycle = cycler(color=[to_rgba(c, alpha=alpha) for c in base_colors])

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.linewidth": 1.0,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.minor.width": 0.8,
            "ytick.minor.width": 0.8,
            "lines.linewidth": 1.6,
            "lines.markersize": 6.0,
            "axes.prop_cycle": alpha_cycle,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "mathtext.fontset": "stix",
            "axes.grid": False,
            "figure.figsize": JOURNAL_FIGURE_SIZES["single"],
        }
    )


def reset_journal_style() -> None:
    """Reset matplotlib defaults to their original settings."""
    import matplotlib as mpl

    mpl.rcParams.update(mpl.rcParamsDefault)


def set_figure_size(kind: str = "single", aspect: float | None = None) -> None:
    """Set default figure size for common journal layouts."""
    if kind not in JOURNAL_FIGURE_SIZES:
        raise ValueError(f"Unknown figure size preset: {kind!r}")

    width, height = JOURNAL_FIGURE_SIZES[kind]
    if aspect is not None:
        height = width * aspect

    import matplotlib as mpl

    mpl.rcParams["figure.figsize"] = (width, height)


# Apply defaults on import for convenience.
apply_journal_style()


__all__ = [
    "JOURNAL_FIGURE_SIZES",
    "apply_journal_style",
    "reset_journal_style",
    "set_figure_size",
]
