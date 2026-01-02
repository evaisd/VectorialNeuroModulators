"""Matplotlib journal-style defaults for physics and neuroscience plots.

Clean, scientific style with:
- Thick lines and markers with 0.5 alpha
- Clean sans-serif fonts (Helvetica/Arial style)
- Physics/neuroscience journal conventions
"""

from __future__ import annotations


# Standard journal column widths (inches)
JOURNAL_FIGURE_SIZES = {
    "single": (3.4, 2.8),      # Single column
    "1.5column": (5.0, 3.5),   # 1.5 column
    "double": (7.0, 4.0),      # Double column / full width
    "wide": (7.0, 2.8),        # Wide aspect ratio
    "square": (3.4, 3.4),      # Square format
}

# Scientific color palette - colorblind-friendly, print-safe
SCIENCE_COLORS = [
    "#0173B2",  # Blue
    "#DE8F05",  # Orange
    "#029E73",  # Green
    "#D55E00",  # Vermillion
    "#CC78BC",  # Purple
    "#CA9161",  # Brown
    "#FBAFE4",  # Pink
    "#949494",  # Gray
    "#ECE133",  # Yellow
    "#56B4E9",  # Sky blue
]

# Default alpha for scatter/marker plots
DEFAULT_ALPHA = 0.5


def apply_journal_style(alpha: float = DEFAULT_ALPHA) -> None:
    """Apply clean, scientific journal-style matplotlib defaults.

    Args:
        alpha: Default alpha for colors in the color cycle (default 0.5).
    """
    import matplotlib as mpl
    from matplotlib import cycler
    from matplotlib.colors import to_rgba

    # Create color cycle with alpha
    alpha_colors = [to_rgba(c, alpha=alpha) for c in SCIENCE_COLORS]
    color_cycle = cycler(color=alpha_colors)

    mpl.rcParams.update(
        {
            # Figure
            "figure.dpi": 150,
            "figure.facecolor": "white",
            "figure.edgecolor": "white",
            "figure.figsize": JOURNAL_FIGURE_SIZES["single"],
            "figure.constrained_layout.use": True,

            # Saving
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,

            # Fonts - clean sans-serif
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 11,
            "font.weight": "normal",

            # Text
            "text.color": "black",
            "mathtext.fontset": "dejavusans",

            # Axes
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelweight": "normal",
            "axes.linewidth": 1.2,
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": color_cycle,
            "axes.grid": False,
            "axes.axisbelow": True,

            # Ticks
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.top": False,
            "ytick.right": False,
            "xtick.bottom": True,
            "ytick.left": True,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.color": "black",
            "ytick.color": "black",

            # Lines - THICK
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
            "lines.markeredgewidth": 0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",

            # Scatter
            "scatter.edgecolors": "none",

            # Patches (bars, etc.)
            "patch.linewidth": 1.0,
            "patch.edgecolor": "black",

            # Legend
            "legend.frameon": False,
            "legend.fontsize": 10,
            "legend.labelspacing": 0.3,
            "legend.handletextpad": 0.5,
            "legend.borderaxespad": 0.5,
            "legend.columnspacing": 1.0,

            # Grid (off by default, but styled if enabled)
            "grid.color": "#E5E5E5",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.7,

            # Histogram
            "hist.bins": "auto",

            # Errorbar
            "errorbar.capsize": 3,

            # Image
            "image.cmap": "viridis",
        }
    )


def reset_journal_style() -> None:
    """Reset matplotlib defaults to their original settings."""
    import matplotlib as mpl

    mpl.rcParams.update(mpl.rcParamsDefault)


def set_figure_size(kind: str = "single", aspect: float | None = None) -> None:
    """Set default figure size for common journal layouts.

    Args:
        kind: Figure size preset ('single', '1.5column', 'double', 'wide', 'square').
        aspect: Optional aspect ratio (height = width * aspect).
    """
    if kind not in JOURNAL_FIGURE_SIZES:
        raise ValueError(f"Unknown figure size preset: {kind!r}")

    width, height = JOURNAL_FIGURE_SIZES[kind]
    if aspect is not None:
        height = width * aspect

    import matplotlib as mpl

    mpl.rcParams["figure.figsize"] = (width, height)


def get_color(index: int = 0, alpha: float | None = None) -> tuple:
    """Get a color from the scientific palette.

    Args:
        index: Color index (wraps around if > len(SCIENCE_COLORS)).
        alpha: Optional alpha value (default uses DEFAULT_ALPHA).

    Returns:
        RGBA tuple.
    """
    from matplotlib.colors import to_rgba

    color = SCIENCE_COLORS[index % len(SCIENCE_COLORS)]
    if alpha is None:
        alpha = DEFAULT_ALPHA
    return to_rgba(color, alpha=alpha)


def get_colors(n: int, alpha: float | None = None) -> list[tuple]:
    """Get n colors from the scientific palette.

    Args:
        n: Number of colors to return.
        alpha: Optional alpha value (default uses DEFAULT_ALPHA).

    Returns:
        List of RGBA tuples.
    """
    return [get_color(i, alpha=alpha) for i in range(n)]


def get_solid_color(index: int = 0) -> str:
    """Get a solid (alpha=1.0) color from the scientific palette.

    Args:
        index: Color index (wraps around if > len(SCIENCE_COLORS)).

    Returns:
        Hex color string.
    """
    return SCIENCE_COLORS[index % len(SCIENCE_COLORS)]


# Apply defaults on import for convenience.
apply_journal_style()


__all__ = [
    "JOURNAL_FIGURE_SIZES",
    "SCIENCE_COLORS",
    "DEFAULT_ALPHA",
    "apply_journal_style",
    "reset_journal_style",
    "set_figure_size",
    "get_color",
    "get_colors",
    "get_solid_color",
]
