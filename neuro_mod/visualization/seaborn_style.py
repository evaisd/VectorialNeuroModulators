"""Seaborn integration with journal_style.

This module provides utilities for applying journal-style defaults to seaborn,
building on top of the matplotlib rcParams set by journal_style.

Usage:
    >>> from neuro_mod.visualization import journal_style, seaborn_style
    >>> journal_style.apply_journal_style()  # Sets matplotlib defaults
    >>> seaborn_style.apply_seaborn_journal_style()  # Configures seaborn

    Or simply:
    >>> seaborn_style.apply_full_style()  # Applies both
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import seaborn as sns

# Lazy seaborn import
_sns = None


def _get_seaborn() -> "sns":
    """Lazy import seaborn."""
    global _sns
    if _sns is None:
        import seaborn
        _sns = seaborn
    return _sns


def apply_seaborn_journal_style() -> None:
    """Apply journal-style defaults to seaborn.

    Should be called after journal_style.apply_journal_style()
    to ensure matplotlib rcParams are set first.

    This configures seaborn to:
    - Use the "ticks" style (minimal spines)
    - Disable top and right spines
    - Use the scientific color palette from journal_style
    """
    from neuro_mod.visualization.journal_style import SCIENCE_COLORS

    sns = _get_seaborn()

    # Set seaborn theme that respects matplotlib rcParams
    sns.set_theme(
        style="ticks",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )

    # Set color palette
    sns.set_palette(SCIENCE_COLORS)


def apply_full_style(alpha: float = 0.5) -> None:
    """Apply both journal_style and seaborn_style.

    Convenience function that applies both matplotlib and seaborn styling.

    Args:
        alpha: Alpha value for the color cycle (default 0.5).
    """
    from neuro_mod.visualization import journal_style

    journal_style.apply_journal_style(alpha=alpha)
    apply_seaborn_journal_style()


def get_palette(n_colors: int | None = None) -> list[str]:
    """Get the science color palette for seaborn.

    Args:
        n_colors: Number of colors to return (cycles if > 10).
            If None, returns all 10 colors.

    Returns:
        List of hex color strings.
    """
    from neuro_mod.visualization.journal_style import SCIENCE_COLORS

    if n_colors is None:
        return list(SCIENCE_COLORS)

    return [SCIENCE_COLORS[i % len(SCIENCE_COLORS)] for i in range(n_colors)]


def get_categorical_palette(n_categories: int) -> list[str]:
    """Get a categorical color palette.

    Uses the science palette for up to 10 categories,
    then falls back to seaborn's husl palette for more.

    Args:
        n_categories: Number of categories.

    Returns:
        List of hex color strings.
    """
    from neuro_mod.visualization.journal_style import SCIENCE_COLORS

    if n_categories <= len(SCIENCE_COLORS):
        return list(SCIENCE_COLORS[:n_categories])

    # Fall back to seaborn's husl palette for many categories
    sns = _get_seaborn()
    return sns.husl_palette(n_categories).as_hex()


def get_sequential_palette(n_colors: int = 9, color: str = "blue") -> list[str]:
    """Get a sequential color palette.

    Args:
        n_colors: Number of colors in the gradient.
        color: Base color ("blue", "green", "red", "purple", "orange").

    Returns:
        List of hex color strings from light to dark.
    """
    sns = _get_seaborn()

    palette_map = {
        "blue": "Blues",
        "green": "Greens",
        "red": "Reds",
        "purple": "Purples",
        "orange": "Oranges",
    }

    palette_name = palette_map.get(color, "Blues")
    return sns.color_palette(palette_name, n_colors).as_hex()


def get_diverging_palette(n_colors: int = 11) -> list[str]:
    """Get a diverging color palette (good for centered data).

    Args:
        n_colors: Number of colors (should be odd for clear center).

    Returns:
        List of hex color strings from negative to positive.
    """
    sns = _get_seaborn()
    return sns.color_palette("RdBu_r", n_colors).as_hex()


__all__ = [
    "apply_seaborn_journal_style",
    "apply_full_style",
    "get_palette",
    "get_categorical_palette",
    "get_sequential_palette",
    "get_diverging_palette",
]
