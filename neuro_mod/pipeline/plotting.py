"""Plotting utilities for the experiment pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.figure as mfig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Lazy seaborn import
_seaborn = None


def _get_seaborn():
    """Lazy import seaborn."""
    global _seaborn
    if _seaborn is None:
        import seaborn as sns
        _seaborn = sns
    return _seaborn


@dataclass
class PlotSpec:
    """Specification for a single plot.

    Attributes:
        name: Unique name for the plot (used as filename).
        plot_type: Type of plot ("line", "scatter", "bar", "hist", "heatmap", "box", "violin", "strip", "point").
        x: Column name for x-axis.
        y: Column name for y-axis.
        hue: Column name for color encoding.
        style: Column name for line style encoding.
        size: Column name for size encoding.
        row: Column name for row faceting.
        col: Column name for column faceting.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size (width, height) in inches.
        kwargs: Additional kwargs passed to the seaborn plot function.
    """

    name: str
    plot_type: str
    x: str | None = None
    y: str | None = None
    hue: str | None = None
    style: str | None = None
    size: str | None = None
    row: str | None = None
    col: str | None = None
    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    figsize: tuple[float, float] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


class BasePlotter(ABC):
    """Abstract base class for plotters."""

    @abstractmethod
    def plot(
        self,
        data: pd.DataFrame,
        metrics: dict[str, Any] | None = None,
        save_dir: Path | None = None,
        **kwargs: Any,
    ) -> list[mfig.Figure]:
        """Generate plots from data.

        Args:
            data: DataFrame to visualize.
            metrics: Optional summary metrics.
            save_dir: Optional directory to save plots.
            **kwargs: Additional plotting arguments.

        Returns:
            List of matplotlib Figure objects.
        """
        ...


class SeabornPlotter(BasePlotter):
    """Seaborn-based plotter with journal_style integration.

    Provides both automatic plotting based on data structure
    and manual plot specification via PlotSpec.

    Example:
        >>> plotter = SeabornPlotter()
        >>> figures = plotter.plot(df, save_dir=Path("plots"))

        >>> # With custom specs
        >>> specs = [
        ...     PlotSpec(name="scatter", plot_type="scatter", x="time", y="value", hue="group"),
        ...     PlotSpec(name="hist", plot_type="hist", x="duration_ms"),
        ... ]
        >>> plotter = SeabornPlotter(specs=specs)
    """

    def __init__(
        self,
        specs: list[PlotSpec] | None = None,
        apply_journal_style: bool = True,
        default_alpha: float = 0.5,
        auto_generate: bool = True,
    ) -> None:
        """Initialize the plotter.

        Args:
            specs: Optional list of plot specifications.
            apply_journal_style: Whether to apply journal_style defaults.
            default_alpha: Alpha value for plots.
            auto_generate: Whether to auto-generate plots if no specs provided.
        """
        self.specs = specs or []
        self.apply_journal_style = apply_journal_style
        self.default_alpha = default_alpha
        self.auto_generate = auto_generate

        if apply_journal_style:
            try:
                from neuro_mod.visualization import journal_style
                journal_style.apply_journal_style(alpha=default_alpha)
            except ImportError:
                pass

    def plot(
        self,
        data: pd.DataFrame,
        metrics: dict[str, Any] | None = None,
        save_dir: Path | None = None,
        **kwargs: Any,
    ) -> list[mfig.Figure]:
        """Generate plots from DataFrame.

        Args:
            data: DataFrame to plot.
            metrics: Optional summary metrics.
            save_dir: Optional directory to save plots.
            **kwargs: Additional plotting arguments.

        Returns:
            List of matplotlib Figure objects.
        """
        sns = _get_seaborn()
        figures: list[mfig.Figure] = []

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        # Use provided specs or auto-generate
        specs = self.specs if self.specs else (
            self._auto_generate_specs(data) if self.auto_generate else []
        )

        for spec in specs:
            try:
                fig = self._create_plot(data, spec, sns)
                figures.append(fig)

                if save_dir:
                    fig.savefig(save_dir / f"{spec.name}.png", dpi=300, bbox_inches="tight")
                    plt.close(fig)
            except Exception as e:
                # Log but don't fail on individual plot errors
                import logging
                logging.getLogger("SeabornPlotter").warning(
                    f"Failed to create plot '{spec.name}': {e}"
                )

        return figures

    def _auto_generate_specs(self, data: pd.DataFrame) -> list[PlotSpec]:
        """Auto-generate plot specifications based on data structure."""
        specs: list[PlotSpec] = []

        # Detect column types
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        # Remove index-like columns
        numeric_cols = [c for c in numeric_cols if c not in ("repeat", "sweep_idx")]

        # Check for common grouping columns
        has_sweep = "sweep_value" in data.columns
        has_repeat = "repeat" in data.columns

        # Generate appropriate plots based on structure
        if has_sweep and numeric_cols:
            # Line plot showing metric vs sweep value
            for y_col in numeric_cols[:3]:
                specs.append(PlotSpec(
                    name=f"sweep_{y_col}",
                    plot_type="line",
                    x="sweep_value",
                    y=y_col,
                    hue="repeat" if has_repeat else None,
                    title=f"{y_col} vs Sweep Parameter",
                    kwargs={"errorbar": "sd" if has_repeat else None, "marker": "o"},
                ))

        if has_repeat and numeric_cols:
            # Box plots showing distribution across repeats
            for y_col in numeric_cols[:2]:
                if has_sweep:
                    specs.append(PlotSpec(
                        name=f"box_{y_col}",
                        plot_type="box",
                        x="sweep_value",
                        y=y_col,
                        title=f"Distribution of {y_col}",
                    ))
                else:
                    specs.append(PlotSpec(
                        name=f"hist_{y_col}",
                        plot_type="hist",
                        x=y_col,
                        title=f"Distribution of {y_col}",
                        kwargs={"bins": 30, "kde": True},
                    ))

        # If no grouping columns, generate simple histograms
        if not has_sweep and not has_repeat and numeric_cols:
            for y_col in numeric_cols[:3]:
                specs.append(PlotSpec(
                    name=f"hist_{y_col}",
                    plot_type="hist",
                    x=y_col,
                    title=f"Distribution of {y_col}",
                    kwargs={"bins": 30, "kde": True},
                ))

        return specs

    def _create_plot(
        self,
        data: pd.DataFrame,
        spec: PlotSpec,
        sns: Any,
    ) -> mfig.Figure:
        """Create a single plot from specification."""
        figsize = spec.figsize or (7, 5)

        # Map plot types to seaborn functions
        plot_funcs = {
            "line": sns.lineplot,
            "scatter": sns.scatterplot,
            "bar": sns.barplot,
            "hist": sns.histplot,
            "heatmap": sns.heatmap,
            "box": sns.boxplot,
            "violin": sns.violinplot,
            "strip": sns.stripplot,
            "point": sns.pointplot,
            "kde": sns.kdeplot,
        }

        plot_func = plot_funcs.get(spec.plot_type)
        if plot_func is None:
            raise ValueError(f"Unknown plot type: {spec.plot_type}")

        # Handle faceting separately
        if spec.row or spec.col:
            return self._create_faceted_plot(data, spec, sns, plot_func)

        # Create single figure
        fig, ax = plt.subplots(figsize=figsize)

        # Build kwargs
        plot_kwargs: dict[str, Any] = {
            "data": data,
            "ax": ax,
            **spec.kwargs,
        }

        # Add optional parameters
        for param in ["x", "y", "hue", "style", "size"]:
            value = getattr(spec, param)
            if value is not None:
                plot_kwargs[param] = value

        # Special handling for heatmap (no x, y, hue)
        if spec.plot_type == "heatmap":
            plot_kwargs.pop("x", None)
            plot_kwargs.pop("y", None)
            plot_kwargs.pop("hue", None)
            plot_kwargs.pop("style", None)
            plot_kwargs.pop("size", None)

        plot_func(**plot_kwargs)

        # Apply labels
        if spec.title:
            ax.set_title(spec.title)
        if spec.xlabel:
            ax.set_xlabel(spec.xlabel)
        if spec.ylabel:
            ax.set_ylabel(spec.ylabel)

        return fig

    def _create_faceted_plot(
        self,
        data: pd.DataFrame,
        spec: PlotSpec,
        sns: Any,
        plot_func: Any,
    ) -> mfig.Figure:
        """Create a faceted plot using FacetGrid."""
        facet_kwargs = spec.kwargs.pop("facet_kwargs", {})

        g = sns.FacetGrid(
            data,
            row=spec.row,
            col=spec.col,
            **facet_kwargs,
        )

        # Build map kwargs
        map_kwargs: dict[str, Any] = {}
        for param in ["x", "y", "hue", "style", "size"]:
            value = getattr(spec, param)
            if value is not None:
                map_kwargs[param] = value

        map_kwargs.update(spec.kwargs)

        g.map_dataframe(plot_func, **map_kwargs)
        g.add_legend()

        if spec.title:
            g.figure.suptitle(spec.title, y=1.02)

        return g.figure


class ComposablePlotter(BasePlotter):
    """Plotter that composes multiple sub-plotters."""

    def __init__(self, plotters: list[BasePlotter]) -> None:
        """Initialize with a list of plotters.

        Args:
            plotters: List of plotter instances to compose.
        """
        self.plotters = plotters

    def plot(
        self,
        data: pd.DataFrame,
        metrics: dict[str, Any] | None = None,
        save_dir: Path | None = None,
        **kwargs: Any,
    ) -> list[mfig.Figure]:
        """Generate plots from all composed plotters.

        Args:
            data: DataFrame to visualize.
            metrics: Optional summary metrics.
            save_dir: Optional directory to save plots.
            **kwargs: Additional plotting arguments.

        Returns:
            Combined list of Figure objects from all plotters.
        """
        figures: list[mfig.Figure] = []
        for plotter in self.plotters:
            figures.extend(plotter.plot(data, metrics, save_dir, **kwargs))
        return figures


class MatplotlibPlotter(BasePlotter):
    """Simple matplotlib-based plotter for basic plots.

    Use this when seaborn is not needed or for custom matplotlib plots.
    """

    def __init__(
        self,
        plot_functions: list[Any] | None = None,
        apply_journal_style: bool = True,
    ) -> None:
        """Initialize the plotter.

        Args:
            plot_functions: Optional list of custom plot functions.
                Each function should have signature:
                (data: pd.DataFrame, ax: plt.Axes, **kwargs) -> None
            apply_journal_style: Whether to apply journal_style defaults.
        """
        self.plot_functions = plot_functions or []

        if apply_journal_style:
            try:
                from neuro_mod.visualization import journal_style
                journal_style.apply_journal_style()
            except ImportError:
                pass

    def plot(
        self,
        data: pd.DataFrame,
        metrics: dict[str, Any] | None = None,
        save_dir: Path | None = None,
        **kwargs: Any,
    ) -> list[mfig.Figure]:
        """Generate plots using custom matplotlib functions.

        Args:
            data: DataFrame to visualize.
            metrics: Optional summary metrics.
            save_dir: Optional directory to save plots.
            **kwargs: Additional plotting arguments.

        Returns:
            List of matplotlib Figure objects.
        """
        figures: list[mfig.Figure] = []

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        for i, plot_fn in enumerate(self.plot_functions):
            fig, ax = plt.subplots()
            plot_fn(data, ax, metrics=metrics, **kwargs)
            figures.append(fig)

            if save_dir:
                fig.savefig(save_dir / f"plot_{i}.png", dpi=300, bbox_inches="tight")
                plt.close(fig)

        return figures


__all__ = [
    "PlotSpec",
    "BasePlotter",
    "SeabornPlotter",
    "ComposablePlotter",
    "MatplotlibPlotter",
]
