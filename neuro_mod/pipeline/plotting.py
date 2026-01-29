"""Plotting utilities for the experiment pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import logging
import matplotlib.figure as mfig
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from neuro_mod.analysis.base_analyzer import BaseAnalyzer, MetricResult

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

    Core fields support analyzer-driven specs; legacy fields support DataFrame plotting.
    """

    name: str
    manipulation: str | None = None
    metric: str | None = None
    plot_type: str | None = None
    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    manipulation_kwargs: dict[str, Any] = field(default_factory=dict)
    metric_kwargs: dict[str, Any] = field(default_factory=dict)
    plot_kwargs: dict[str, Any] = field(default_factory=dict)
    figsize: tuple[float, float] = (7.0, 5.0)
    # Legacy DataFrame plotting fields
    x: str | None = None
    y: str | None = None
    hue: str | None = None
    style: str | None = None
    size: str | None = None
    row: str | None = None
    col: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    def resolved_plot_kwargs(self) -> dict[str, Any]:
        if self.plot_kwargs and self.kwargs:
            merged = dict(self.kwargs)
            merged.update(self.plot_kwargs)
            return merged
        return self.plot_kwargs or self.kwargs


def _resolve_dataframe(analyzer: BaseAnalyzer | pd.DataFrame) -> pd.DataFrame:
    if isinstance(analyzer, pd.DataFrame):
        return analyzer
    if hasattr(analyzer, "df"):
        return analyzer.df
    raise TypeError("Expected analyzer with .df or a pandas DataFrame")


class BasePlotter(ABC):
    """Abstract base class for plotters."""

    @abstractmethod
    def plot(
        self,
        analyzer: BaseAnalyzer,
        save_dir: Path | None = None,
        **kwargs: Any,
    ) -> list[mfig.Figure]:
        """Generate plots from an analyzer.

        Args:
            analyzer: Analyzer providing DataFrames and metrics.
            save_dir: Optional directory to save plots.
            **kwargs: Additional plotting arguments.

        Returns:
            List of matplotlib Figure objects.
        """
        ...


class SpecPlotter(BasePlotter):
    """Spec-driven plotter for analyzers."""

    def __init__(self, specs: list[PlotSpec]) -> None:
        self.specs = specs

    def plot(
        self,
        analyzer: BaseAnalyzer,
        save_dir: Path | None = None,
        **kwargs: Any,
    ) -> list[mfig.Figure]:
        figures: list[mfig.Figure] = []

        if isinstance(analyzer, pd.DataFrame):
            raise TypeError("SpecPlotter requires an analyzer, not a DataFrame")

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        for spec in self.specs:
            fig = self._create_spec_plot(analyzer, spec)
            figures.append(fig)
            if save_dir:
                fig.savefig(save_dir / f"{spec.name}.png", dpi=300, bbox_inches="tight")
                plt.close(fig)

        return figures

    def _create_spec_plot(self, analyzer: BaseAnalyzer, spec: PlotSpec) -> mfig.Figure:
        if spec.plot_type is None:
            raise ValueError(f"PlotSpec '{spec.name}' missing plot_type")
        if spec.metric is None:
            raise ValueError(f"PlotSpec '{spec.name}' missing metric")

        df = None
        if spec.manipulation:
            df = analyzer.manipulation(spec.manipulation, **spec.manipulation_kwargs)
        result = analyzer.metric(spec.metric, df=df, **spec.metric_kwargs)
        fig, ax = plt.subplots(figsize=spec.figsize)
        plot_kwargs = dict(spec.resolved_plot_kwargs())

        self._render_plot(ax, spec.plot_type, result, plot_kwargs)

        if spec.title:
            ax.set_title(spec.title)
        if spec.xlabel:
            ax.set_xlabel(spec.xlabel)
        if spec.ylabel:
            ax.set_ylabel(spec.ylabel)

        return fig

    def _render_plot(
        self,
        ax: plt.Axes,
        plot_type: str,
        result: MetricResult,
        plot_kwargs: dict[str, Any],
    ) -> None:
        if plot_type == "scatter":
            ax.scatter(result.x, result.y, **plot_kwargs)
        elif plot_type == "line":
            ax.plot(result.x, result.y, **plot_kwargs)
        elif plot_type == "hist":
            ax.hist(result.x, **plot_kwargs)
        elif plot_type == "bar":
            ax.bar(result.x, result.y, **plot_kwargs)
        elif plot_type == "heatmap":
            if result.x.size == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_axis_off()
                return
            cmap = plot_kwargs.pop("cmap", "magma")
            log_transform = plot_kwargs.pop("log_transform", False)
            log_eps = plot_kwargs.pop("log_eps", 1e-6)
            cbar_override = plot_kwargs.pop("colorbar_label", None)
            data = result.x
            if log_transform:
                data = np.log10(data + log_eps)
            im = ax.imshow(data, cmap=cmap, aspect="auto", **plot_kwargs)
            metadata = result.metadata or {}
            index_labels = metadata.get("index")
            column_labels = metadata.get("columns")
            if index_labels is not None and len(index_labels) <= 20:
                ax.set_yticks(np.arange(len(index_labels)))
                ax.set_yticklabels(index_labels)
            if column_labels is not None and len(column_labels) <= 20:
                ax.set_xticks(np.arange(len(column_labels)))
                ax.set_xticklabels(column_labels, rotation=90)
            cbar_label = cbar_override or metadata.get("colorbar_label")
            plt.colorbar(im, ax=ax, label=cbar_label)
        else:
            raise ValueError(f"Unsupported plot_type '{plot_type}'")


class SeabornPlotter(BasePlotter):
    """Seaborn-based plotter with journal_style integration.

    Uses analyzer manipulations/metrics to build plots from MetricResult.
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
            auto_generate: Unused (kept for API compatibility).
        """
        self.specs = specs or []
        self.apply_journal_style = apply_journal_style
        self.default_alpha = default_alpha

        if apply_journal_style:
            try:
                from neuro_mod.visualization import journal_style
                journal_style.apply_journal_style(alpha=default_alpha)
            except ImportError:
                pass

    def plot(
        self,
        analyzer: BaseAnalyzer,
        save_dir: Path | None = None,
        **kwargs: Any,
    ) -> list[mfig.Figure]:
        """Generate plots from analyzer metrics."""
        if isinstance(analyzer, pd.DataFrame):
            raise TypeError("SeabornPlotter requires an analyzer, not a DataFrame")

        sns = _get_seaborn()
        figures: list[mfig.Figure] = []

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        if not self.specs:
            logging.getLogger("SeabornPlotter").warning("No plot specs provided.")
            return figures

        for spec in self.specs:
            try:
                fig = self._create_metric_plot(analyzer, spec, sns)
                figures.append(fig)

                if save_dir:
                    fig.savefig(save_dir / f"{spec.name}.png", dpi=300, bbox_inches="tight")
                    plt.close(fig)
            except Exception as e:
                logging.getLogger("SeabornPlotter").warning(
                    f"Failed to create plot '{spec.name}': {e}"
                )

        return figures

    def _create_metric_plot(
        self,
        analyzer: BaseAnalyzer,
        spec: PlotSpec,
        sns: Any,
    ) -> mfig.Figure:
        """Create a single plot from analyzer-driven metric results."""
        if spec.plot_type is None:
            raise ValueError(f"PlotSpec '{spec.name}' missing plot_type")
        if spec.metric is None:
            raise ValueError(f"PlotSpec '{spec.name}' missing metric")

        df = (
            analyzer.manipulation(spec.manipulation, **spec.manipulation_kwargs)
            if spec.manipulation
            else analyzer.df
        )
        result = analyzer.metric(spec.metric, df=df, **spec.metric_kwargs)

        fig, ax = plt.subplots(figsize=spec.figsize or (7, 5))
        plot_kwargs = dict(spec.resolved_plot_kwargs())
        xscale = plot_kwargs.pop("xscale", None)
        yscale = plot_kwargs.pop("yscale", None)

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

        if spec.plot_type == "heatmap":
            if result.x.size == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_axis_off()
                return fig
            plot_func(result.x, ax=ax, **plot_kwargs)
            metadata = result.metadata or {}
            index_labels = metadata.get("index")
            column_labels = metadata.get("columns")
            if index_labels is not None and len(index_labels) <= 20:
                ax.set_yticklabels(index_labels, rotation=0)
            if column_labels is not None and len(column_labels) <= 20:
                ax.set_xticklabels(column_labels, rotation=90)
            cbar_label = metadata.get("colorbar_label")
            if cbar_label and ax.collections and ax.collections[0].colorbar:
                ax.collections[0].colorbar.set_label(cbar_label)
        else:
            df_metric, x_col, y_col, hue_col, size_col = self._metric_to_dataframe(result, spec)
            plot_kwargs["data"] = df_metric
            plot_kwargs["ax"] = ax
            if spec.plot_type in {"line", "scatter", "bar", "box", "violin", "strip", "point"}:
                if y_col is None:
                    raise ValueError(
                        f"PlotSpec '{spec.name}' requires y values for '{spec.plot_type}'"
                    )
                plot_kwargs["x"] = x_col
                plot_kwargs["y"] = y_col
            elif spec.plot_type in {"hist", "kde"}:
                plot_kwargs["x"] = x_col
            if hue_col:
                plot_kwargs["hue"] = hue_col
            if size_col and spec.plot_type == "scatter":
                plot_kwargs["size"] = size_col
                plot_kwargs.pop("s", None)
            if spec.plot_type != "scatter":
                plot_kwargs.pop("s", None)
            legend_mode = plot_kwargs.pop("legend_mode", None)
            if legend_mode == "hue":
                plot_kwargs["legend"] = False
            errorbar_kwargs = plot_kwargs.pop("errorbar_kwargs", None)
            plot_func(**plot_kwargs)
            if legend_mode == "hue" and hue_col:
                self._add_hue_legend(
                    ax,
                    df_metric[hue_col],
                    plot_kwargs.get("palette"),
                    (result.metadata or {}).get("legend_title") or hue_col,
                    plot_type=spec.plot_type,
                    linewidth=plot_kwargs.get("linewidth"),
                )
            self._apply_error_bars(ax, result, errorbar_kwargs, plot_kwargs)

        if spec.title:
            ax.set_title(spec.title)
        if spec.xlabel:
            ax.set_xlabel(spec.xlabel)
        if spec.ylabel:
            ax.set_ylabel(spec.ylabel)
        if xscale:
            ax.set_xscale(xscale)
        if yscale:
            ax.set_yscale(yscale)

        return fig

    def _metric_to_dataframe(
        self,
        result: MetricResult,
        spec: PlotSpec,
    ) -> tuple[pd.DataFrame, str, str | None, str | None, str | None]:
        x_name = spec.x or "x"
        y_name = spec.y or "y"
        hue_name = spec.hue or "label"
        size_name = spec.size

        data: dict[str, Any] = {x_name: result.x}
        if result.y is not None and result.y.size:
            data[y_name] = result.y
        if result.labels is not None:
            data[hue_name] = result.labels
        if size_name:
            metadata = result.metadata or {}
            if size_name in metadata:
                data[size_name] = metadata[size_name]

        df = pd.DataFrame(data)
        y_col = y_name if y_name in df.columns else None
        hue_col = hue_name if hue_name in df.columns else None
        size_col = size_name if size_name in df.columns else None
        return df, x_name, y_col, hue_col, size_col

    def _add_hue_legend(
        self,
        ax: plt.Axes,
        hue_values: pd.Series,
        palette: str | list[str] | None,
        title: str | None,
        *,
        plot_type: str,
        linewidth: float | None,
    ) -> None:
        values = hue_values.dropna().unique()
        if values.size == 0:
            return
        if np.issubdtype(values.dtype, np.number):
            values = np.sort(values)
        n_colors = len(values)
        colors = _get_seaborn().color_palette(palette, n_colors=n_colors)

        handles = []
        labels = [str(v) for v in values]
        for color in colors:
            if plot_type == "line":
                handles.append(Line2D([0], [0], color=color, lw=linewidth or 2.0))
            else:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="none",
                        markerfacecolor=color,
                        markersize=6,
                    )
                )
        ax.legend(handles, labels, title=title)

    def _apply_error_bars(
        self,
        ax: plt.Axes,
        result: MetricResult,
        errorbar_kwargs: dict[str, Any] | None,
        plot_kwargs: dict[str, Any],
    ) -> None:
        metadata = result.metadata or {}
        yerr = metadata.get("yerr")
        xerr = metadata.get("xerr")
        if yerr is None and xerr is None:
            return
        if result.y is None or result.y.size == 0:
            return
        err_kwargs = dict(errorbar_kwargs or {})
        err_kwargs.setdefault("fmt", "none")
        err_kwargs.setdefault("alpha", plot_kwargs.get("alpha", self.default_alpha))
        color = plot_kwargs.get("color")
        if color and "ecolor" not in err_kwargs:
            err_kwargs["ecolor"] = color
        ax.errorbar(result.x, result.y, xerr=xerr, yerr=yerr, **err_kwargs)


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
        analyzer: BaseAnalyzer,
        save_dir: Path | None = None,
        **kwargs: Any,
    ) -> list[mfig.Figure]:
        """Generate plots from all composed plotters.

        Args:
            analyzer: Analyzer to visualize.
            save_dir: Optional directory to save plots.
            **kwargs: Additional plotting arguments.

        Returns:
            Combined list of Figure objects from all plotters.
        """
        figures: list[mfig.Figure] = []
        is_dataframe = isinstance(analyzer, pd.DataFrame)
        for plotter in self.plotters:
            if is_dataframe and isinstance(plotter, (SpecPlotter, SeabornPlotter)):
                logging.getLogger("ComposablePlotter").warning(
                    "Skipping analyzer-only plotter because analyzer is a DataFrame"
                )
                continue
            figures.extend(plotter.plot(analyzer, save_dir, **kwargs))
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
        analyzer: BaseAnalyzer,
        save_dir: Path | None = None,
        **kwargs: Any,
    ) -> list[mfig.Figure]:
        """Generate plots using custom matplotlib functions.

        Args:
            analyzer: Analyzer providing the base DataFrame.
            save_dir: Optional directory to save plots.
            **kwargs: Additional plotting arguments.

        Returns:
            List of matplotlib Figure objects.
        """
        figures: list[mfig.Figure] = []
        metrics = kwargs.pop("metrics", None) or {}
        if isinstance(analyzer, pd.DataFrame):
            data = analyzer
        else:
            data = _resolve_dataframe(analyzer)
            metrics = analyzer.get_summary_metrics()

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
    "SpecPlotter",
    "SeabornPlotter",
    "ComposablePlotter",
    "MatplotlibPlotter",
]
