"""Sweep summary utilities for pipeline results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuro_mod.pipeline.io import save_dataframe, save_metrics
from neuro_mod.visualization import folder_plots_to_pdf


def _coerce_params(value: Any) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(v) for v in value]
    return [float(value)]


def _format_sweep_value(params: list[float]) -> str:
    if len(params) == 1:
        return f"{params[0]:g}"
    return ", ".join(f"{value:g}" for value in params)


def _to_python(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def build_sweep_summary(
    result: Any,
    sweep_values: Iterable[Any] | None = None,
) -> pd.DataFrame:
    """Build a sweep summary DataFrame from PipelineResult metrics."""
    if sweep_values is None:
        sweep_values = None
        if hasattr(result, "sweep_metadata"):
            sweep_values = result.sweep_metadata.get("values")
    sweep_values = list(sweep_values) if sweep_values is not None else []

    rows: list[dict[str, Any]] = []
    if sweep_values:
        for idx, value in enumerate(sweep_values):
            metrics = result.metrics.get(f"sweep_{idx}")
            if metrics is None:
                continue
            params = _coerce_params(value)
            row: dict[str, Any] = {
                "sweep_idx": idx,
                "sweep_value": _format_sweep_value(params),
            }
            for j, param in enumerate(params):
                row[f"param_{j}"] = float(param)
            for key, metric_value in metrics.items():
                row[key] = _to_python(metric_value)
            rows.append(row)
    else:
        for key, metrics in sorted(result.metrics.items()):
            if not key.startswith("sweep_") or key == "aggregated":
                continue
            if "_repeat_" in key:
                continue
            idx = key.split("_", 1)[1]
            row = {"sweep_idx": int(idx)}
            for metric_key, metric_value in metrics.items():
                row[metric_key] = _to_python(metric_value)
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("sweep_idx").reset_index(drop=True)


def save_sweep_summary(save_dir: Path, summary_df: pd.DataFrame) -> None:
    """Persist sweep summary as dataframe, metrics JSON, and index CSV."""
    if summary_df.empty:
        return
    save_dataframe(summary_df, save_dir / "dataframes", "sweep_summary")
    save_metrics(summary_df.to_dict(orient="records"), save_dir / "metrics", "sweep_summary")
    metadata_dir = save_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    index_cols = [col for col in summary_df.columns if col.startswith("param_")]
    cols = ["sweep_idx", "sweep_value", *index_cols]
    summary_df[cols].to_csv(metadata_dir / "sweep_index.csv", index=False)


def _plot_metric_over_sweep(
    summary_df: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    if metric not in summary_df.columns:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="tight")
    if "param_1" not in summary_df.columns and "param_0" in summary_df.columns:
        x = summary_df["param_0"].to_numpy()
        ax.plot(x, summary_df[metric].to_numpy(), marker="o")
        ax.set_xlabel("sweep_value")
    else:
        x = np.arange(len(summary_df))
        labels = summary_df["sweep_value"].astype(str).to_list()
        ax.plot(x, summary_df[metric].to_numpy(), marker="o")
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.set_xlabel("sweep_value")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_duration_by_sweep(aggregated_df: pd.DataFrame, output_path: Path) -> None:
    if aggregated_df.empty:
        return
    if "duration" not in aggregated_df.columns or "sweep_value" not in aggregated_df.columns:
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="tight")
    if "sweep_idx" in aggregated_df.columns:
        order = (
            aggregated_df[["sweep_idx", "sweep_value"]]
            .drop_duplicates(subset=["sweep_idx"])
            .sort_values("sweep_idx")
        )
        data = []
        labels = []
        for _, row in order.iterrows():
            idx = row["sweep_idx"]
            label = str(row["sweep_value"])
            values = aggregated_df.loc[aggregated_df["sweep_idx"] == idx, "duration"].to_numpy()
            if values.size == 0:
                continue
            data.append(values)
            labels.append(label)
    else:
        tmp = aggregated_df.copy()
        tmp["sweep_label"] = tmp["sweep_value"].astype(str)
        grouped = tmp.groupby("sweep_label")["duration"]
        labels = [str(label) for label in grouped.groups.keys()]
        data = [grouped.get_group(key).to_numpy() for key in grouped.groups.keys()]

    if not data:
        plt.close(fig)
        return

    try:
        ax.boxplot(data, tick_labels=labels, showfliers=False)
    except TypeError:
        ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_xlabel("sweep_value")
    ax.set_ylabel("duration (ms)")
    ax.set_title("Duration Distribution by Sweep Value")
    ax.tick_params(axis="x", rotation=45)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_sweep_summary(
    save_dir: Path,
    summary_df: pd.DataFrame,
    aggregated_df: pd.DataFrame | None = None,
    *,
    metrics_to_plot: Iterable[str] | None = None,
) -> None:
    """Generate sweep summary plots and export a PDF."""
    if summary_df.empty:
        return
    plots_dir = save_dir / "plots" / "sweep_summary"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = list(metrics_to_plot) if metrics_to_plot is not None else [
        "num_states",
        "mean_lifespan_ms",
    ]
    for metric in metrics:
        _plot_metric_over_sweep(
            summary_df,
            metric=metric,
            ylabel=metric,
            title=f"{metric} vs Sweep Value",
            output_path=plots_dir / f"{metric}_vs_sweep.png",
        )

    if aggregated_df is not None:
        _plot_duration_by_sweep(
            aggregated_df,
            output_path=plots_dir / "duration_by_sweep.png",
        )

    if any(plots_dir.glob("*.png")):
        try:
            folder_plots_to_pdf(
                plots_dir,
                output_path=plots_dir / "sweep_summary.pdf",
            )
        except ValueError as exc:
            print(f"Skipping sweep summary PDF export: {exc}")
