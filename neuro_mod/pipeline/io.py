"""I/O utilities for pipeline reproducibility and persistence."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def get_git_commit() -> str | None:
    """Get the current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def get_timestamp() -> str:
    """Get the current timestamp in ISO format."""
    return datetime.now().isoformat()


def save_config(config: Any, path: Path) -> None:
    """Save pipeline config to JSON.

    Args:
        config: PipelineConfig instance with to_dict() method.
        path: Path to save the config file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)


def load_config(path: Path) -> dict[str, Any]:
    """Load pipeline config from JSON.

    Args:
        path: Path to the config file.

    Returns:
        Config dictionary.
    """
    with open(path) as f:
        return json.load(f)


def save_metadata(
    save_dir: Path,
    seeds: list[int],
    timestamp: str,
    git_commit: str | None,
    sweep_metadata: dict[str, Any] | None = None,
    duration_seconds: float = 0.0,
) -> None:
    """Save reproducibility metadata.

    Args:
        save_dir: Directory to save metadata.
        seeds: List of seeds used.
        timestamp: Execution timestamp.
        git_commit: Git commit hash.
        sweep_metadata: Optional sweep parameters metadata.
        duration_seconds: Total execution duration.
    """
    metadata_dir = save_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "seeds": seeds,
        "timestamp": timestamp,
        "git_commit": git_commit,
        "duration_seconds": duration_seconds,
    }
    if sweep_metadata:
        metadata["sweep"] = sweep_metadata

    with open(metadata_dir / "pipeline_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Also save seeds as numpy for easy loading
    np.save(metadata_dir / "seeds.npy", np.array(seeds))


def load_metadata(save_dir: Path) -> dict[str, Any]:
    """Load reproducibility metadata.

    Args:
        save_dir: Directory containing metadata.

    Returns:
        Metadata dictionary.
    """
    metadata_path = save_dir / "metadata" / "pipeline_metadata.json"
    with open(metadata_path) as f:
        return json.load(f)


def save_dataframe(
    df: pd.DataFrame,
    path: Path,
    name: str,
    *,
    compression: str = "zstd",
    float_precision: str = "float32",
) -> None:
    """Save a DataFrame to parquet format (or CSV as fallback).

    Args:
        df: DataFrame to save.
        path: Directory to save in.
        name: Name for the file (without extension).
    """
    path.mkdir(parents=True, exist_ok=True)
    df_to_save = df.copy()
    if float_precision in ("float32", "float64"):
        float_cols = df_to_save.select_dtypes(include=["float"]).columns
        if len(float_cols) > 0:
            df_to_save[float_cols] = df_to_save[float_cols].astype(float_precision)

    compression_arg = None if compression == "none" else compression
    try:
        df_to_save.to_parquet(
            path / f"{name}.parquet",
            index=False,
            compression=compression_arg,
        )
    except ImportError:
        # Fall back to CSV if parquet engines not available
        df_to_save.to_csv(path / f"{name}.csv", index=False)


def load_dataframe(path: Path, name: str) -> pd.DataFrame:
    """Load a DataFrame from parquet format (or CSV as fallback).

    Args:
        path: Directory containing the file.
        name: Name of the file (without extension).

    Returns:
        Loaded DataFrame.
    """
    parquet_path = path / f"{name}.parquet"
    csv_path = path / f"{name}.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"No parquet or CSV file found for '{name}' in {path}")


def save_metrics(metrics: dict[str, Any], path: Path, name: str) -> None:
    """Save metrics dictionary to JSON.

    Args:
        metrics: Metrics dictionary.
        path: Directory to save in.
        name: Name for the file (without extension).
    """
    path.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    def convert(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    with open(path / f"{name}.json", "w") as f:
        json.dump(convert(metrics), f, indent=2)


def load_metrics(path: Path, name: str) -> dict[str, Any]:
    """Load metrics dictionary from JSON.

    Args:
        path: Directory containing the file.
        name: Name of the file (without extension).

    Returns:
        Metrics dictionary.
    """
    with open(path / f"{name}.json") as f:
        return json.load(f)


def _filter_dataframes(
    dataframes: dict[str, pd.DataFrame],
    policy: str,
) -> dict[str, pd.DataFrame]:
    if policy == "none":
        return {}
    if policy == "aggregated_only":
        return {"aggregated": dataframes["aggregated"]} if "aggregated" in dataframes else {}
    if policy == "base":
        filtered: dict[str, pd.DataFrame] = {}
        for name, df in dataframes.items():
            if name == "aggregated":
                filtered[name] = df
                continue
            if name.endswith(("_per_attractor", "_time", "_tpm")):
                continue
            filtered[name] = df
        return filtered
    return dataframes


def save_result(result: Any, save_dir: Path, *, config: Any | None = None) -> None:
    """Save full PipelineResult to disk.

    Args:
        result: PipelineResult instance.
        save_dir: Directory to save results.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    save_config(result.config, save_dir / "metadata" / "pipeline_config.json")

    # Save metadata
    save_metadata(
        save_dir=save_dir,
        seeds=result.seeds_used,
        timestamp=result.timestamp,
        git_commit=result.git_commit,
        sweep_metadata=result.sweep_metadata,
        duration_seconds=result.duration_seconds,
    )

    effective_config = config if config is not None else getattr(result, "config", None)
    save_dataframes = getattr(effective_config, "save_dataframes", True)
    save_metrics_flag = getattr(effective_config, "save_metrics", True)
    df_policy = getattr(effective_config, "analysis_dataframe_policy", "all")
    df_compression = getattr(effective_config, "dataframe_compression", "zstd")
    df_precision = getattr(effective_config, "dataframe_float_precision", "float32")

    # Save dataframes
    if save_dataframes:
        df_dir = save_dir / "dataframes"
        filtered = _filter_dataframes(result.dataframes, df_policy)
        for name, df in filtered.items():
            save_dataframe(
                df,
                df_dir,
                name,
                compression=df_compression,
                float_precision=df_precision,
            )

    # Save metrics
    if save_metrics_flag:
        metrics_dir = save_dir / "metrics"
        for name, metrics in result.metrics.items():
            save_metrics(metrics, metrics_dir, name)


def load_result(save_dir: Path) -> dict[str, Any]:
    """Load PipelineResult data from disk.

    Args:
        save_dir: Directory containing saved results.

    Returns:
        Dictionary with config, metadata, dataframes, and metrics.
    """
    save_dir = Path(save_dir)

    result_data: dict[str, Any] = {
        "config": load_config(save_dir / "metadata" / "pipeline_config.json"),
        "metadata": load_metadata(save_dir),
        "dataframes": {},
        "metrics": {},
    }

    # Load dataframes
    df_dir = save_dir / "dataframes"
    if df_dir.exists():
        for parquet_file in df_dir.glob("*.parquet"):
            name = parquet_file.stem
            result_data["dataframes"][name] = load_dataframe(df_dir, name)

    # Load metrics
    metrics_dir = save_dir / "metrics"
    if metrics_dir.exists():
        for json_file in metrics_dir.glob("*.json"):
            name = json_file.stem
            result_data["metrics"][name] = load_metrics(metrics_dir, name)

    return result_data


__all__ = [
    "get_git_commit",
    "get_timestamp",
    "save_config",
    "load_config",
    "save_metadata",
    "load_metadata",
    "save_dataframe",
    "load_dataframe",
    "save_metrics",
    "load_metrics",
    "save_result",
    "load_result",
]
