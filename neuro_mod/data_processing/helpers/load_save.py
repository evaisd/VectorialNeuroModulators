"""Generic utilities for saving and loading processed simulation data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_processed_data(
        data: dict,
        path: Path,
        config: dict | None = None,
        *,
        data_filename: str = "processed_data.npy",
        config_filename: str = "config.json",
) -> None:
    """Save processed data and configuration to disk.

    Args:
        data: The processed data dictionary to save.
        path: Directory where files will be saved.
        config: Optional configuration metadata.
        data_filename: Name for the data file.
        config_filename: Name for the config file.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    data_path = path / data_filename
    np.save(data_path, data, allow_pickle=True)

    if config is not None:
        config_path = path / config_filename
        config["files"] = config.get("files", {})
        config["files"]["data"] = data_filename
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True))


def load_processed_data(
        path: Path,
        *,
        config_filename: str = "config.json",
) -> tuple[dict, dict]:
    """Load processed data and configuration from disk.

    Args:
        path: Directory containing saved files.
        config_filename: Name of the config file.

    Returns:
        Tuple of (data, config) dictionaries.
    """
    path = Path(path)
    config_path = path / config_filename

    if config_path.exists():
        config = json.loads(config_path.read_text())
    else:
        config = {}

    files = config.get("files", {})
    data_filename = files.get("data", "processed_data.npy")
    data_path = path / data_filename

    data = np.load(data_path, allow_pickle=True).item()
    return data, config


def save_array(
        array: np.ndarray,
        path: Path,
        filename: str,
) -> Path:
    """Save a numpy array to disk.

    Args:
        array: Array to save.
        path: Directory where file will be saved.
        filename: Name for the file.

    Returns:
        Full path to the saved file.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / filename
    np.save(file_path, array)
    return file_path


def load_array(
        path: Path,
        filename: str,
) -> np.ndarray:
    """Load a numpy array from disk.

    Args:
        path: Directory containing the file.
        filename: Name of the file.

    Returns:
        The loaded numpy array.
    """
    path = Path(path)
    file_path = path / filename
    return np.load(file_path)


def save_config(
        config: dict[str, Any],
        path: Path,
        filename: str = "config.json",
) -> Path:
    """Save configuration to a JSON file.

    Args:
        config: Configuration dictionary.
        path: Directory where file will be saved.
        filename: Name for the file.

    Returns:
        Full path to the saved file.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / filename
    file_path.write_text(json.dumps(config, indent=2, sort_keys=True))
    return file_path


def load_config(
        path: Path,
        filename: str = "config.json",
) -> dict[str, Any]:
    """Load configuration from a JSON file.

    Args:
        path: Directory containing the file.
        filename: Name of the file.

    Returns:
        The configuration dictionary.
    """
    path = Path(path)
    file_path = path / filename
    return json.loads(file_path.read_text())
