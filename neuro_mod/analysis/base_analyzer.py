"""Base class for simulation data analyzers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import logging

import pandas as pd


class _BaseAnalyzer(ABC):
    """Base class for analyzing processed simulation data.

    Subclasses implement analysis pipelines for specific simulation types,
    loading processed data and computing metrics, transforming to DataFrames,
    and providing various analysis utilities.
    """

    def __init__(self, processed_data: dict | Path) -> None:
        """Initialize the analyzer with processed data.

        Args:
            processed_data: Either a dictionary of processed data or a Path
                to a directory containing saved processed data.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        if isinstance(processed_data, (str, Path)):
            self._processed_data = self._load_from_path(Path(processed_data))
        else:
            self._processed_data = processed_data

    @property
    def processed_data(self) -> dict:
        """Return the processed data dictionary."""
        return self._processed_data

    @abstractmethod
    def _load_from_path(self, path: Path) -> dict:
        """Load processed data from a directory path.

        Args:
            path: Directory containing saved processed data.

        Returns:
            The processed data dictionary.
        """
        pass

    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """Convert processed data to a pandas DataFrame.

        Returns:
            DataFrame representation of the processed data.
        """
        pass

    @abstractmethod
    def get_summary_metrics(self) -> dict[str, Any]:
        """Extract summary metrics from the processed data.

        Returns:
            Dictionary of metric names to values.
        """
        pass
