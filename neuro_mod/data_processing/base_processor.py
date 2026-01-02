"""Base class for simulation data processors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import logging


class _BaseSimProcessor(ABC):
    """Base class for processing raw simulation outputs into structured data.

    Subclasses implement the processing pipeline for specific simulation types,
    transforming raw output files into processed data structures that can be
    saved to disk and later loaded for analysis.
    """

    def __init__(self) -> None:
        """Initialize the processor with a logger."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._processed_data: dict | None = None

    @property
    def processed_data(self) -> dict | None:
        """Return the processed data, or None if not yet processed."""
        return self._processed_data

    @abstractmethod
    def _load_raw_data(self) -> Any:
        """Load raw simulation output.

        Returns:
            Raw data in whatever format the simulation produces.
        """
        pass

    @abstractmethod
    def process(self) -> dict:
        """Process raw data into structured output.

        This is the main entry point for the processing pipeline.
        Subclasses should implement the full transformation from
        raw simulation output to a structured dictionary.

        Returns:
            Processed data dictionary.
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save processed data to disk.

        Args:
            path: Directory path where processed data should be saved.
        """
        pass

    @classmethod
    @abstractmethod
    def load_processed(cls, path: Path) -> dict:
        """Load previously processed data from disk.

        Args:
            path: Directory path containing saved processed data.

        Returns:
            The processed data dictionary.
        """
        pass
