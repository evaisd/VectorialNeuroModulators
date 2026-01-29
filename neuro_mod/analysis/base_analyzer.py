"""Analyzer base classes and registries for pipeline analysis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar

import pandas as pd

from neuro_mod.analysis.helpers import *


class BaseAnalyzer(ABC):
    """Base class for analyzers with manipulation/metric registry."""

    _manipulations: ClassVar[dict[str, Callable[..., pd.DataFrame]]] = {}
    _readers: ClassVar[dict[str, Callable[..., pd.DataFrame]]] = {}
    _metrics: ClassVar[dict[str, tuple[str | None, Callable[..., MetricResult]]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._manipulations = dict(getattr(cls, "_manipulations", {}))
        cls._metrics = dict(getattr(cls, "_metrics", {}))
        cls._readers = dict(getattr(cls, "_readers", {}))
        for _, value in cls.__dict__.items():
            if callable(value) and getattr(value, "_is_manipulation", False):
                name = getattr(value, "_manipulation_name", None) or value.__name__
                cls._manipulations[name] = value
            if callable(value) and getattr(value, "_is_reader", False):
                name = getattr(value, "_reader_name", None) or value.__name__
                cls._readers[name] = value
            if callable(value) and getattr(value, "_is_metric", False):
                name = getattr(value, "_metric_name", None) or value.__name__
                expects = getattr(value, "_metric_expects", None)
                cls._metrics[name] = (expects, value)

    @property
    @abstractmethod
    def df(self) -> pd.DataFrame:
        """Base occurrence-level DataFrame."""
        ...

    def manipulation(self, name: str, **kwargs: Any) -> pd.DataFrame:
        """Call a registered manipulation by name."""
        if name not in self._manipulations:
            available = ", ".join(sorted(self._manipulations))
            raise KeyError(f"Unknown manipulation '{name}'. Available: {available}")
        func = self._manipulations[name]
        return func(self, **kwargs)

    def reader(self, name: str, **kwargs: Any) -> pd.DataFrame:
        """Call a registered reader by name."""
        if name not in self._readers:
            available = ", ".join(sorted(self._readers))
            raise KeyError(f"Unknown reader '{name}'. Available: {available}")
        func = self._readers[name]
        return func(self, **kwargs)

    def metric(
        self,
        name: str,
        df: pd.DataFrame | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> MetricResult:
        """Call a registered metric by name."""
        if name not in self._metrics:
            available = ", ".join(sorted(self._metrics))
            raise KeyError(f"Unknown metric '{name}'. Available: {available}")
        expects, func = self._metrics[name]
        if df is None:
            df = self.manipulation(expects) if expects else self.df
        return func(self, df, *args, **kwargs)

    @classmethod
    def list_manipulations(cls) -> list[str]:
        """List registered manipulation names."""
        return sorted(cls._manipulations)

    @classmethod
    def list_metrics(cls) -> list[str]:
        """List registered metric names."""
        return sorted(cls._metrics)

    @classmethod
    def list_readers(cls) -> list[str]:
        """List registered reader names."""
        return sorted(cls._readers)

    @abstractmethod
    def get_summary_metrics(self) -> dict[str, Any]:
        """Return a dictionary of summary metrics."""
        ...


__all__ = [
    "BaseAnalyzer",
    "MetricResult",
    "manipulation",
    "metric",
    "reader",
]
