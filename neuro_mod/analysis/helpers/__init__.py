
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass
class MetricResult:
    """Result container for metrics plotted from analyzers."""

    x: np.ndarray
    y: np.ndarray | None
    labels: np.ndarray | None = None
    metadata: dict[str, Any] | None = None


def manipulation(name: str | None = None):
    """Decorator to register a method as a manipulation (returns DataFrame)."""

    def decorator(func: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
        setattr(func, "_is_manipulation", True)
        setattr(func, "_manipulation_name", name)
        return func

    return decorator


def reader(name: str | None = None):
    """Decorator to register a method as a reader (returns DataFrame)."""

    def decorator(func: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
        setattr(func, "_is_reader", True)
        setattr(func, "_reader_name", name)
        return func

    return decorator


def metric(name: str | None = None, *, expects: str | None = None):
    """Decorator to register a method as a metric (returns MetricResult)."""

    def decorator(func: Callable[..., MetricResult]) -> Callable[..., MetricResult]:
        setattr(func, "_is_metric", True)
        setattr(func, "_metric_name", name)
        setattr(func, "_metric_expects", expects)
        return func

    return decorator


__all__ = [
    "MetricResult",
    "manipulation",
    "metric",
    "reader",
]
