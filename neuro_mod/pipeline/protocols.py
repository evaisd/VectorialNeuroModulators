"""Protocol definitions for the experiment pipeline.

Protocols define the contracts that pipeline components must satisfy.
Using Python's structural typing (Protocols), existing classes automatically
conform without modification if they have the required methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

import pandas as pd


# Type variables for generic protocols
TRawOutput = TypeVar("TRawOutput", covariant=True)
TProcessedData = TypeVar("TProcessedData", covariant=True)


@runtime_checkable
class Simulator(Protocol[TRawOutput]):
    """Protocol for simulation components.

    Any class implementing run() that returns simulation outputs
    conforms to this protocol. Examples: _Stager subclasses.
    """

    def run(self, *args: Any, **kwargs: Any) -> TRawOutput:
        """Execute simulation and return raw outputs."""
        ...


@runtime_checkable
class SimulatorFactory(Protocol[TRawOutput]):
    """Protocol for creating simulators with different seeds/configs.

    A factory is a callable that creates Simulator instances,
    typically parameterized by seed for reproducibility.
    """

    def __call__(self, seed: int, **kwargs: Any) -> Simulator[TRawOutput]:
        """Create a simulator instance with the given seed."""
        ...


@runtime_checkable
class Processor(Protocol[TRawOutput, TProcessedData]):
    """Protocol for data processing components.

    Transforms raw simulation outputs into structured data.
    Examples: _BaseSimProcessor subclasses.
    """

    @property
    def processed_data(self) -> TProcessedData | None:
        """Return the processed data, or None if not yet processed."""
        ...

    def process(self) -> TProcessedData:
        """Process raw data into structured output."""
        ...

    def save(self, path: Path) -> None:
        """Save processed data to disk."""
        ...

    @classmethod
    def load_processed(cls, path: Path) -> TProcessedData:
        """Load previously processed data."""
        ...


@runtime_checkable
class ProcessorFactory(Protocol[TRawOutput, TProcessedData]):
    """Protocol for creating processors.

    A factory is a callable that creates Processor instances
    from raw simulation output.
    """

    def __call__(
        self, raw_data: TRawOutput, **kwargs: Any
    ) -> Processor[TRawOutput, TProcessedData]:
        """Create a processor instance for the given raw data."""
        ...


@runtime_checkable
class Analyzer(Protocol[TProcessedData]):
    """Protocol for analysis components.

    Analyzes processed data and provides DataFrame conversion.
    Examples: _BaseAnalyzer subclasses.
    """

    @property
    def processed_data(self) -> TProcessedData:
        """Return the processed data dictionary."""
        ...

    def to_dataframe(self, *columns: str) -> pd.DataFrame:
        """Convert processed data to pandas DataFrame."""
        ...

    def get_summary_metrics(self) -> dict[str, Any]:
        """Extract summary metrics from processed data."""
        ...


@runtime_checkable
class AnalyzerFactory(Protocol[TProcessedData]):
    """Protocol for creating analyzers.

    A factory is a callable that creates Analyzer instances
    from processed data or a path to saved data.
    """

    def __call__(
        self, processed_data: TProcessedData | Path, **kwargs: Any
    ) -> Analyzer[TProcessedData]:
        """Create an analyzer instance."""
        ...


@runtime_checkable
class Plotter(Protocol):
    """Protocol for plotting components.

    Generates visualizations from DataFrames and metrics.
    """

    def plot(
        self,
        data: pd.DataFrame,
        metrics: dict[str, Any] | None = None,
        save_dir: Path | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Generate plots from DataFrame and metrics.

        Args:
            data: DataFrame to visualize.
            metrics: Optional summary metrics dictionary.
            save_dir: Optional directory to save plot files.
            **kwargs: Additional plotting arguments.

        Returns:
            List of figure objects (matplotlib Figure or similar).
        """
        ...


__all__ = [
    "Simulator",
    "SimulatorFactory",
    "Processor",
    "ProcessorFactory",
    "Analyzer",
    "AnalyzerFactory",
    "Plotter",
    "TRawOutput",
    "TProcessedData",
]
