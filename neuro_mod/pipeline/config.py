"""Configuration and result classes for the experiment pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import pandas as pd


class ExecutionMode(Enum):
    """Pipeline execution modes."""

    SINGLE = auto()
    """Single simulation run."""

    REPEATED = auto()
    """Multiple runs with seed management."""

    SWEEP = auto()
    """Parameter variations (single run per value)."""

    SWEEP_REPEATED = auto()
    """Sweep + repeats (factorial design)."""


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution.

    Attributes:
        mode: Execution mode (SINGLE, REPEATED, SWEEP, SWEEP_REPEATED).
        n_repeats: Number of repeated runs (for REPEATED/SWEEP_REPEATED modes).
        base_seed: Base seed for deterministic seed generation.
        seeds: Explicit seed list (overrides base_seed if provided).
        sweep_param: Parameter path(s) to sweep (e.g., "arousal.level" or ["tau_m", "tau_s"]).
        sweep_values: Values to sweep over.
        parallel: Enable parallel execution.
        max_workers: Maximum number of parallel workers.
        executor: Executor type ("thread" or "process").
        save_dir: Directory to save results.
        save_raw: Whether to save raw simulation outputs.
        save_processed: Whether to save processed data.
        save_analysis: Whether to save analysis results.
        save_plots: Whether to save plot files.
        run_processing: Whether to run the processing phase.
        run_analysis: Whether to run the analysis phase.
        run_plotting: Whether to run the plotting phase.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_to_file: Whether to write logs to file.
        progress_callback: Optional callback for progress updates.
    """

    # Execution mode
    mode: ExecutionMode = ExecutionMode.SINGLE

    # Repeat settings
    n_repeats: int = 1
    base_seed: int = 256
    seeds: list[int] | None = None

    # Sweep settings
    sweep_param: str | list[str] | None = None
    sweep_values: list[Any] | None = None

    # Parallelization
    parallel: bool = False
    max_workers: int | None = None
    executor: str = "thread"

    # I/O settings
    save_dir: Path | None = None
    save_raw: bool = False
    save_processed: bool = True
    save_analysis: bool = True
    save_plots: bool = True

    # Phase toggles
    run_processing: bool = True
    run_analysis: bool = True
    run_plotting: bool = True

    # Processing mode
    unified_processing: bool = True
    """If True, process all repeats together for consistent attractor IDs.

    For REPEATED mode: all repeats processed together.
    For SWEEP mode: each sweep value processed separately (no unification).
    For SWEEP_REPEATED mode: repeats within each sweep value processed together,
        but different sweep values processed separately (different dynamics).
    """

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    progress_callback: Any = None  # Callable[[int, int, str], None] | None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.mode in (ExecutionMode.SWEEP, ExecutionMode.SWEEP_REPEATED):
            if self.sweep_param is None or self.sweep_values is None:
                raise ValueError(
                    "sweep_param and sweep_values are required for SWEEP modes"
                )

        if self.save_dir is not None:
            self.save_dir = Path(self.save_dir)

        if self.executor not in ("thread", "process"):
            raise ValueError(f"executor must be 'thread' or 'process', got {self.executor!r}")

        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            raise ValueError(
                f"log_level must be DEBUG/INFO/WARNING/ERROR, got {self.log_level!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to a JSON-serializable dictionary."""
        return {
            "mode": self.mode.name,
            "n_repeats": self.n_repeats,
            "base_seed": self.base_seed,
            "seeds": self.seeds,
            "sweep_param": self.sweep_param,
            "sweep_values": self.sweep_values,
            "parallel": self.parallel,
            "max_workers": self.max_workers,
            "executor": self.executor,
            "save_dir": str(self.save_dir) if self.save_dir else None,
            "save_raw": self.save_raw,
            "save_processed": self.save_processed,
            "save_analysis": self.save_analysis,
            "save_plots": self.save_plots,
            "run_processing": self.run_processing,
            "run_analysis": self.run_analysis,
            "run_plotting": self.run_plotting,
            "unified_processing": self.unified_processing,
            "log_level": self.log_level,
            "log_to_file": self.log_to_file,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        """Create config from a dictionary."""
        data = data.copy()
        if "mode" in data and isinstance(data["mode"], str):
            data["mode"] = ExecutionMode[data["mode"]]
        if "save_dir" in data and data["save_dir"] is not None:
            data["save_dir"] = Path(data["save_dir"])
        # Remove non-serializable fields
        data.pop("progress_callback", None)
        return cls(**data)


@dataclass
class PipelineResult:
    """Container for pipeline execution results.

    Attributes:
        mode: Execution mode used.
        config: Pipeline configuration used.
        raw_outputs: Raw simulation outputs (if saved).
        processed_data: Processed data per run.
        dataframes: DataFrames per run, plus "aggregated" for combined data.
        metrics: Summary metrics per run, plus "aggregated" for combined metrics.
        figures: List of generated figure objects.
        seeds_used: Seeds used for each run (for reproducibility).
        sweep_metadata: Metadata about sweep parameters and values.
        timestamp: When the pipeline was executed.
        git_commit: Git commit hash (if available).
        duration_seconds: Total execution time.
    """

    mode: ExecutionMode
    config: PipelineConfig

    # Raw outputs (optional, can be large)
    raw_outputs: dict[str, Any] = field(default_factory=dict)

    # Processed data per run key
    processed_data: dict[str, Any] = field(default_factory=dict)

    # Analysis results
    dataframes: dict[str, pd.DataFrame] = field(default_factory=dict)
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Plotting artifacts
    figures: list[Any] = field(default_factory=list)

    # Reproducibility metadata
    seeds_used: list[int] = field(default_factory=list)
    sweep_metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    git_commit: str | None = None
    duration_seconds: float = 0.0

    def get_aggregated_dataframe(self) -> pd.DataFrame | None:
        """Get the aggregated DataFrame across all runs."""
        return self.dataframes.get("aggregated")

    def get_aggregated_metrics(self) -> dict[str, Any] | None:
        """Get the aggregated metrics across all runs."""
        return self.metrics.get("aggregated")

    def summary(self) -> dict[str, Any]:
        """Get a summary of the pipeline results."""
        return {
            "mode": self.mode.name,
            "n_runs": len([k for k in self.dataframes if k != "aggregated"]),
            "seeds_used": self.seeds_used,
            "sweep_metadata": self.sweep_metadata,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "duration_seconds": self.duration_seconds,
            "aggregated_metrics": self.get_aggregated_metrics(),
        }


__all__ = [
    "ExecutionMode",
    "PipelineConfig",
    "PipelineResult",
]
