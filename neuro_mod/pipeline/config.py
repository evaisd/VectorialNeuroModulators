"""Configuration and result classes for the experiment pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
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

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    verbose_memory: bool = False
    """If True, log memory usage after each repeat (useful for debugging memory issues)."""
    progress_callback: Any = None  # Callable[[int, int, str], None] | None

    # Time evolution sampling
    time_evolution_dt: float | None = None
    time_evolution_num_steps: int | None = None

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

        # Repeated and sweep+repeated modes require save_dir for disk-based processing
        if self.mode in (ExecutionMode.REPEATED, ExecutionMode.SWEEP_REPEATED):
            if self.save_dir is None:
                raise ValueError(
                    f"{self.mode.name} mode requires save_dir to be set for disk-based processing"
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
            "log_level": self.log_level,
            "log_to_file": self.log_to_file,
            "verbose_memory": self.verbose_memory,
            "time_evolution_dt": self.time_evolution_dt,
            "time_evolution_num_steps": self.time_evolution_num_steps,
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
        # Remove deprecated fields (for backward compatibility with old configs)
        data.pop("streaming", None)
        data.pop("unified_processing", None)
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
        analyzers: Analyzer instances per run key (not serialized).
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
    analyzers: dict[str, Any] = field(default_factory=dict)

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

    def align_transition_matrices(
        self,
        *,
        labels: str = "identity",
        canonical_labels: list[Any] | None = None,
        include_aggregated: bool = False,
    ) -> tuple[list[Any], dict[str, pd.DataFrame]]:
        """Return transition matrices aligned to a canonical label set."""
        from neuro_mod.core.spiking_net.analysis import helpers as snn_helpers

        logger = logging.getLogger("PipelineResult")
        tpms: dict[str, pd.DataFrame] = {}

        for key, analyzer in self.analyzers.items():
            if not include_aggregated and key == "aggregated":
                continue
            if hasattr(analyzer, "transitions_matrix"):
                try:
                    tpm = analyzer.transitions_matrix(labels=labels)
                except TypeError:
                    tpm = analyzer.transitions_matrix()
                if isinstance(tpm, pd.DataFrame) and not tpm.empty:
                    tpms[key] = tpm
            elif hasattr(analyzer, "manipulation") and hasattr(analyzer, "list_manipulations"):
                if labels != "idx":
                    logger.warning(
                        "Analyzer for '%s' does not support label selection; using idx labels.",
                        key,
                    )
                if "transitions" in analyzer.list_manipulations():
                    tpm = analyzer.manipulation("transitions")
                    if isinstance(tpm, pd.DataFrame) and not tpm.empty:
                        tpms[key] = tpm

        if not tpms:
            for name, df in self.dataframes.items():
                if not name.endswith("_tpm"):
                    continue
                key = name[:-4]
                if not include_aggregated and key == "aggregated":
                    continue
                if isinstance(df, pd.DataFrame) and not df.empty:
                    tpms[key] = df
            if tpms and labels != "idx":
                logger.warning(
                    "No analyzers available; using stored TPM labels (likely idx)."
                )

        if not tpms:
            return [], {}

        if canonical_labels is None:
            canonical_labels = snn_helpers.build_canonical_labels_from_tpms(tpms.values())

        aligned = {
            key: snn_helpers.align_transition_matrix(tpm, canonical_labels)
            for key, tpm in tpms.items()
        }
        return list(canonical_labels), aligned


__all__ = [
    "ExecutionMode",
    "PipelineConfig",
    "PipelineResult",
]
