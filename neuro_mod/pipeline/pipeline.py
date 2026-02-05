"""Main Pipeline class for orchestrating experiment workflows."""

from __future__ import annotations

import logging
import pickle
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
import pandas as pd

from neuro_mod.pipeline.config import ExecutionMode, PipelineConfig, PipelineResult
from neuro_mod.pipeline.io import get_git_commit, get_timestamp, save_result
from neuro_mod.pipeline.plotting import SeabornPlotter, SpecPlotter
from neuro_mod.pipeline.protocols import (
    AnalyzerFactory,
    BatchProcessorFactory,
    Plotter,
    ProcessorFactory,
    SimulatorFactory,
)


def _persist_raw_output_to_dir(raw: Any, save_dir: Path, key: str, compress: bool) -> Any:
    """Persist raw outputs to disk and inject file paths."""
    data_dir = save_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(raw, dict):
        if "spikes_path" not in raw:
            spikes = raw.get("spikes")
            if spikes is not None:
                clusters = raw.get("clusters")
                spikes_path = data_dir / f"{key}_spikes.npz"
                if clusters is not None:
                    if compress:
                        np.savez_compressed(spikes_path, spikes=spikes, clusters=clusters)
                    else:
                        np.savez(spikes_path, spikes=spikes, clusters=clusters)
                else:
                    spikes_path = spikes_path.with_suffix(".npy")
                    np.save(spikes_path, spikes)
                updated = dict(raw)
                updated["spikes_path"] = str(spikes_path)
                updated.setdefault("clusters_path", None)
                return updated

        raw_path = data_dir / f"{key}_raw.npy"
        np.save(raw_path, raw, allow_pickle=True)
        updated = dict(raw)
        updated.setdefault("raw_path", str(raw_path))
        return updated

    raw_path = data_dir / f"{key}_raw.pkl"
    with open(raw_path, "wb") as handle:
        pickle.dump(raw, handle)
    return raw


def _minimize_raw_output(raw: Any) -> Any:
    """Remove large in-memory payloads when disk-backed paths exist."""
    if isinstance(raw, dict) and ("spikes_path" in raw or "raw_path" in raw):
        minimized = {}
        for key, value in raw.items():
            if key in ("spikes_path", "raw_path"):
                minimized.setdefault(key, value)
        return minimized
    return raw


def _simulate_and_persist_worker(
    simulator_factory: SimulatorFactory[Any],
    save_dir: Path,
    seed: int,
    key: str,
    log_timings: bool,
    compress: bool,
) -> Any:
    t0 = time.time()
    simulator = simulator_factory(seed)
    raw = simulator.run()
    t1 = time.time()
    raw = _persist_raw_output_to_dir(raw, save_dir, key, compress)
    t2 = time.time()
    raw = _minimize_raw_output(raw)
    t3 = time.time()
    if log_timings:
        return raw, {
            "simulate_seconds": t1 - t0,
            "persist_seconds": t2 - t1,
            "minimize_seconds": t3 - t2,
            "total_seconds": t3 - t0,
        }
    return raw


def _simulate_with_timing(
    simulator_factory: SimulatorFactory[Any],
    seed: int,
    log_timings: bool,
) -> Any:
    t0 = time.time()
    simulator = simulator_factory(seed)
    raw = simulator.run()
    t1 = time.time()
    if log_timings:
        return raw, {"simulate_seconds": t1 - t0}
    return raw


def _simulate_with_param_and_persist_worker(
    simulator_factory: SimulatorFactory[Any],
    save_dir: Path,
    seed: int,
    key: str,
    log_timings: bool,
    compress: bool,
    sweep_param: str | list[str],
    sweep_value: Any,
    sweep_idx: int | None,
) -> Any:
    t0 = time.time()
    simulator = simulator_factory(
        seed,
        sweep_param=sweep_param,
        sweep_value=sweep_value,
        sweep_idx=sweep_idx,
    )
    raw = simulator.run()
    t1 = time.time()
    raw = _persist_raw_output_to_dir(raw, save_dir, key, compress)
    t2 = time.time()
    raw = _minimize_raw_output(raw)
    t3 = time.time()
    if log_timings:
        return raw, {
            "simulate_seconds": t1 - t0,
            "persist_seconds": t2 - t1,
            "minimize_seconds": t3 - t2,
            "total_seconds": t3 - t0,
        }
    return raw


def _simulate_with_param_timing(
    simulator_factory: SimulatorFactory[Any],
    seed: int,
    log_timings: bool,
    sweep_param: str | list[str],
    sweep_value: Any,
    sweep_idx: int | None,
) -> Any:
    t0 = time.time()
    simulator = simulator_factory(
        seed,
        sweep_param=sweep_param,
        sweep_value=sweep_value,
        sweep_idx=sweep_idx,
    )
    raw = simulator.run()
    t1 = time.time()
    if log_timings:
        return raw, {"simulate_seconds": t1 - t0}
    return raw


TRaw = TypeVar("TRaw")
TProcessed = TypeVar("TProcessed")


class Pipeline(Generic[TRaw, TProcessed]):
    """Generic experiment pipeline orchestrating Simulation -> Processing -> Analysis -> Plotting.

    The Pipeline class handles all execution modes (single, repeated, sweep, sweep+repeated)
    internally, providing a unified interface for experiment execution.

    Type Parameters:
        TRaw: Type of raw simulation output.
        TProcessed: Type of processed data.

    Example:
        >>> pipeline = Pipeline(
        ...     simulator_factory=lambda seed: StageSNNSimulation(config, random_seed=seed),
        ...     processor_factory=lambda raw: SNNProcessor(raw["spikes"], raw["clusters"]),
        ...     analyzer_factory=SNNAnalyzer,
        ... )
        >>> result = pipeline.run(PipelineConfig(mode=ExecutionMode.SINGLE))
    """

    def __init__(
        self,
        simulator_factory: SimulatorFactory[TRaw],
        processor_factory: ProcessorFactory[TRaw, TProcessed] | None = None,
        batch_processor_factory: BatchProcessorFactory[TRaw, TProcessed] | None = None,
        analyzer_factory: AnalyzerFactory[TProcessed] | None = None,
        plotter: Plotter | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            simulator_factory: Factory creating Simulator instances (called with seed).
            processor_factory: Optional factory creating Processor instances (per-run).
            batch_processor_factory: Factory for unified batch processing.
                Required for REPEATED and SWEEP_REPEATED modes. Processes all repeats
                together for consistent attractor IDs across runs.
            analyzer_factory: Optional factory creating Analyzer instances.
            plotter: Optional Plotter instance for visualization.
            logger: Optional logger instance.
        """
        self.simulator_factory = simulator_factory
        self.processor_factory = processor_factory
        self.batch_processor_factory = batch_processor_factory
        self.analyzer_factory = analyzer_factory
        self.plotter = plotter
        self._logger = logger

    @property
    def logger(self) -> logging.Logger:
        """Get the pipeline logger."""
        if self._logger is None:
            self._logger = logging.getLogger("Pipeline")
        return self._logger

    def run(self, config: PipelineConfig) -> PipelineResult:
        """Execute the pipeline with the given configuration.

        Args:
            config: Pipeline configuration.

        Returns:
            PipelineResult containing all outputs.
        """
        start_time = time.time()

        # Setup logging
        self._setup_logging(config)

        self.logger.info(f"Starting pipeline in {config.mode.name} mode")

        # Initialize result container
        result = PipelineResult(
            mode=config.mode,
            config=config,
            timestamp=get_timestamp(),
            git_commit=get_git_commit(),
        )

        # Generate seeds
        seeds = self._generate_seeds(config)
        result.seeds_used = seeds
        self.logger.info(f"Using {len(seeds)} seed(s): {seeds[:5]}{'...' if len(seeds) > 5 else ''}")

        # Route to appropriate execution method
        if config.mode == ExecutionMode.SINGLE:
            self._run_single(config, result, seeds[0])
        elif config.mode == ExecutionMode.REPEATED:
            self._run_repeated(config, result, seeds)
        elif config.mode == ExecutionMode.SWEEP:
            self._run_sweep(config, result, seeds[0])
        elif config.mode == ExecutionMode.SWEEP_REPEATED:
            self._run_sweep_repeated(config, result, seeds)

        # Record duration
        result.duration_seconds = time.time() - start_time
        self.logger.info(f"Pipeline complete in {result.duration_seconds:.2f}s")

        self._align_transition_matrices(result)

        # Save results if save_dir specified
        if config.save_dir:
            self._save_results(config, result)

        return result

    def process_existing(
        self,
        config: PipelineConfig,
        raw_refs: list[TRaw],
        metadata: list[dict[str, Any]],
        *,
        key: str = "aggregated",
        sweep_value: Any = None,
    ) -> PipelineResult:
        """Process existing raw outputs without running simulations."""
        start_time = time.time()
        self._setup_logging(config)
        self.logger.info(f"Processing existing data for '{key}'")

        result = PipelineResult(
            mode=config.mode,
            config=config,
            timestamp=get_timestamp(),
            git_commit=get_git_commit(),
        )
        result.seeds_used = [m.get("seed") for m in metadata]

        analyzer = self._process_and_analyze_unified(
            config,
            result,
            raw_refs,
            metadata,
            key=key,
            sweep_value=sweep_value,
        )

        if config.run_plotting:
            self._generate_plots(config, result, key, analyzer)

        result.duration_seconds = time.time() - start_time
        self.logger.info(f"Processing complete in {result.duration_seconds:.2f}s")

        self._align_transition_matrices(result)

        if config.save_dir:
            self._save_results(config, result)

        return result

    def _log_memory_usage(self, config: PipelineConfig, context: str = "") -> None:
        """Log current memory usage if verbose_memory is enabled."""
        if not config.verbose_memory:
            return
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            rss_mb = mem_info.rss / (1024 * 1024)
            vms_mb = mem_info.vms / (1024 * 1024)
            self.logger.info(f"[Memory] {context}: RSS={rss_mb:.1f}MB, VMS={vms_mb:.1f}MB")
        except ImportError:
            # Fallback if psutil not available
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            max_rss_mb = usage.ru_maxrss / 1024  # macOS returns bytes, Linux returns KB
            import sys
            if sys.platform == "darwin":
                max_rss_mb = usage.ru_maxrss / (1024 * 1024)
            self.logger.info(f"[Memory] {context}: MaxRSS={max_rss_mb:.1f}MB")

    def _get_analysis_dir(self, config: PipelineConfig) -> Path:
        """Get the analysis directory."""
        return config.save_dir / "analysis"

    def _save_dataframe(
        self,
        df: pd.DataFrame,
        config: PipelineConfig,
        name: str = "dataframe",
    ) -> Path:
        """Save a DataFrame to disk.

        Args:
            df: DataFrame to save.
            config: Pipeline configuration.
            name: Base name for the file.

        Returns:
            Path to the saved file.
        """
        analysis_dir = self._get_analysis_dir(config)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        path = analysis_dir / f"{name}.pkl"
        df.to_pickle(path)
        self.logger.debug(f"Saved {name} to {path}")
        return path


    def _save_metrics(
        self,
        metrics: dict[str, Any],
        config: PipelineConfig,
    ) -> Path:
        """Save metrics to disk."""
        import json
        analysis_dir = self._get_analysis_dir(config)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        path = analysis_dir / "metrics.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        return path

    def _save_analysis_artifacts(
        self,
        config: PipelineConfig,
        result: PipelineResult,
        key: str,
    ) -> None:
        """Persist analysis outputs for a run key."""
        if not (config.save_analysis and config.save_dir):
            return
        self._save_dataframe(result.dataframes[key], config, "dataframe")
        self._save_metrics(result.metrics[key], config)
        per_attr_key = f"{key}_per_attractor"
        time_key = f"{key}_time"
        if per_attr_key in result.dataframes:
            self._save_dataframe(result.dataframes[per_attr_key], config, "per_attractor")
        if time_key in result.dataframes:
            self._save_dataframe(result.dataframes[time_key], config, "time_evolution")
        tpm_key = f"{key}_tpm"
        if tpm_key in result.dataframes:
            self._save_dataframe(result.dataframes[tpm_key], config, "tpm")

    def _setup_logging(self, config: PipelineConfig) -> None:
        """Configure logging based on config."""
        level = getattr(logging, config.log_level, logging.INFO)
        self.logger.setLevel(level)

        # Add console handler if not present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(level)
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Add file handler if requested
        if config.log_to_file and config.save_dir:
            log_dir = config.save_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / "pipeline.log")
            file_handler.setLevel(logging.DEBUG)  # Always debug level for file
            file_handler.setFormatter(logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            self.logger.addHandler(file_handler)

    def _generate_seeds(self, config: PipelineConfig) -> list[int]:
        """Generate seeds for execution."""
        if config.seeds is not None:
            return list(config.seeds)

        rng = np.random.default_rng(config.base_seed)
        n_seeds = config.n_repeats if config.mode in (
            ExecutionMode.REPEATED,
            ExecutionMode.SWEEP_REPEATED,
        ) else 1
        return rng.integers(0, 2**31, size=n_seeds).tolist()

    def _run_single(
        self,
        config: PipelineConfig,
        result: PipelineResult,
        seed: int,
    ) -> None:
        """Execute single simulation run."""
        self.logger.info(f"Running single simulation (seed={seed})")
        raw = self._simulate(seed, config)
        raw = self._persist_raw_output(raw, config, "single")
        analyzer = self._process_and_analyze(config, result, raw, key="single")

        if config.run_plotting:
            self._generate_plots(config, result, "single", analyzer)

    def _run_repeated(
        self,
        config: PipelineConfig,
        result: PipelineResult,
        seeds: list[int],
    ) -> None:
        """Execute repeated runs with unified processing.

        For repeated runs, batch_processor_factory is required to ensure
        consistent attractor identities across all repeats.

        The flow is:
        1. Run each simulation sequentially (or in parallel)
        2. Save raw data to disk immediately after each simulation
        3. After ALL simulations complete, batch processor loads data from disk
           and generates a single unified processed output
        """
        import gc

        n_total = len(seeds)

        if not self._supports_batch_processing():
            raise ValueError(
                "Repeated mode requires batch_processor_factory for unified processing. "
                "Set batch_processor_factory when creating the Pipeline."
            )

        self._log_memory_usage(config, "Start")

        # Phase 1: Run all simulations, saving raw data to disk immediately
        # Only keep minimal metadata (file paths) in memory
        self.logger.info(f"Running {n_total} simulations")
        raw_refs: list[TRaw] = []
        metadata: list[dict[str, Any]] = []

        if config.parallel and n_total > 1:
            raw_refs, metadata = self._simulate_repeated_parallel(config, seeds)
        else:
            for i, seed in enumerate(seeds):
                self.logger.info(f"Simulating repeat {i + 1}/{n_total} (seed={seed})")
                self._report_progress(config, i, n_total, f"simulate {i + 1}/{n_total}")

                t0 = time.time()
                raw = self._simulate(seed, config)
                t1 = time.time()
                self._log_memory_usage(config, f"After simulate repeat {i + 1}")

                # Save to disk and get back reference with file paths
                raw = self._persist_raw_output(raw, config, f"repeat_{i}")
                t2 = time.time()
                # Remove large in-memory data, keep only paths
                raw = self._minimize_raw_output(raw)
                t3 = time.time()
                raw_refs.append(raw)
                metadata.append({"seed": seed, "repeat_idx": i})

                # Force garbage collection after each simulation
                gc.collect()
                self._log_memory_usage(config, f"After GC repeat {i + 1}")
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        "Repeat %s timings: simulate=%.2fs persist=%.2fs minimize=%.2fs total=%.2fs",
                        i,
                        t1 - t0,
                        t2 - t1,
                        t3 - t2,
                        t3 - t0,
                    )

        # Phase 2: Unified batch processing (processor loads from disk)
        self.logger.info("Starting unified batch processing (loading from disk)")
        self._log_memory_usage(config, "Before unified processing")
        analyzer = self._process_and_analyze_unified(
            config, result, raw_refs, metadata, key="aggregated"
        )
        self._log_memory_usage(config, "After unified processing")

        if config.run_plotting:
            self._generate_plots(config, result, "aggregated", analyzer)

    def _simulate_repeated_parallel(
        self,
        config: PipelineConfig,
        seeds: list[int],
    ) -> tuple[list[TRaw], list[dict[str, Any]]]:
        """Run simulations in parallel, returning outputs and metadata.

        Returns:
            Tuple of (raw_outputs, metadata) lists, ordered by repeat index.
        """
        use_process = config.executor == "process"
        ExecutorClass = ProcessPoolExecutor if use_process else ThreadPoolExecutor
        n_total = len(seeds)

        self.logger.info(f"Running {n_total} repeats in parallel ({config.executor} executor)")
        log_timings = self.logger.isEnabledFor(logging.DEBUG)

        # Pre-allocate lists to maintain order
        raw_outputs: list[TRaw | None] = [None] * n_total
        metadata: list[dict[str, Any]] = [{}] * n_total

        with ExecutorClass(max_workers=config.max_workers) as executor:
            if use_process and config.persist_raw_in_worker:
                futures = {
                    executor.submit(
                        _simulate_and_persist_worker,
                        self.simulator_factory,
                        config.save_dir,
                        seed,
                        f"repeat_{i}",
                        log_timings,
                        config.save_compressed,
                    ): (i, seed)
                    for i, seed in enumerate(seeds)
                }
            else:
                futures = {
                    executor.submit(
                        _simulate_with_timing,
                        self.simulator_factory,
                        seed,
                        log_timings,
                    ): (i, seed)
                    for i, seed in enumerate(seeds)
                }

            for future in as_completed(futures):
                i, seed = futures[future]
                try:
                    raw = future.result()
                    timings = None
                    if log_timings:
                        raw, timings = raw
                    if not (use_process and config.persist_raw_in_worker):
                        t0 = time.time()
                        raw = self._persist_raw_output(raw, config, f"repeat_{i}")
                        t1 = time.time()
                        raw = self._minimize_raw_output(raw)
                        t2 = time.time()
                        if log_timings:
                            persist_timing = {
                                "persist_seconds": t1 - t0,
                                "minimize_seconds": t2 - t1,
                                "total_seconds": t2 - t0,
                            }
                            if timings:
                                timings.update(persist_timing)
                            else:
                                timings = persist_timing
                    raw_outputs[i] = raw
                    metadata[i] = {"seed": seed, "repeat_idx": i}
                    self.logger.debug(f"Repeat {i} complete (seed={seed})")
                    self._report_progress(config, i + 1, n_total, f"repeat {i + 1} done")
                    if timings:
                        self.logger.debug(
                            "Repeat %s timings: %s",
                            i,
                            ", ".join(f"{k}={v:.2f}s" for k, v in timings.items()),
                        )
                except Exception as exc:
                    self.logger.error(f"Repeat {i} failed: {exc}")
                    raise

        return raw_outputs, metadata  # type: ignore[return-value]

    def _simulate_sweep_repeated_parallel(
        self,
        config: PipelineConfig,
        seeds: list[int],
    ) -> tuple[dict[int, list[TRaw]], dict[int, list[dict[str, Any]]]]:
        """Run sweep + repeated simulations in parallel.

        Returns:
            Tuple of (raw_refs_by_sweep, meta_by_sweep) dicts.
        """
        assert config.sweep_values is not None
        assert config.sweep_param is not None

        use_process = config.executor == "process"
        ExecutorClass = ProcessPoolExecutor if use_process else ThreadPoolExecutor

        n_values = len(config.sweep_values)
        n_repeats = len(seeds)
        n_total = n_values * n_repeats

        self.logger.info(
            "Running %s sweep×repeat simulations in parallel (%s executor)",
            n_total,
            config.executor,
        )
        log_timings = self.logger.isEnabledFor(logging.DEBUG)

        raw_refs_by_sweep: dict[int, list[TRaw | None]] = {
            i: [None] * n_repeats for i in range(n_values)
        }
        meta_by_sweep: dict[int, list[dict[str, Any] | None]] = {
            i: [None] * n_repeats for i in range(n_values)
        }

        with ExecutorClass(max_workers=config.max_workers) as executor:
            futures: dict[Any, tuple[int, int, int, Any]] = {}
            for i, value in enumerate(config.sweep_values):
                for j, seed in enumerate(seeds):
                    key = f"sweep_{i}_repeat_{j}"
                    if use_process and config.persist_raw_in_worker:
                        future = executor.submit(
                            _simulate_with_param_and_persist_worker,
                            self.simulator_factory,
                            config.save_dir,
                            seed,
                            key,
                            log_timings,
                            config.save_compressed,
                            config.sweep_param,
                            value,
                            i,
                        )
                    else:
                        future = executor.submit(
                            _simulate_with_param_timing,
                            self.simulator_factory,
                            seed,
                            log_timings,
                            config.sweep_param,
                            value,
                            i,
                        )
                    futures[future] = (i, j, seed, value)

            completed = 0
            for future in as_completed(futures):
                i, j, seed, value = futures[future]
                try:
                    raw = future.result()
                    timings = None
                    if log_timings:
                        raw, timings = raw
                    if not (use_process and config.persist_raw_in_worker):
                        t0 = time.time()
                        raw = self._persist_raw_output(raw, config, f"sweep_{i}_repeat_{j}")
                        t1 = time.time()
                        raw = self._minimize_raw_output(raw)
                        t2 = time.time()
                        if log_timings:
                            persist_timing = {
                                "persist_seconds": t1 - t0,
                                "minimize_seconds": t2 - t1,
                                "total_seconds": t2 - t0,
                            }
                            if timings:
                                timings.update(persist_timing)
                            else:
                                timings = persist_timing
                    raw_refs_by_sweep[i][j] = raw
                    meta_by_sweep[i][j] = {
                        "seed": seed,
                        "repeat_idx": j,
                        "sweep_value": value,
                        "sweep_idx": i,
                    }
                    completed += 1
                    self.logger.debug(
                        "Sweep %s repeat %s complete (seed=%s)",
                        i,
                        j,
                        seed,
                    )
                    self._report_progress(
                        config,
                        completed,
                        n_total,
                        f"sweep {i + 1}, repeat {j + 1} done",
                    )
                    if timings:
                        self.logger.debug(
                            "Sweep %s repeat %s timings: %s",
                            i,
                            j,
                            ", ".join(f"{k}={v:.2f}s" for k, v in timings.items()),
                        )
                except Exception as exc:
                    self.logger.error("Sweep %s repeat %s failed: %s", i, j, exc)
                    raise

        raw_refs: dict[int, list[TRaw]] = {}
        meta_refs: dict[int, list[dict[str, Any]]] = {}
        for i in range(n_values):
            missing = [idx for idx, raw in enumerate(raw_refs_by_sweep[i]) if raw is None]
            if missing:
                raise RuntimeError(f"Missing sweep {i} outputs for repeats: {missing}")
            raw_refs[i] = [raw for raw in raw_refs_by_sweep[i] if raw is not None]
            meta_refs[i] = [meta for meta in meta_by_sweep[i] if meta is not None]

        return raw_refs, meta_refs

    def _run_sweep(
        self,
        config: PipelineConfig,
        result: PipelineResult,
        seed: int,
    ) -> None:
        """Execute parameter sweep."""
        assert config.sweep_param is not None
        assert config.sweep_values is not None

        result.sweep_metadata = {
            "param": config.sweep_param,
            "values": config.sweep_values,
        }

        n_values = len(config.sweep_values)
        for i, value in enumerate(config.sweep_values):
            self.logger.info(f"Sweep step {i + 1}/{n_values}: {config.sweep_param}={value}")
            self._report_progress(config, i, n_values, f"sweep {i + 1}/{n_values}")

            raw = self._simulate_with_param(seed, config, config.sweep_param, value, sweep_idx=i)
            raw = self._persist_raw_output(raw, config, f"sweep_{i}")
            raw = self._minimize_raw_output(raw)
            analyzer = self._process_and_analyze(config, result, raw, key=f"sweep_{i}")
            if config.run_plotting:
                self._generate_plots(config, result, f"sweep_{i}", analyzer)

        self._aggregate_sweep_results(config, result)
        if config.run_plotting:
            self._generate_plots(config, result, "aggregated", None)

    def _run_sweep_repeated(
        self,
        config: PipelineConfig,
        result: PipelineResult,
        seeds: list[int],
    ) -> None:
        """Execute sweep with repeats at each parameter value.

        Hierarchy: sweep_param → attractors
        - Different sweep values produce different dynamics (not comparable)
        - Repeats within each sweep value are unified for consistent attractor IDs

        The flow is:
        1. Run all simulations, saving raw data to disk after each
        2. After ALL simulations complete, batch processor loads data per sweep value
           and generates unified processed output for each sweep value
        """
        import gc

        assert config.sweep_param is not None
        assert config.sweep_values is not None

        if not self._supports_batch_processing():
            raise ValueError(
                "Sweep+repeated mode requires batch_processor_factory for unified processing. "
                "Set batch_processor_factory when creating the Pipeline."
            )

        result.sweep_metadata = {
            "param": config.sweep_param,
            "values": config.sweep_values,
            "n_repeats": len(seeds),
        }

        n_values = len(config.sweep_values)
        n_repeats = len(seeds)
        n_total = n_values * n_repeats

        self._log_memory_usage(config, "Start")

        # Phase 1: Run all simulations, saving raw data to disk immediately
        # Group references by sweep value for later unified processing
        if config.parallel and n_total > 1:
            raw_refs_by_sweep, meta_by_sweep = self._simulate_sweep_repeated_parallel(
                config, seeds
            )
        else:
            raw_refs_by_sweep: dict[int, list[TRaw]] = defaultdict(list)
            meta_by_sweep: dict[int, list[dict[str, Any]]] = defaultdict(list)
            step = 0

            for i, value in enumerate(config.sweep_values):
                for j, seed in enumerate(seeds):
                    self.logger.info(
                        f"Sweep {i + 1}/{n_values}, repeat {j + 1}/{n_repeats} "
                        f"({config.sweep_param}={value}, seed={seed})"
                    )
                    self._report_progress(config, step, n_total, f"sweep {i + 1}, repeat {j + 1}")

                    raw = self._simulate_with_param(
                        seed,
                        config,
                        config.sweep_param,
                        value,
                        sweep_idx=i,
                    )
                    self._log_memory_usage(config, f"After simulate sweep {i + 1}, repeat {j + 1}")

                    # Save to disk and get back reference with file paths
                    raw = self._persist_raw_output(raw, config, f"sweep_{i}_repeat_{j}")
                    raw = self._minimize_raw_output(raw)
                    raw_refs_by_sweep[i].append(raw)
                    meta_by_sweep[i].append({
                        "seed": seed,
                        "repeat_idx": j,
                        "sweep_value": value,
                        "sweep_idx": i,
                    })
                    step += 1

                    # Force garbage collection after each simulation
                    gc.collect()
                    self._log_memory_usage(config, f"After GC sweep {i + 1}, repeat {j + 1}")

        # Phase 2: Unified processing PER SWEEP VALUE (processor loads from disk)
        self.logger.info("Starting unified batch processing per sweep value")
        for i, value in enumerate(config.sweep_values):
            self.logger.info(f"Processing sweep {i + 1}/{n_values} ({config.sweep_param}={value})")
            self._log_memory_usage(config, f"Before processing sweep {i + 1}")
            analyzer = self._process_and_analyze_unified(
                config, result,
                raw_refs_by_sweep[i],
                meta_by_sweep[i],
                key=f"sweep_{i}",
                sweep_value=value,
            )
            self._log_memory_usage(config, f"After processing sweep {i + 1}")
            if config.run_plotting:
                self._generate_plots(config, result, f"sweep_{i}", analyzer)

        # Phase 3: Aggregate across sweep values
        self._aggregate_sweep_repeated_results(config, result)
        if config.run_plotting:
            self._generate_plots(config, result, "aggregated", None)

    def _simulate(self, seed: int, config: PipelineConfig) -> TRaw:
        """Run simulation with given seed."""
        simulator = self.simulator_factory(seed)
        self.logger.debug(f"Starting simulation (seed={seed})")
        result = simulator.run()
        self.logger.debug(f"Simulation complete (seed={seed})")
        return result

    def _simulate_with_param(
        self,
        seed: int,
        config: PipelineConfig,
        param: str | list[str],
        value: Any,
        *,
        sweep_idx: int | None = None,
    ) -> TRaw:
        """Run simulation with modified parameter.

        Note: The simulator_factory is expected to handle parameter modification
        internally, or the factory should be configured to accept param overrides.
        For now, we pass the value through kwargs.
        """
        # Create simulator with sweep parameter override
        # The factory implementation should handle this
        simulator = self.simulator_factory(
            seed,
            sweep_param=param,
            sweep_value=value,
            sweep_idx=sweep_idx,
        )
        self.logger.debug(f"Starting simulation (seed={seed}, {param}={value})")
        result = simulator.run()
        self.logger.debug(f"Simulation complete (seed={seed})")
        return result

    def _persist_raw_output(
        self,
        raw: TRaw,
        config: PipelineConfig,
        key: str,
    ) -> TRaw:
        """Persist raw outputs to disk and inject file paths.

        Saves raw simulation output to disk when save_dir is set.
        Returns the raw output dict with added file paths (e.g., 'spikes_path')
        so processors can load data from disk.
        """
        if config.save_dir is None:
            return raw
        return _persist_raw_output_to_dir(raw, config.save_dir, key, config.save_compressed)

    def _minimize_raw_output(self, raw: TRaw) -> TRaw:
        """Remove large in-memory payloads when disk-backed paths exist."""
        return _minimize_raw_output(raw)

    def _process_and_analyze(
        self,
        config: PipelineConfig,
        result: PipelineResult,
        raw: TRaw,
        key: str,
    ) -> Any | None:
        """Run processing and analysis phases for single runs.

        Used for SINGLE and SWEEP modes where each run is processed independently.
        """
        # Store raw if requested
        if config.save_raw:
            result.raw_outputs[key] = raw

        processed: Any = raw
        processor: Any | None = None

        # Processing phase
        if config.run_processing and self.processor_factory is not None:
            self.logger.debug(f"Processing {key}")
            t0 = time.time()
            processor = self.processor_factory(raw)
            processed = processor.process()
            t1 = time.time()
            result.processed_data[key] = processed

            processed_dir = self._get_processed_dir(config, key)
            if processed_dir is not None:
                processor.save(processed_dir)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("Processing %s took %.2fs", key, t1 - t0)
        else:
            result.processed_data[key] = processed

        return self._analyze_and_store(config, result, processed, processor, key)

    def _analyze_and_store(
        self,
        config: PipelineConfig,
        result: PipelineResult,
        processed: Any,
        processor: Any | None,
        key: str,
        *,
        sweep_value: Any = None,
    ) -> Any | None:
        """Run analysis and persist outputs."""
        if not (config.run_analysis and self.analyzer_factory is not None):
            return None
        if processed is None:
            return None

        self.logger.debug(f"Analyzing {key}")
        t0 = time.time()
        analyzer = self._build_analyzer(processed, processor)
        df = analyzer.df
        metrics = analyzer.get_summary_metrics()
        t1 = time.time()

        if sweep_value is not None and "sweep_value" not in df.columns:
            df["sweep_value"] = sweep_value

        result.dataframes[key] = df
        result.metrics[key] = metrics

        if hasattr(result, "analyzers"):
            result.analyzers[key] = analyzer

        if hasattr(analyzer, "list_manipulations"):
            manipulations = set(analyzer.list_manipulations())
            if "per_attractor" in manipulations:
                per_attr_df = analyzer.manipulation("per_attractor")
                if not per_attr_df.empty:
                    result.dataframes[f"{key}_per_attractor"] = per_attr_df
            if "time_evolution" in manipulations:
                time_df = analyzer.manipulation(
                    "time_evolution",
                    dt=config.time_evolution_dt,
                    num_steps=config.time_evolution_num_steps,
                )
                if not time_df.empty:
                    result.dataframes[f"{key}_time"] = time_df
            if "transitions" in manipulations:
                tpm = analyzer.manipulation("transitions")
                if not tpm.empty:
                    result.dataframes[f"{key}_tpm"] = tpm
        else:
            if hasattr(analyzer, "get_per_attractor_dataframe"):
                per_attr_df = analyzer.get_per_attractor_dataframe()
                if not per_attr_df.empty:
                    result.dataframes[f"{key}_per_attractor"] = per_attr_df

                if hasattr(analyzer, "get_time_evolution_dataframe"):
                    time_df = analyzer.get_time_evolution_dataframe(
                        dt=config.time_evolution_dt,
                        num_steps=config.time_evolution_num_steps,
                    )
                    if not time_df.empty:
                        result.dataframes[f"{key}_time"] = time_df

            if hasattr(analyzer, "get_transition_matrix"):
                tpm = pd.DataFrame(analyzer.get_transition_matrix())
                if not tpm.empty:
                    result.dataframes[f"{key}_tpm"] = tpm

        self._save_analysis_artifacts(config, result, key)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Analysis %s took %.2fs", key, t1 - t0)
        return analyzer

    def _supports_batch_processing(self) -> bool:
        """Check if the pipeline supports batch processing.

        Returns:
            True if batch_processor_factory is configured and supports batch mode.
        """
        if self.batch_processor_factory is None:
            return False
        # Check for supports_batch property (from BatchProcessorFactory protocol)
        return getattr(self.batch_processor_factory, "supports_batch", True)

    def _process_and_analyze_unified(
        self,
        config: PipelineConfig,
        result: PipelineResult,
        raw_refs: list[TRaw],
        metadata: list[dict[str, Any]],
        key: str,
        sweep_value: Any = None,
    ) -> Any | None:
        """Process and analyze all runs together for consistent attractor IDs.

        The batch processor loads raw data from disk using the file paths in raw_refs,
        processes all runs together for unified attractor identification, and produces
        a single processed output.

        Args:
            config: Pipeline configuration.
            result: PipelineResult to populate.
            raw_refs: List of raw output references (dicts with file paths like 'spikes_path').
            metadata: List of metadata dicts, one per output. Each contains:
                - seed: Random seed used
                - repeat_idx: Index of the repeat (0, 1, 2, ...)
                - sweep_value: Sweep parameter value (if applicable)
                - sweep_idx: Index of sweep value (if applicable)
            key: Key for storing results (e.g., "unified" or "sweep_0").
            sweep_value: Optional sweep value for logging.
        """
        n_runs = len(raw_refs)
        self.logger.info(
            f"Unified processing of {n_runs} runs"
            + (f" (sweep_value={sweep_value})" if sweep_value is not None else "")
        )

        processed: Any = None
        batch_processor: Any | None = None

        # Processing phase - unified batch processing (processor loads from disk)
        if config.run_processing and self.batch_processor_factory is not None:
            self.logger.debug(f"Batch processing {n_runs} runs for '{key}'")
            t0 = time.time()
            batch_processor = self.batch_processor_factory(raw_refs, metadata)
            processed = batch_processor.process_batch(raw_refs, metadata)
            t1 = time.time()
            result.processed_data[key] = processed

            processed_dir = self._get_processed_dir(config, key)
            if processed_dir is not None:
                batch_processor.save(processed_dir)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("Batch processing '%s' took %.2fs", key, t1 - t0)

        # Analysis phase
        return self._analyze_and_store(
            config,
            result,
            processed,
            batch_processor,
            key,
            sweep_value=sweep_value,
        )

    def _build_analyzer(self, processed: Any, processor: Any | None = None) -> Any:
        """Create analyzer, attaching processor config if supported."""
        if processor is not None and hasattr(processor, "get_config"):
            try:
                return self.analyzer_factory(processed, config=processor.get_config())
            except TypeError:
                pass
        return self.analyzer_factory(processed)

    def _aggregate_sweep_results(
        self,
        config: PipelineConfig,
        result: PipelineResult,
    ) -> None:
        """Aggregate results from parameter sweep."""
        self.logger.info("Aggregating sweep results")

        assert config.sweep_values is not None

        dfs = []
        for i, value in enumerate(config.sweep_values):
            key = f"sweep_{i}"
            if key in result.dataframes:
                df = result.dataframes[key].copy()
                df["sweep_value"] = value
                df["sweep_idx"] = i
                dfs.append(df)

        if dfs:
            result.dataframes["aggregated"] = pd.concat(dfs, ignore_index=True)

        # Aggregate metrics per sweep value
        metrics_by_sweep = {}
        for i, value in enumerate(config.sweep_values):
            key = f"sweep_{i}"
            if key in result.metrics:
                metrics_by_sweep[f"sweep_{i}"] = result.metrics[key]

        if metrics_by_sweep:
            result.metrics["aggregated"] = {
                "by_sweep": metrics_by_sweep,
                "sweep_values": config.sweep_values,
            }

    def _aggregate_sweep_repeated_results(
        self,
        config: PipelineConfig,
        result: PipelineResult,
    ) -> None:
        """Aggregate results from sweep with repeats.

        Handles both unified mode (sweep_N keys with repeat embedded in data)
        and per-run mode (sweep_N_repeat_M keys).
        """
        self.logger.info("Aggregating sweep+repeat results")

        assert config.sweep_values is not None

        dfs = []

        # Check if we're in unified mode (sweep_N keys exist without repeat suffix)
        unified_mode = any(
            f"sweep_{i}" in result.dataframes
            for i in range(len(config.sweep_values))
        )

        if unified_mode:
            # Unified mode: sweep_N DataFrames already contain repeat info
            for i, value in enumerate(config.sweep_values):
                key = f"sweep_{i}"
                if key in result.dataframes:
                    df = result.dataframes[key].copy()
                    # sweep_value should already be set, but ensure it
                    if "sweep_value" not in df.columns:
                        df["sweep_value"] = value
                    if "sweep_idx" not in df.columns:
                        df["sweep_idx"] = i
                    dfs.append(df)
        else:
            # Per-run mode: sweep_N_repeat_M keys
            for i, value in enumerate(config.sweep_values):
                for j in range(config.n_repeats):
                    key = f"sweep_{i}_repeat_{j}"
                    if key in result.dataframes:
                        df = result.dataframes[key].copy()
                        df["sweep_value"] = value
                        df["sweep_idx"] = i
                        df["repeat"] = j
                        dfs.append(df)

        if dfs:
            result.dataframes["aggregated"] = pd.concat(dfs, ignore_index=True)

        # Aggregate metrics
        if unified_mode:
            # Unified mode: metrics from sweep_N keys
            all_metrics = [
                m for k, m in result.metrics.items()
                if k.startswith("sweep_") and "_repeat_" not in k and k != "aggregated"
            ]
        else:
            # Per-run mode: metrics from sweep_N_repeat_M keys
            all_metrics = [
                m for k, m in result.metrics.items()
                if k.startswith("sweep_") and "_repeat_" in k
            ]

        if all_metrics:
            result.metrics["aggregated"] = self._compute_aggregate_metrics(all_metrics)

    def _compute_aggregate_metrics(
        self,
        metrics_list: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute aggregate statistics from metrics list."""
        aggregated: dict[str, Any] = {"n_runs": len(metrics_list)}
        if not metrics_list:
            return aggregated

        keys = metrics_list[0].keys()
        for key in keys:
            values = [m[key] for m in metrics_list if key in m]
            if all(isinstance(v, (int, float)) for v in values):
                arr = np.array(values)
                aggregated[f"{key}_mean"] = float(np.mean(arr))
                aggregated[f"{key}_std"] = float(np.std(arr))
                aggregated[f"{key}_min"] = float(np.min(arr))
                aggregated[f"{key}_max"] = float(np.max(arr))

        return aggregated

    def _generate_plots(
        self,
        config: PipelineConfig,
        result: PipelineResult,
        key: str,
        analyzer: Any | None,
    ) -> None:
        """Generate plots using the Plotter."""
        if self.plotter is None:
            self.logger.debug("No plotter configured, skipping plot generation")
            return

        if analyzer is None and hasattr(result, "analyzers"):
            analyzer = result.analyzers.get(key)

        if analyzer is None:
            df = result.dataframes.get(key)
            if df is None:
                self.logger.warning(f"No analyzer or DataFrame for key '{key}', skipping plots")
                return
            if isinstance(self.plotter, (SpecPlotter, SeabornPlotter)):
                self.logger.warning(
                    f"Plotter requires analyzer for key '{key}', skipping plots"
                )
                return
            analyzer = df
            metrics = result.metrics.get(key)
        else:
            metrics = None

        self.logger.info(f"Generating plots for '{key}'")

        save_dir = self._get_plots_dir(config, key)

        figures = self.plotter.plot(analyzer, save_dir=save_dir, metrics=metrics)
        result.figures.extend(figures)

        self.logger.info(f"Generated {len(figures)} plot(s)")

    def _get_processed_dir(self, config: PipelineConfig, key: str) -> Path | None:
        if not (config.save_dir and config.save_processed):
            return None
        base = config.save_dir / "processed"
        if config.mode in (ExecutionMode.SWEEP, ExecutionMode.SWEEP_REPEATED):
            return base / key
        return base

    def _get_plots_dir(self, config: PipelineConfig, key: str) -> Path | None:
        if not (config.save_dir and config.save_plots):
            return None
        base = config.save_dir / "plots"
        if config.mode in (ExecutionMode.SWEEP, ExecutionMode.SWEEP_REPEATED):
            return base / key
        return base

    def _align_transition_matrices(self, result: PipelineResult) -> None:
        """Align transition matrices across runs and store in result.dataframes."""
        if not hasattr(result, "align_transition_matrices"):
            return
        try:
            canonical_labels, aligned = result.align_transition_matrices(labels="identity")
        except Exception as exc:
            self.logger.warning(f"Failed to align transition matrices: {exc}")
            return
        if not aligned:
            return
        for key, tpm in aligned.items():
            result.dataframes[f"{key}_tpm"] = tpm

    def _save_results(self, config: PipelineConfig, result: PipelineResult) -> None:
        """Save pipeline results to disk."""
        assert config.save_dir is not None
        self.logger.info(f"Saving results to {config.save_dir}")
        save_result(result, config.save_dir)

    def _report_progress(
        self,
        config: PipelineConfig,
        current: int,
        total: int,
        message: str,
    ) -> None:
        """Report progress via callback if configured."""
        if config.progress_callback is not None:
            try:
                config.progress_callback(current, total, message)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")


__all__ = ["Pipeline"]
