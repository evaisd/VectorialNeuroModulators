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
from neuro_mod.pipeline.protocols import (
    AnalyzerFactory,
    BatchProcessorFactory,
    Plotter,
    ProcessorFactory,
    SimulatorFactory,
)


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
            batch_processor_factory: Optional factory for unified batch processing.
                If provided and unified_processing=True, processes all repeats together
                for consistent attractor IDs across runs.
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

        # Save results if save_dir specified
        if config.save_dir:
            self._save_results(config, result)

        return result

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
        self._process_and_analyze(config, result, raw, key="single")

        if config.run_plotting:
            self._generate_plots(config, result, "single")

    def _run_repeated(
        self,
        config: PipelineConfig,
        result: PipelineResult,
        seeds: list[int],
    ) -> None:
        """Execute repeated runs with seed management.

        If unified_processing=True and batch_processor_factory is configured,
        all repeats are processed together for consistent attractor identities.
        """
        n_total = len(seeds)
        use_unified = config.unified_processing and self._supports_batch_processing()

        # Phase 1: Run all simulations
        raw_outputs: list[TRaw] = []
        metadata: list[dict[str, Any]] = []

        if config.parallel and n_total > 1:
            raw_outputs, metadata = self._simulate_repeated_parallel(config, seeds)
        else:
            for i, seed in enumerate(seeds):
                self.logger.info(f"Running repeat {i + 1}/{n_total} (seed={seed})")
                self._report_progress(config, i, n_total, f"repeat {i + 1}/{n_total}")
                raw = self._simulate(seed, config)
                raw = self._persist_raw_output(raw, config, f"repeat_{i}")
                raw_outputs.append(raw)
                metadata.append({"seed": seed, "repeat_idx": i})

        # Phase 2: Processing and Analysis
        if use_unified:
            # Unified processing: all repeats together
            self._process_and_analyze_unified(
                config, result, raw_outputs, metadata, key="unified"
            )
            # Store as aggregated since it's already unified
            if "unified" in result.dataframes:
                result.dataframes["aggregated"] = result.dataframes["unified"]
            if "unified" in result.metrics:
                result.metrics["aggregated"] = result.metrics["unified"]
        else:
            # Fallback: per-run processing
            for i, (raw, meta) in enumerate(zip(raw_outputs, metadata)):
                self._process_and_analyze(config, result, raw, key=f"repeat_{i}")
            # Aggregate across repeats
            self._aggregate_repeated_results(config, result)

        if config.run_plotting:
            self._generate_plots(config, result, "aggregated")

    def _simulate_repeated_parallel(
        self,
        config: PipelineConfig,
        seeds: list[int],
    ) -> tuple[list[TRaw], list[dict[str, Any]]]:
        """Run simulations in parallel, returning outputs and metadata.

        Returns:
            Tuple of (raw_outputs, metadata) lists, ordered by repeat index.
        """
        ExecutorClass = (
            ProcessPoolExecutor if config.executor == "process"
            else ThreadPoolExecutor
        )
        n_total = len(seeds)

        self.logger.info(f"Running {n_total} repeats in parallel ({config.executor} executor)")

        # Pre-allocate lists to maintain order
        raw_outputs: list[TRaw | None] = [None] * n_total
        metadata: list[dict[str, Any]] = [{}] * n_total

        with ExecutorClass(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(self._simulate, seed, config): (i, seed)
                for i, seed in enumerate(seeds)
            }

            for future in as_completed(futures):
                i, seed = futures[future]
                try:
                    raw = future.result()
                    raw = self._persist_raw_output(raw, config, f"repeat_{i}")
                    raw_outputs[i] = raw
                    metadata[i] = {"seed": seed, "repeat_idx": i}
                    self.logger.debug(f"Repeat {i} complete (seed={seed})")
                    self._report_progress(config, i + 1, n_total, f"repeat {i + 1} done")
                except Exception as exc:
                    self.logger.error(f"Repeat {i} failed: {exc}")
                    raise

        return raw_outputs, metadata  # type: ignore[return-value]

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

            raw = self._simulate_with_param(seed, config, config.sweep_param, value)
            raw = self._persist_raw_output(raw, config, f"sweep_{i}")
            self._process_and_analyze(config, result, raw, key=f"sweep_{i}")

        self._aggregate_sweep_results(config, result)

        if config.run_plotting:
            self._generate_plots(config, result, "aggregated")

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
        """
        assert config.sweep_param is not None
        assert config.sweep_values is not None

        result.sweep_metadata = {
            "param": config.sweep_param,
            "values": config.sweep_values,
            "n_repeats": len(seeds),
        }

        n_values = len(config.sweep_values)
        n_repeats = len(seeds)
        n_total = n_values * n_repeats
        use_unified = config.unified_processing and self._supports_batch_processing()

        # Phase 1: Run all simulations, grouped by sweep value
        raw_by_sweep: dict[int, list[TRaw]] = defaultdict(list)
        meta_by_sweep: dict[int, list[dict[str, Any]]] = defaultdict(list)
        step = 0

        for i, value in enumerate(config.sweep_values):
            for j, seed in enumerate(seeds):
                self.logger.info(
                    f"Sweep {i + 1}/{n_values}, repeat {j + 1}/{n_repeats} "
                    f"({config.sweep_param}={value}, seed={seed})"
                )
                self._report_progress(config, step, n_total, f"sweep {i + 1}, repeat {j + 1}")

                raw = self._simulate_with_param(seed, config, config.sweep_param, value)
                raw = self._persist_raw_output(raw, config, f"sweep_{i}_repeat_{j}")
                raw_by_sweep[i].append(raw)
                meta_by_sweep[i].append({
                    "seed": seed,
                    "repeat_idx": j,
                    "sweep_value": value,
                    "sweep_idx": i,
                })
                step += 1

        # Phase 2: Unified processing PER SWEEP VALUE
        for i, value in enumerate(config.sweep_values):
            if use_unified:
                # Process all repeats for this sweep value together
                self._process_and_analyze_unified(
                    config, result,
                    raw_by_sweep[i],
                    meta_by_sweep[i],
                    key=f"sweep_{i}",
                    sweep_value=value,
                )
            else:
                # Fallback: process each repeat separately
                for j, (raw, meta) in enumerate(zip(raw_by_sweep[i], meta_by_sweep[i])):
                    self._process_and_analyze(config, result, raw, key=f"sweep_{i}_repeat_{j}")

        # Phase 3: Aggregate across sweep values
        self._aggregate_sweep_repeated_results(config, result)

        if config.run_plotting:
            self._generate_plots(config, result, "aggregated")

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
    ) -> TRaw:
        """Run simulation with modified parameter.

        Note: The simulator_factory is expected to handle parameter modification
        internally, or the factory should be configured to accept param overrides.
        For now, we pass the value through kwargs.
        """
        # Create simulator with sweep parameter override
        # The factory implementation should handle this
        simulator = self.simulator_factory(seed, sweep_param=param, sweep_value=value)
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
        """Persist raw outputs to disk and inject file paths when possible."""
        if not config.save_raw or config.save_dir is None:
            return raw

        data_dir = config.save_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(raw, dict):
            if "spikes_path" not in raw:
                spikes = raw.get("spikes")
                if spikes is not None:
                    clusters = raw.get("clusters")
                    spikes_path = data_dir / f"{key}_spikes.npz"
                    if clusters is not None:
                        np.savez_compressed(spikes_path, spikes=spikes, clusters=clusters)
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

    def _process_and_analyze(
        self,
        config: PipelineConfig,
        result: PipelineResult,
        raw: TRaw,
        key: str,
    ) -> None:
        """Run processing and analysis phases."""
        # Store raw if requested
        if config.save_raw:
            result.raw_outputs[key] = raw

        processed: Any = raw

        # Processing phase
        if config.run_processing and self.processor_factory is not None:
            self.logger.debug(f"Processing {key}")
            processor = self.processor_factory(raw)
            processed = processor.process()
            result.processed_data[key] = processed

            if config.save_processed and config.save_dir:
                processor.save(config.save_dir / "processed" / key)
        else:
            result.processed_data[key] = processed

        # Analysis phase
        if config.run_analysis and self.analyzer_factory is not None:
            self.logger.debug(f"Analyzing {key}")
            analyzer = self.analyzer_factory(processed)
            result.dataframes[key] = analyzer.to_dataframe()
            result.metrics[key] = analyzer.get_summary_metrics()
            if hasattr(analyzer, "get_time_evolution_dataframe"):
                time_df = analyzer.get_time_evolution_dataframe()
                if not time_df.empty:
                    result.dataframes[f"{key}_time"] = time_df

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
        raw_outputs: list[TRaw],
        metadata: list[dict[str, Any]],
        key: str,
        sweep_value: Any = None,
    ) -> None:
        """Process and analyze all runs together for consistent attractor IDs.

        Args:
            config: Pipeline configuration.
            result: PipelineResult to populate.
            raw_outputs: List of raw simulation outputs.
            metadata: List of metadata dicts, one per output. Each contains:
                - seed: Random seed used
                - repeat_idx: Index of the repeat (0, 1, 2, ...)
                - sweep_value: Sweep parameter value (if applicable)
                - sweep_idx: Index of sweep value (if applicable)
            key: Key for storing results (e.g., "unified" or "sweep_0").
            sweep_value: Optional sweep value for logging.
        """
        n_runs = len(raw_outputs)
        self.logger.info(
            f"Unified processing of {n_runs} runs"
            + (f" (sweep_value={sweep_value})" if sweep_value is not None else "")
        )

        # Store raw outputs if requested
        if config.save_raw:
            for i, (raw, meta) in enumerate(zip(raw_outputs, metadata)):
                raw_key = f"{key}_run_{i}"
                result.raw_outputs[raw_key] = raw

        processed: Any = None

        # Processing phase - unified batch processing
        if config.run_processing and self.batch_processor_factory is not None:
            self.logger.debug(f"Batch processing {n_runs} runs for '{key}'")
            batch_processor = self.batch_processor_factory(raw_outputs, metadata)
            processed = batch_processor.process_batch(raw_outputs, metadata)
            result.processed_data[key] = processed

            if config.save_processed and config.save_dir:
                batch_processor.save(config.save_dir / "processed" / key)

        # Analysis phase
        if config.run_analysis and self.analyzer_factory is not None and processed is not None:
            self.logger.debug(f"Analyzing unified data for '{key}'")
            analyzer = self.analyzer_factory(processed)
            df = analyzer.to_dataframe()

            # Ensure metadata columns are present
            if sweep_value is not None and "sweep_value" not in df.columns:
                df["sweep_value"] = sweep_value

            result.dataframes[key] = df
            result.metrics[key] = analyzer.get_summary_metrics()
            if hasattr(analyzer, "get_time_evolution_dataframe"):
                time_df = analyzer.get_time_evolution_dataframe()
                if not time_df.empty:
                    result.dataframes[f"{key}_time"] = time_df

    def _aggregate_repeated_results(
        self,
        config: PipelineConfig,
        result: PipelineResult,
    ) -> None:
        """Aggregate results from repeated runs."""
        self.logger.info("Aggregating repeated run results")

        # Concatenate DataFrames with repeat index
        dfs = []
        for key, df in result.dataframes.items():
            if key.startswith("repeat_"):
                repeat_idx = int(key.split("_")[1])
                df = df.copy()
                df["repeat"] = repeat_idx
                dfs.append(df)

        if dfs:
            result.dataframes["aggregated"] = pd.concat(dfs, ignore_index=True)

        # Aggregate metrics
        metrics_list = [m for k, m in result.metrics.items() if k.startswith("repeat_")]
        if metrics_list:
            result.metrics["aggregated"] = self._compute_aggregate_metrics(metrics_list)

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
    ) -> None:
        """Generate plots using the Plotter."""
        if self.plotter is None:
            self.logger.debug("No plotter configured, skipping plot generation")
            return

        if key not in result.dataframes:
            self.logger.warning(f"No DataFrame for key '{key}', skipping plots")
            return

        self.logger.info(f"Generating plots for '{key}'")

        save_dir = config.save_dir / "plots" if config.save_dir and config.save_plots else None

        figures = self.plotter.plot(
            data=result.dataframes[key],
            metrics=result.metrics.get(key),
            save_dir=save_dir,
        )
        result.figures.extend(figures)

        self.logger.info(f"Generated {len(figures)} plot(s)")

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
