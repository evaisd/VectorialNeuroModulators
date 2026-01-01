"""Repeatable simulation runner with seed management."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import shutil

import numpy as np

from neuro_mod.execution.helpers import Logger
from neuro_mod.execution.helpers.repeater_helpers import (
    default_run_fn,
    default_save_outputs,
    call_stager_factory,
    validate_process_pickling,
    build_process_save_fn,
    run_process_pool,
)
from neuro_mod.visualization import folder_plots_to_pdf


class Repeater:
    """Run repeatable simulations with reproducible seeds and optional output handling."""

    def __init__(
        self,
        n_repeats: int | None,
        save_dir: Path | str,
        *,
        stager_factory: Callable[..., Any],
        config: Path | str | None = None,
        seed: int = 256,
        seeds_file: Path | str | None = None,
        load_saved_seeds: bool = False,
        parallel: bool = False,
        max_workers: int | None = None,
        executor: str = "thread",
        run_fn: Callable[[Any], dict] | None = None,
        save_fn: Callable[[dict, int, Any], None] | None = None,
        export_plots_pdf: bool = False,
        plots_pdf_path: Path | str | None = None,
        logger: Logger | None = None,
    ) -> None:
        """Initialize the repeater.

        Args:
            n_repeats: Number of repetitions or None to infer from seeds.
            save_dir: Directory to store outputs.
            stager_factory: Callable that creates a stager for a seed (optionally accepts logger=).
            config: Optional config path for metadata copy.
            seed: Base seed for generating repeat seeds.
            seeds_file: Optional file with pre-generated seeds.
            load_saved_seeds: Whether to load seeds from `save_dir`.
            parallel: Whether to run repeats in parallel.
            max_workers: Maximum number of worker threads/processes to use.
            executor: Execution backend when parallel (thread or process).
            run_fn: Optional function that runs a stager.
            save_fn: Optional function that saves outputs.
            export_plots_pdf: Whether to export plots folder into a single PDF.
            plots_pdf_path: Optional output path for the plots PDF.
            logger: Optional logger instance.
        """
        self.save_dir = Path(save_dir)
        self.config = config
        self.seed = seed
        self.logger = logger or Logger(name=self.__class__.__name__)
        self.stager_factory = stager_factory
        self.run_fn = run_fn or default_run_fn
        self.save_fn = save_fn or self._default_save_outputs
        self.parallel = parallel
        self.max_workers = max_workers
        self.executor = executor
        self._save_fn_is_default = save_fn is None
        self.export_plots_pdf = export_plots_pdf
        self.plots_pdf_path = plots_pdf_path

        self._set_dirs()
        self.seeds, self.n_repeats = self._initialize_seeds(
            n_repeats=n_repeats,
            seeds_file=seeds_file,
            load_saved_seeds=load_saved_seeds,
        )
        self._store_meta()

    def run(self) -> None:
        """Execute all repeats."""
        self.logger.info(f"Running {self.n_repeats} repeats.")
        if self.parallel and self.n_repeats > 1:
            worker_desc = self.max_workers if self.max_workers is not None else "default"
            self.logger.info(
                f"Parallel execution enabled with max_workers={worker_desc} "
                f"using executor={self.executor}."
            )
            self._run_parallel()
        else:
            for idx, seed in enumerate(self.seeds):
                self._step(seed=seed, idx=idx)
        if self.export_plots_pdf:
            self._export_plots_pdf()
        self.logger.info("Repeats complete.")

    def _step(self, seed: int, idx: int) -> None:
        self.logger.info(f"Starting repeat {idx + 1}/{self.n_repeats}.")
        try:
            stager = call_stager_factory(self.stager_factory, seed, self.logger)
            outputs = self.run_fn(stager)
            self.save_fn(outputs, idx, stager)
        except Exception:
            self.logger.error(
                f"Repeat {idx + 1}/{self.n_repeats} failed.\n{traceback.format_exc()}"
            )
            raise
        self.logger.info(f"Finished repeat {idx + 1}/{self.n_repeats}.")

    def _run_parallel(self) -> None:
        if self.executor not in {"thread", "process"}:
            raise ValueError(f"Unknown executor: {self.executor}")

        if self.executor == "process":
            self._run_parallel_process()
            return

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._step, seed, idx): (idx, seed)
                for idx, seed in enumerate(self.seeds)
            }
            for future in as_completed(futures):
                idx, seed = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    self.logger.error(
                        f"Parallel repeat {idx + 1}/{self.n_repeats} "
                        f"with seed {seed} failed: {exc}"
                    )
                    raise

    def _run_parallel_process(self) -> None:
        logger_settings = {
            "name": self.logger.name,
            "level": self.logger.level,
            "file_path": self.logger.file_path,
            "include_timestamp": self.logger.include_timestamp,
        }
        process_save_fn = build_process_save_fn(
            self.save_fn,
            use_default=self._save_fn_is_default,
            data_dir=self._data_dir,
            plot_dir=self._plot_dir,
            save_dir=self.save_dir,
        )
        validate_process_pickling(
            stager_factory=self.stager_factory,
            run_fn=self.run_fn,
            save_fn=process_save_fn,
        )
        executor, futures = run_process_pool(
            seeds=self.seeds,
            n_repeats=self.n_repeats,
            max_workers=self.max_workers,
            stager_factory=self.stager_factory,
            run_fn=self.run_fn,
            save_fn=process_save_fn,
            logger_settings=logger_settings,
        )
        with executor:
            for future in as_completed(futures):
                idx, seed = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    self.logger.error(
                        f"Parallel repeat {idx + 1}/{self.n_repeats} "
                        f"with seed {seed} failed: {exc}"
                    )
                    raise

    def _set_dirs(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = self.save_dir / "data"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._plot_dir = self.save_dir / "plots"
        self._plot_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_dir = self.save_dir / "metadata"
        self._metadata_dir.mkdir(parents=True, exist_ok=True)
        if self.logger.file_path is None:
            self.logger.attach_file(str(self._metadata_dir / "run.log"))

    def _initialize_seeds(
        self,
        *,
        n_repeats: int | None,
        seeds_file: Path | str | None,
        load_saved_seeds: bool,
    ) -> tuple[np.ndarray, int]:
        seeds_path = None
        if seeds_file is not None:
            seeds_path = Path(seeds_file)
        elif load_saved_seeds:
            candidate = self._metadata_dir / "seeds.txt"
            if candidate.exists():
                seeds_path = candidate

        if seeds_path is not None:
            self.logger.info(f"Loading seeds from {seeds_path}.")
            seeds = self._load_seeds(seeds_path)
            if n_repeats is None:
                n_repeats = len(seeds)
            elif n_repeats != len(seeds):
                raise ValueError(
                    "n_repeats does not match seeds file length: "
                    f"{n_repeats} != {len(seeds)}"
                )
            return seeds, n_repeats

        if n_repeats is None:
            raise ValueError("n_repeats is required when not loading seeds.")

        rng = np.random.RandomState(self.seed)
        seeds = rng.randint(low=0, high=2**31, size=n_repeats)
        return seeds, n_repeats

    @staticmethod
    def _load_seeds(path: Path) -> np.ndarray:
        lines = path.read_text().splitlines()
        seeds = [int(line.strip()) for line in lines if line.strip()]
        return np.array(seeds, dtype=np.int64)

    def _store_meta(self) -> None:
        seeds_file = self._metadata_dir / "seeds.txt"
        seeds_file.write_text("\n".join([str(seed) for seed in self.seeds.tolist()]))
        if self.config is None:
            return
        config_path = Path(self.config)
        if config_path.exists():
            config_file = self._metadata_dir / "config.yaml"
            shutil.copy(config_path, config_file)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    def _default_save_outputs(self, outputs: dict, idx: int, stager: Any) -> None:
        default_save_outputs(
            outputs,
            idx,
            stager,
            data_dir=self._data_dir,
            plot_dir=self._plot_dir,
            save_dir=self.save_dir,
        )

    def _export_plots_pdf(self) -> None:
        try:
            output_path = folder_plots_to_pdf(
                self._plot_dir,
                output_path=self.plots_pdf_path,
            )
        except ValueError as exc:
            self.logger.warning(f"Skipping plots PDF export: {exc}")
            return
        self.logger.info(f"Saved plots PDF to {output_path}.")
