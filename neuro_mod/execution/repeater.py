from __future__ import annotations

from pathlib import Path
from typing import Callable, Any
import shutil

import numpy as np

from neuro_mod.execution.helpers import Logger


class Repeater:
    """Run repeatable simulations with reproducible seeds and optional output handling."""

    def __init__(
        self,
        n_repeats: int | None,
        save_dir: Path | str,
        *,
        stager_factory: Callable[[int], Any],
        config: Path | str | None = None,
        seed: int = 256,
        seeds_file: Path | str | None = None,
        load_saved_seeds: bool = False,
        run_fn: Callable[[Any], dict] | None = None,
        save_fn: Callable[[dict, int, Any], None] | None = None,
        logger: Logger | None = None,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.config = config
        self.seed = seed
        self.logger = logger or Logger(name=self.__class__.__name__)
        self.stager_factory = stager_factory
        self.run_fn = run_fn or (lambda stager: stager.run())
        self.save_fn = save_fn or self._default_save_outputs

        self._set_dirs()
        self.seeds, self.n_repeats = self._initialize_seeds(
            n_repeats=n_repeats,
            seeds_file=seeds_file,
            load_saved_seeds=load_saved_seeds,
        )
        self._store_meta()

    def run(self) -> None:
        self.logger.info(f"Running {self.n_repeats} repeats.")
        for idx, seed in enumerate(self.seeds):
            self._step(seed=seed, idx=idx)
        self.logger.info("Repeats complete.")

    def _step(self, seed: int, idx: int) -> None:
        self.logger.info(f"Starting repeat {idx + 1}/{self.n_repeats}.")
        stager = self.stager_factory(seed)
        outputs = self.run_fn(stager)
        self.save_fn(outputs, idx, stager)

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
        spikes = outputs.get("spikes") if isinstance(outputs, dict) else None
        if spikes is not None:
            np.save(self._data_dir / f"spikes_{idx}.npy", spikes)
            if hasattr(stager, "_plot"):
                stager._plot(spikes, plt_path=self._plot_dir / f"spikes_{idx}.png")

        clusters = outputs.get("clusters") if isinstance(outputs, dict) else None
        if clusters is not None and idx == 0:
            np.save(self.save_dir / "clusters.npy", clusters)
