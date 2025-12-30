
import os
import numpy as np
from pathlib import Path
import shutil
from neuro_mod.execution.stagers import StageSNNSimulation
from neuro_mod.execution.helpers import Logger
from neuro_mod.spiking_neuron_net.analysis.analyzer import Analyzer


class Repeater():

    n_repeats: int
    seed: int = 256

    def __init__(self, n_repeats: int, config: str, save_dir: Path | str):
        self.n_repeats = n_repeats
        np.random.seed(self.seed)
        self.seeds = np.random.randint(low=0, high=2**31, size=n_repeats)
        self.config: str = config
        self.save_dir = Path(save_dir)
        self.logger = Logger(name=self.__class__.__name__)
        self._set_dirs()
        self._store_meta()

    def run(self):
        self.logger.info(f"Running {self.n_repeats} repeats.")
        for i, seed in enumerate(self.seeds):
            self._step(seed=seed, idx=i)
        self.logger.info("Repeats complete.")

    def _step(self, seed: int, idx: int):
        self.logger.info(f"Starting repeat {idx + 1}/{self.n_repeats}.")
        stager = StageSNNSimulation(self.config, random_seed=seed, logger=self.logger)
        outputs = stager.run()
        np.save(self._data_dir / f"spikes_{idx}.npy", outputs["spikes"])
        if idx == 0:
            np.save(self.save_dir / "clusters.npy", outputs["clusters"])
        stager._plot(
            outputs["spikes"],
            plt_path=self._plot_dir / f"spikes_{idx}.png"
        )

    def _set_dirs(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = self.save_dir / "data"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._plot_dir = self.save_dir / "plots"
        self._plot_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_dir = self.save_dir / "metadata"
        self._metadata_dir.mkdir(parents=True, exist_ok=True)
        if self.logger.file_path is None:
            self.logger.attach_file(str(self._metadata_dir / "run.log"))

    def _store_meta(self):
        seeds_file = self._metadata_dir / "seeds.txt"
        seeds_file.write_text("\n".join([str(seed) for seed in self.seeds.tolist()]))
        config_file = self._metadata_dir / "config.yaml"
        shutil.copy(self.config, config_file)


def main():
    wd = Path(next(p for p in Path().resolve().parents if p.name == 'VectorialNeuroModulators'))
    os.chdir(wd)
    save_dir = Path('simulations/repeated_test')
    repeater = Repeater(n_repeats=2, config="configs/snn_long_run.yaml", save_dir=save_dir)
    repeater.run()
    repeater.logger.info("Building analyzer and saving analysis.")
    analyzer = Analyzer(save_dir / "data", clusters=save_dir / "clusters.npy")
    analyzer.save_analysis(save_dir / "analysis")
    repeater.logger.info("Analysis generation complete.")


if __name__ == "__main__":
    main()


# codex resume 019b6a12-d22a-7283-b885-4739684a18b1
