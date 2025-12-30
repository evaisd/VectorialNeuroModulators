
import os
from pathlib import Path

from neuro_mod.execution import Repeater
from neuro_mod.execution.stagers import StageSNNSimulation
from neuro_mod.execution.helpers import Logger
from neuro_mod.spiking_neuron_net.analysis.analyzer import Analyzer

def main():
    wd = Path(next(p for p in Path().resolve().parents if p.name == 'VectorialNeuroModulators'))
    os.chdir(wd)
    save_dir = Path('simulations/snn_test_run_2')
    logger = Logger(name="Repeater")
    config = "configs/snn_test_run.yaml"
    repeater = Repeater(
        n_repeats=2,
        config=config,
        save_dir=save_dir,
        logger=logger,
        stager_factory=lambda seed: StageSNNSimulation(
            config,
            random_seed=seed,
            logger=logger,
        ),
        seeds_file='simulations/snn_test_run_1/metadata/seeds.txt'
    )
    repeater.run()
    repeater.logger.info("Building analyzer and saving analysis.")
    analyzer = Analyzer(save_dir / "data", clusters=save_dir / "clusters.npy")
    analyzer.save_analysis(save_dir / "analysis")
    repeater.logger.info("Analysis generation complete.")


if __name__ == "__main__":
    main()


# codex resume 019b6a12-d22a-7283-b885-4739684a18b1
