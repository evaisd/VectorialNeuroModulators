
import os
from pathlib import Path
from neuro_mod.execution import StageSNNSimulation

if __name__ == "__main__":
    wd = Path().cwd().parent
    os.chdir(wd)
    stager = StageSNNSimulation("configs/default_snn_params.yaml")
    stager.execute(plot_arg='spikes')