
import os
from pathlib import Path
from neuro_mod.execution import StageSNNSimulation

if __name__ == "__main__":
    wd = Path().cwd().parent
    os.chdir(wd)
    stager = StageSNNSimulation("configs/18_clusters_snn.yaml")
    stager.execute(plot_arg='spikes')