
import os
from pathlib import Path
from neuro_mod.spiking_neuron_net.execution.staging import StageSimulation


if __name__ == "__main__":
    wd = Path(__file__).parent.parent
    os.chdir(wd)
    stager = StageSimulation("spiking_network_simulation_config_template.yaml")
    stager.execute()