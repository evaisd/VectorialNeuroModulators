
import os
from pathlib import Path
from scipy.linalg import circulant
from neuro_mod.execution.stagers import StageSNNSimulation
import numpy as np


if __name__ == '__main__':
    config = 'configs/18_clusters_snn.yaml'
    wd = Path().cwd().parent
    os.chdir(wd)
    n_perturbed = 6
    alpha = 5.
    beta = 0.5
    perturbation = np.ones((38, 38))
    base_pert = circulant(np.eye(n_perturbed)[-1])
    pert_exc = alpha * base_pert
    pert_inh = beta * base_pert.T
    perturbation[:n_perturbed, :n_perturbed] += pert_exc
    perturbation[:n_perturbed, 19:19 + n_perturbed] += pert_inh
    stager = StageSNNSimulation(config,
                                j_perturbation=perturbation)
    stager.execute(plot_arg='spikes')