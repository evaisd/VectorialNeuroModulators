Vectorial Neuro Modulators
===========================

**VectorialNeuroModulators** is a small research-oriented toolkit for simulating
recurrent spiking neural networks and their mean-field (population-rate) approximations.
It provides:

- **Mean-field models** of clustered leaky integrate-and-fire (LIF) networks.
- **Spiking network simulations** built on PyTorch for efficient experimentation.
- **Config-driven experiments** using simple YAML configuration files.

This project is intended for rapid prototyping and exploration, not as a polished
end-user package.

---

### Installation

**Requirements**

- **Python**: 3.10+ (recommended)
- **Core libraries**: `numpy`, `scipy`, `matplotlib`, `torch`, `pyyaml`

Install dependencies:

```bash
pip install -r requirements.txt
```

Then, either install the package in editable mode:

```bash
pip install -e .
```

or add the repository root to your `PYTHONPATH` when running scripts.

---

### Project structure

- **`neuro_mod/`**: Main Python package.
  - **`mean_field/`**
    - **`core/lif_mean_field.py`**: `LIFMeanField` class implementing the
      population-rate mean-field model and fixed-point / stability analysis.
    - **`runners/main_runner.py`**: `MainMFRunner` helper to run mean-field
      fixed-point solvers and stability classification.
  - **`spiking_neuron_net/`**
    - **`lif_net.py`**: `LIFNet` PyTorch module for simulating a recurrent
      LIF spiking network with synaptic and membrane dynamics.
    - **`clustering/`**: Utilities to construct clustered network connectivity.
    - **`execution/`**: Staging and orchestration for simulations.
    - **`external/`**: Stimulus and external current generators.
- **`configs/`**: Example YAML configuration templates for simulations.
- **`examples/`**:
  - `mean_field_example.ipynb`: Interactive example of mean-field usage.
  - `spiking_net_test_simulation.py`: Example script that stages and runs a
    full spiking-network simulation from a YAML config.
- **`test_sim/`**: Example output folder (plots and data) from a test run.

---

### Quick start

#### Run a spiking network test simulation

From the repository root:

```bash
python examples/spiking_net_test_simulation.py
```

By default this uses `configs/spiking_network_simulation_config_template.yaml`
and writes outputs (raster plots, `.npz` data) under `test_sim/sim_name/`.

To customize a simulation, copy the template, edit it, and point the stager to it:

```bash
cp configs/spiking_network_simulation_config_template.yaml configs/my_sim.yaml
# edit configs/my_sim.yaml
python -m neuro_mod.spiking_neuron_net.execution.staging configs/my_sim.yaml
```

#### Use the mean-field model directly

The `LIFMeanField` class exposes population-level equations and utilities for
finding fixed points and their stability:

```python
import numpy as np
from neuro_mod.mean_field.core import LIFMeanField

n_clusters = 2
C = np.ones((n_clusters, n_clusters), dtype=int)
J = 0.1 * np.ones((n_clusters, n_clusters))
j_ext = np.array([2.3, 2.3])
c_ext = np.array([320, 320])
nu_ext = np.array([7.0, 7.0])

lif_mf = LIFMeanField(
    n_clusters=n_clusters,
    c_matrix=C,
    j_matrix=J,
    j_ext=j_ext,
    c_ext=c_ext,
    nu_ext=nu_ext,
    tau_membrane=[0.02, 0.02],
    tau_synaptic=[0.005, 0.005],
    threshold=[1.5, 0.75],
    reset_voltage=0.0,
    tau_refractory=0.005,
)

sol = lif_mf.solve_rates()
nu_star = sol.x
fp_type, eigvals = lif_mf.determine_stability(nu_star)
print("Fixed point:", nu_star)
print("Type:", fp_type)
```

---

### Google-style docstrings

Public classes and methods in this project follow Google-style docstrings. For example:

```python
class LIFMeanField:
    """Mean-field model of a clustered LIF network.

    Args:
        n_clusters: Number of populations (clusters).
        c_matrix: Connectivity matrix (numbers of synapses).
        j_matrix: Synaptic efficacy matrix.
        j_ext: External synaptic efficacies per population.
        c_ext: External in-degree per population.
        nu_ext: External Poisson input rates per population (Hz).
        tau_membrane: Membrane time constants.
        tau_synaptic: Synaptic time constants.
        threshold: Firing thresholds.
        reset_voltage: Reset voltages after spikes.
        tau_refractory: Absolute refractory periods.
    """
```

When contributing new code, keep docstrings concise and follow this pattern for
parameters, returns, and raises.

---

### License

This project is distributed under the terms of the license specified in `LICENSE`.