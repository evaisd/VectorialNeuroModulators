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
  - **`core/`**
    - **`mean_field/`**
      - **`core/lif_mean_field.py`**: `LIFMeanField` class implementing the
        population-rate mean-field model and fixed-point / stability analysis.
      - **`analysis/`**: Mean-field analysis helpers and visualization.
    - **`spiking_net/`**
      - **`core/lif_net.py`**: `LIFNet` PyTorch module for simulating a recurrent
        LIF spiking network with synaptic and membrane dynamics.
      - **`core/external/`**: Stimulus and external current generators.
      - **`processing/`**: Data processing (spikes to attractors).
      - **`analysis/`**: Spiking-network analysis and attractor logic.
    - **`clustering/`**: Utilities to construct clustered network connectivity.
    - **`perturbations/`**: Perturbation generators for structured inputs.
  - **`pipeline/`**: Generic experiment pipeline orchestrating Simulation -> Processing -> Analysis -> Plotting.
  - **`execution/`**: Stagers, sweep runners, repeaters, and logging helpers.
  - **`visualization/`**: Plotting utilities including journal-style and seaborn integration.
- **`configs/`**: Example YAML configuration templates for simulations.
- **`scripts/`**:
  - `run_snn.py`: Run repeated spiking-net simulations and write analysis artifacts.
  - `run_perturbed_snn.py`: Run repeated spiking-net simulations with vectorial perturbations.
  - `load_snn_analysis.py`: Load a saved spiking-net analysis folder.
- **`test_sim/`**: Example output folder (plots and data) from a test run.

---

### Main apps and usage

#### Spiking network stager (single simulation)

`StageSNNSimulation` stages a single spiking-network run from a YAML config
and returns voltages, currents, spikes, and metadata.

```bash
python examples/spiking_net_test_simulation.py
```

Minimal Python usage:

```python
from neuro_mod.execution.stagers import StageSNNSimulation

stager = StageSNNSimulation("configs/default_snn_params.yaml", random_seed=123)
outputs = stager.run()
stager.execute(plot_arg="spikes")
```

If `settings.save` and `settings.plot` are enabled in the config, `execute()`
writes outputs under `settings.save_dir/settings.sim_name/` with `data/`,
`plots/`, and `metadata/` subfolders.

To customize a simulation, copy a template and point the stager to it:

```bash
cp configs/templates/spiking_net_template.yaml configs/my_sim.yaml
# edit configs/my_sim.yaml
python -c "from neuro_mod.execution.stagers import StageSNNSimulation; StageSNNSimulation('configs/my_sim.yaml').execute(plot_arg='spikes')"
```

#### Spiking network stager (repeated runs + analysis)

Use `scripts/run_snn.py` to repeat a config and build a saved analysis bundle:

```bash
python scripts/run_snn.py \
  --config configs/snn_test_run.yaml \
  --save-dir simulations/snn_test_run \
  --n-repeats 4
```

This runs `Repeater` + `StageSNNSimulation`, writes `data/spikes_*.npy`,
`clusters.npy`, and saves attractor analysis to `analysis/`.

Parallel execution is available:

```bash
python scripts/run_snn.py \
  --config configs/snn_test_run.yaml \
  --save-dir simulations/snn_test_run \
  --n-repeats 4 \
  --parallel \
  --executor process
```

#### Perturbed spiking network runs

Use `scripts/run_perturbed_snn.py` to run repeated simulations with
vectorial perturbations defined in the config:

```bash
python scripts/run_perturbed_snn.py \
  --config configs/perturbed/default_snn_params_with_perturbation.yaml \
  --save-dir simulations/perturbed_snn \
  --n-repeats 4
```

Perturbations are generated in cluster space via `VectorialPerturbation`,
logged with summary stats, and saved to `metadata/perturbations.npz` for
reproducibility.

#### Mean-field stagers

`FullMeanFieldStager` runs fixed-point solves from random initial conditions,
and `ReducedMeanFieldStager` computes 2D potential landscapes for selected
populations.

```python
from neuro_mod.execution.stagers.mean_field import FullMeanFieldStager, ReducedMeanFieldStager

full = FullMeanFieldStager("configs/2_cluster_mf.yaml", random_seed=7)
full_outputs = full.run(n_runs=50)

reduced = ReducedMeanFieldStager("configs/2_cluster_mf.yaml", random_seed=7)
reduced_outputs = reduced.run(focus_pops=[0, 1], grid_density=0.5, grid_lims=(0.0, 60.0))
```

#### Sweep stagers (parameter sweeps)

Sweep runners are base classes that wrap stagers and iterate a parameter list.
To use them, subclass and implement `_step`, `summary`, and plotting.

```python
from neuro_mod.execution.sweep.spiking_net import SNNBaseSweepRunner

class SpikeCountSweep(SNNBaseSweepRunner):
    def _step(self, param, idx, sweep_param, **kwargs):
        outputs = self._sweep_object.run(**kwargs)
        return int(outputs["spikes"].sum())

    def _store(self, *args, **kwargs):
        pass

    def summary(self, results, sweep_params):
        print(list(zip(sweep_params, results)))

    def _summary_plot(self, *args, **kwargs):
        pass

    def summarize_repeated_run(self, *args, **kwargs):
        pass

sweep = SpikeCountSweep()
sweep.execute(
    main_dir="simulations/sweep_demo",
    baseline_params="configs/snn_test_run.yaml",
    param=["external_currents", "nu_ext_baseline"],
    sweep_params=[5.0, 7.5, 10.0],
    param_idx=0,
)
```

Each sweep step writes a config snapshot under `configs/` in the sweep output
directory so runs are reproducible.

Parallel sweeps are supported as well:

```python
sweep.execute(
    main_dir="simulations/sweep_demo",
    baseline_params="configs/snn_test_run.yaml",
    param=["external_currents", "nu_ext_baseline"],
    sweep_params=[5.0, 7.5, 10.0],
    param_idx=0,
    parallel=True,
    executor="process",
)
```

Note: parallel runs keep deterministic outputs given fixed seeds, but log line
ordering can interleave across workers.

#### Experiment Pipeline

The `Pipeline` class provides a unified interface for running experiments with
full reproducibility and logging. It orchestrates the workflow:
**Simulation -> Processing -> Analysis -> DataFrame -> Plotting (seaborn)**.

```python
from pathlib import Path
from neuro_mod.pipeline import Pipeline, PipelineConfig, ExecutionMode, SeabornPlotter
from neuro_mod.execution.stagers import StageSNNSimulation
from neuro_mod.core.spiking_net.processing import SNNProcessor
from neuro_mod.core.spiking_net.analysis import SNNAnalyzer

# Define factories for each pipeline phase
pipeline = Pipeline(
    simulator_factory=lambda seed: StageSNNSimulation("configs/snn_test_run.yaml", random_seed=seed),
    processor_factory=lambda raw: SNNProcessor(raw["spikes_path"]),
    analyzer_factory=SNNAnalyzer,
    plotter=SeabornPlotter(),
)

# Run a swept repeated experiment
result = pipeline.run(PipelineConfig(
    mode=ExecutionMode.SWEEP_REPEATED,
    n_repeats=10,
    sweep_param="arousal.level",
    sweep_values=[0.1, 0.3, 0.5, 0.7, 0.9],
    parallel=True,
    save_dir=Path("experiments/arousal_sweep"),
))

# Access results
df = result.dataframes["aggregated"]  # DataFrame with sweep_value, repeat columns
metrics = result.metrics["aggregated"]
```

**Execution modes:**
- `SINGLE`: Single simulation run
- `REPEATED`: Multiple runs with seed management
- `SWEEP`: Parameter variations (one run per value)
- `SWEEP_REPEATED`: Factorial design (sweep x repeats)

**Reproducibility features:**
- Deterministic seed generation from `base_seed`
- Config persistence to JSON
- Git commit hash tracking
- Timestamps and duration logging

**Logging:**
- Configurable log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- File logging to `save_dir/logs/pipeline.log`
- Progress callbacks for UI integration

#### Perturbation config shape

Perturbations are configured per parameter under `perturbation` in YAML.
Each block defines vectors, coefficients, optional time dependence, and an
optional RNG seed:

```yaml
perturbation:
  rate:
    vectors:
      - [1, 1, -1]
      - [-1, 1, 1]
      - [1, -1, 1]
    involved_clusters:
      - [0, 1, 2]
      - [1, 2, 3]
      - [2, 3, 4]
    seed: 256
    params: [0.3, 0.02, 0.2]
    time_dependence:
      shape: hat
      onset_time: 0
      offset_time:
  arousal_level:
    vectors:
      - [1, 1, -1]
      - [-1, 1, 1]
      - [1, -1, 1]
    involved_clusters:
      - [0, 1, 2]
      - [1, 2, 3]
      - [2, 3, 4]
    seed: 256
    params: [0.3, 0.02, 0.2]
    time_dependence:
      shape: hat
      onset_time: 0
      offset_time:
```

Supported targets include `rate`, `j_baseline`, `j_potentiated`, `j_ext`,
`threshold`, `tau_membrane`, `tau_synaptic`, `tau_refractory`, and arousal
parameters (`arousal_level`, `arousal_L`, `arousal_x_0`, `arousal_k`, `arousal_M`).

#### Spiking-net processing and analysis

The `SNNProcessor` transforms raw spike data into attractor representations,
and `SNNAnalyzer` computes statistics and transition matrices.

```python
from neuro_mod.core.spiking_net.processing import SNNProcessor
from neuro_mod.core.spiking_net.analysis import SNNAnalyzer

# Process raw spikes into attractors
processor = SNNProcessor(
    spikes_path="simulations/snn_test_run/data",
    clusters_path="simulations/snn_test_run/clusters.npy",
    dt=0.5e-3,
)
attractors_data = processor.process()
processor.save("simulations/snn_test_run/analysis")

# Analyze the processed data
analyzer = SNNAnalyzer("simulations/snn_test_run/analysis")
print(f"Number of attractors: {analyzer.get_num_states()}")
print(f"Transition matrix shape: {analyzer.get_transition_matrix().shape}")

# Convert to DataFrame for further analysis
df = analyzer.to_dataframe()
metrics = analyzer.get_summary_metrics()
```

The helper script `scripts/load_snn_analysis.py` shows how to load a saved
analysis bundle.

---

### Use the mean-field model directly

The `LIFMeanField` class exposes population-level equations and utilities for
finding fixed points and their stability:

```python
import numpy as np
from neuro_mod.core.mean_field.core import LIFMeanField

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
