"""Base stager definitions for simulation execution."""

from io import BytesIO
from pathlib import Path
from abc import abstractmethod, ABC
import shutil
import yaml
import numpy as np

from neuro_mod.execution.helpers import Logger

from neuro_mod.clustering import setup_matrices


class _Stager(ABC):

    n_neurons: int
    n_excitatory: int
    n_inhibitory: int
    duration_sec: float
    delta_t: float
    mechanism: str
    p_mat: np.ndarray
    j_mat: np.ndarray
    cluster_vec: np.ndarray
    types: dict
    main_dir: Path = Path()
    data_dir: Path = Path()
    plots_dir: Path = Path()

    def __init__(
            self,
            config: Path | str | bytes,
            random_seed: int = None,
            logger: Logger | None = None,
            **kwargs
    ):
        self.config = config
        self.logger = logger or Logger(name=self.__class__.__name__)
        self.settings = self._reader('settings')

        if self.settings["save"]:
            self._set_dirs()
        if random_seed is None:
            random_seed = self.settings["random_seed"]
        self.rng = np.random.default_rng(random_seed)
        self.network_params = self._reader('architecture', 'network')
        self.clusters_params = self._reader('architecture', 'clusters')
        self.arousal_params = self._reader('arousal')
        self.arousal_denom = self._get_arousal_denom()

        for key, value in self._reader('init_params').items():
            setattr(self, key, value)
        j_perturbation = kwargs.pop('j_perturbation', None)
        self._setup_clustered_matrices(j_perturbation=j_perturbation)

    @classmethod
    def from_dict(cls, params: dict):
        yml_string = yaml.dump(params, default_flow_style=False)
        d_bytes = yml_string.encode('utf-8')
        return cls(d_bytes)

    def _reader(self, *keys: str):
        if isinstance(self.config, bytes):
            stream = BytesIO(self.config)
        else:
            stream = open(self.config, 'r')
        with stream:
            yml = yaml.safe_load(stream)
            out = yml
            for key in keys:
                out = out[key]
            return out

    def _set_dirs(self):
        parent_dir = Path(self.settings['save_dir'])
        self.main_dir = parent_dir / self.settings['sim_name']
        self.main_dir.mkdir(parents=True, exist_ok=True)

        self.data_dir = self.main_dir.joinpath('data')
        self.data_dir.mkdir(exist_ok=True)
        self.plots_dir = self.main_dir.joinpath('plots')
        self.plots_dir.mkdir(exist_ok=True)
        self.metadata_dir = self.main_dir.joinpath('metadata')
        self.metadata_dir.mkdir(exist_ok=True)
        if self.logger.file_path is None:
            self.logger.attach_file(str(self.metadata_dir / "run.log"))

    def _setup_clustered_matrices(self, *args, **kwargs):
        j_baseline = self.clusters_params.pop('j_baseline')
        j_baseline[0] *= (1 - self._get_arousal_jee())
        params = {
            "n_neurons": self.n_neurons,
            "n_excitatory": self.n_excitatory,
            "j_baseline": j_baseline,
            **self.clusters_params
        }
        p, j, b, t = setup_matrices(**params)
        self.p_mat = p
        j_perturbation = kwargs['j_perturbation']
        j_perturbation = np.ones_like(j) if j_perturbation is None else j_perturbation
        self.j_mat = j * j_perturbation
        self.cluster_vec = b
        self.types = t

    def _get_arousal_denom(self):
        k = self.arousal_params["k"]
        x_0 = self.arousal_params["x_0"]
        x = max(1e-6, self.arousal_params["level"])
        C = np.log2(1 / (1 + ((1 - x_0) / x_0) ** (1/k)))

        return 1 + (x ** C - 1) ** k

    def _get_arousal_nu(self):
        M = self.arousal_params["M"]
        z = self.rng.beta(10, 10, size=[self.n_neurons])
        return M * z / self.arousal_denom

    def _get_arousal_jee(self):
        L = self.arousal_params["L"]
        return L / self.arousal_denom

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    def _save(self, **kwargs):
        self.data_file = self.data_dir / "outputs.npz"
        np.savez(self.data_file,
                 **kwargs)
        shutil.copy(self.config, self.main_dir.joinpath('config.yaml'))

    @abstractmethod
    def _plot(self, *args, **kwargs):
        pass

    def execute(self,
                plot_arg: str,
                save_outputs: list[str] = None,
                *args, **kwargs):
        self.logger.info("Starting execution.")
        outputs = self.run(*args, **kwargs)
        if self.settings['save']:
            save_outputs = save_outputs if save_outputs is not None else outputs.keys()
            to_save = {k: outputs[k] for k in save_outputs if k in outputs}
            self._save(**to_save)
            self.logger.info(f"Saved outputs to {self.data_dir}.")
        if self.settings['plot']:
            self._plot(outputs.get(plot_arg))
            self.logger.info(f"Saved plot to {self.plots_dir}.")
        self.logger.info("Execution complete.")

    @staticmethod
    def _project_to_cluster_space(
            original: np.ndarray[float] | list[float],
            n_populations: int,
            n_excitatory: int,
    ) -> np.ndarray:
        if np.asarray(original).shape == (n_populations,):
            return original
        arr = np.zeros(n_populations)
        arr[:n_excitatory] = original[0]
        arr[n_excitatory:] = original[1]
        return arr
