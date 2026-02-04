"""Base stager definitions for simulation execution."""

from io import BytesIO
from pathlib import Path
from abc import abstractmethod, ABC
import shutil
import yaml
import numpy as np

from neuro_mod.execution.helpers.logger import Logger
from neuro_mod.core.clustering import setup_matrices
from neuro_mod.core.perturbations.vectorial import VectorialPerturbation


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

        # Read init_params early (needed for time-dependent perturbations)
        for key, value in self._reader('init_params').items():
            setattr(self, key, value)

        self.network_params = self._reader('architecture', 'network')
        self.clusters_params = self._reader('architecture', 'clusters')
        self.arousal_params = self._reader('arousal')

        # Extract perturbations from kwargs, then merge with config-based perturbations
        kwargs_perturbations = self._extract_perturbations(kwargs)
        config_perturbations = self._generate_perturbations_from_config()
        # kwargs perturbations override config perturbations
        self._init_perturbations = {**config_perturbations, **kwargs_perturbations}

        self._apply_arousal_perturbations(self._init_perturbations)
        self._validate_arousal_level()
        self.arousal_denom = self._get_arousal_denom()

        j_perturbation = self._init_perturbations.get('j')
        j_baseline_perturbation = self._init_perturbations.get('j_baseline')
        j_potentiated_perturbation = self._init_perturbations.get('j_potentiated')
        self._setup_clustered_matrices(
            j_perturbation=j_perturbation,
            j_baseline_perturbation=j_baseline_perturbation,
            j_potentiated_perturbation=j_potentiated_perturbation,
        )

    @classmethod
    def from_dict(cls, params: dict, **kwargs):
        yml_string = yaml.dump(params, default_flow_style=False)
        d_bytes = yml_string.encode('utf-8')
        return cls(d_bytes, **kwargs)

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
        clusters_params = dict(self.clusters_params)
        j_baseline = np.asarray(clusters_params.pop('j_baseline'), dtype=float)
        j_potentiated = np.asarray(clusters_params.pop('j_potentiated'), dtype=float)
        j_baseline[0] *= (1 - self._get_arousal_jee())
        params = {
            "n_neurons": self.n_neurons,
            "n_excitatory": self.n_excitatory,
            "j_baseline": j_baseline,
            "j_potentiated": j_potentiated,
            **clusters_params
        }
        p, j, b, t = setup_matrices(**params)
        self.p_mat = p
        n_clusters = clusters_params["n_clusters"]
        j_baseline_perturbation = kwargs.get('j_baseline_perturbation')
        j_potentiated_perturbation = kwargs.get('j_potentiated_perturbation')
        j = self._apply_matrix_perturbation(j, j_baseline_perturbation, as_delta=True)
        j = self._apply_potentiated_perturbation(j, j_potentiated_perturbation, n_clusters, as_delta=True)
        j_perturbation = kwargs.get('j_perturbation')
        j = self._apply_matrix_perturbation(j, j_perturbation, as_delta=False)
        self.j_mat = j
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

    @staticmethod
    def _extract_perturbations(kwargs: dict) -> dict:
        """Extract perturbation overrides from keyword arguments.

        Args:
            kwargs: Keyword arguments passed to the stager.

        Returns:
            Dictionary of perturbation arrays keyed by parameter name.
        """
        perturbations = dict(kwargs.pop("perturbations", {}))
        suffix = "_perturbation"
        for key in list(kwargs.keys()):
            if key.endswith(suffix):
                perturbations[key[:-len(suffix)]] = kwargs.pop(key)
        return perturbations

    def _generate_perturbations_from_config(self) -> dict:
        """Generate perturbation arrays from the YAML config.

        Reads the 'perturbation' section from the config file and generates
        perturbation arrays using VectorialPerturbation.

        Returns:
            Dictionary of perturbation arrays keyed by target name.
        """
        try:
            perturbation_config = self._reader('perturbation')
        except KeyError:
            # No perturbation section in config
            return {}

        if not perturbation_config:
            return {}

        perturbations = {}
        for name, cfg in perturbation_config.items():
            if not isinstance(cfg, dict) or "params" not in cfg:
                continue

            perturbator = self._build_perturbator(cfg)
            coeffs = np.asarray(cfg["params"], dtype=float)

            # Handle time-dependent perturbations
            time_vec = self._get_time_vector(cfg)
            if time_vec is not None:
                coeffs = np.outer(coeffs, time_vec)

            values = perturbator.get_perturbation(*coeffs)
            perturbations[name] = values

            self.logger.debug(
                f"Perturbation {name}: shape={np.asarray(values).shape} "
                f"mean={np.mean(values):.4f}"
            )

        return perturbations

    def _build_perturbator(self, cfg: dict) -> VectorialPerturbation:
        """Build a VectorialPerturbation from a perturbation config block.

        Args:
            cfg: Perturbation configuration dictionary with keys:
                - vectors: Optional list of basis vectors
                - involved_clusters: Optional cluster indices per vector
                - seed: Random seed for vector generation
                - Other VectorialPerturbation kwargs

        Returns:
            Configured VectorialPerturbation instance.
        """
        cfg = dict(cfg)  # Don't mutate original
        vectors = cfg.pop("vectors", [])
        cfg.pop("params", None)  # Used separately
        cfg.pop("time_dependence", None)  # Used separately
        seed = cfg.pop("seed", None)

        # Use stager's RNG if no seed specified, otherwise create new RNG
        rng = self.rng if seed is None else np.random.default_rng(seed)

        # Get length from cluster config
        length = self.clusters_params.get("total_pops", self.clusters_params.get("n_clusters", 0) * 2 + 2)

        params = {
            **cfg,
            "rng": rng,
            "length": length,
        }
        return VectorialPerturbation(*vectors, **params)

    def _get_time_vector(self, cfg: dict) -> np.ndarray | None:
        """Build a time mask vector from perturbation config.

        Args:
            cfg: Perturbation configuration dictionary.

        Returns:
            Time mask vector or None if no time dependence configured.
        """
        time_dependence = cfg.get("time_dependence")
        if not time_dependence or "shape" not in time_dependence:
            return None

        n_steps = int(self.duration_sec / self.delta_t)
        time_vec = np.zeros(n_steps)

        onset = int(time_dependence.get("onset_time", 0) / self.delta_t)
        offset = time_dependence.get("offset_time")
        offset = None if offset is None else int(offset / self.delta_t)

        time_vec[slice(onset, offset)] = 1
        return time_vec

    def _apply_arousal_perturbations(self, perturbations: dict):
        """Apply arousal perturbations to arousal parameters.

        Args:
            perturbations: Mapping of perturbation names to values.
        """
        if not perturbations:
            return
        for key in ("level", "L", "x_0", "k", "M"):
            pert = perturbations.get(f"arousal_{key}")
            if pert is None:
                continue
            if np.ndim(pert) == 0:
                self.arousal_params[key] += float(pert)
            else:
                n_pops = self.clusters_params["total_pops"]
                pert_vec = self._coerce_cluster_vector(pert, n_pops)
                self.arousal_params[key] += float(np.mean(pert_vec))

    def _validate_arousal_level(self):
        """Clamp arousal level to the [0, 1] range."""
        level = self.arousal_params.get("level")
        if level is None:
            return
        self.arousal_params["level"] = max(0.0, min(1.0, float(level)))

    @staticmethod
    def _coerce_cluster_vector(perturbation, n_populations: int) -> np.ndarray:
        """Validate and coerce a cluster-space perturbation vector.

        Args:
            perturbation: Scalar or population-length vector/array.
            n_populations: Expected number of cluster populations.

        Returns:
            A 1D NumPy array of length `n_populations`.
        """
        arr = np.asarray(perturbation, dtype=float)
        if arr.ndim == 0:
            return np.full(n_populations, float(arr))
        if arr.ndim == 2:
            if arr.shape[0] != n_populations:
                raise ValueError("Perturbation has incompatible population size.")
            arr = arr[:, 0]
        if arr.shape != (n_populations,):
            raise ValueError("Perturbation has incompatible population size.")
        return arr

    def _apply_matrix_perturbation(
            self,
            matrix: np.ndarray,
            perturbation,
            as_delta: bool = True
    ) -> np.ndarray:
        """Apply cluster-space scaling to a matrix.

        Args:
            matrix: Base matrix to perturb.
            perturbation: Scalar, vector, or matrix perturbation.
            as_delta: Whether to interpret values as multiplicative deltas.

        Returns:
            Perturbed matrix.
        """
        if perturbation is None:
            return matrix
        pert = np.asarray(perturbation, dtype=float)
        if pert.shape == matrix.shape:
            return matrix * (1 + pert) if as_delta else matrix * pert
        pert = self._coerce_cluster_vector(pert, matrix.shape[0])
        scale = (1 + pert) if as_delta else pert
        return matrix * scale[:, None]

    @staticmethod
    def _potentiated_mask(n_clusters: int) -> np.ndarray:
        """Build a mask for within-cluster potentiated blocks.

        Args:
            n_clusters: Number of excitatory clusters.

        Returns:
            Boolean mask marking potentiated entries.
        """
        n_pops = 2 * n_clusters + 2
        mask = np.zeros((n_pops, n_pops), dtype=bool)
        for k in range(n_clusters):
            e_idx = k
            i_idx = n_clusters + 1 + k
            mask[e_idx, e_idx] = True
            mask[e_idx, i_idx] = True
            mask[i_idx, e_idx] = True
            mask[i_idx, i_idx] = True
        return mask

    def _apply_potentiated_perturbation(
            self,
            matrix: np.ndarray,
            perturbation,
            n_clusters: int,
            as_delta: bool = True
    ) -> np.ndarray:
        """Apply perturbations only to potentiated synapse blocks.

        Args:
            matrix: Base synaptic matrix.
            perturbation: Scalar or vector perturbation in cluster space.
            n_clusters: Number of excitatory clusters.
            as_delta: Whether to interpret values as multiplicative deltas.

        Returns:
            Perturbed matrix.
        """
        if perturbation is None:
            return matrix
        pert = self._coerce_cluster_vector(perturbation, matrix.shape[0])
        mask = self._potentiated_mask(n_clusters)
        if not mask.any():
            return matrix
        updated = matrix.copy()
        rows, cols = np.where(mask)
        scale = (1 + pert) if as_delta else pert
        updated[rows, cols] *= scale[cols]
        return updated

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    def _save(self, **kwargs):
        self.data_file = self.data_dir / "outputs.npz"
        np.savez(self.data_file,
                 **kwargs)
        config_path = self.main_dir.joinpath('config.yaml')
        if isinstance(self.config, bytes):
            config_path.write_bytes(self.config)
        else:
            shutil.copy(self.config, config_path)

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
