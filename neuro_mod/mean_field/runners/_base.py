
from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from neuro_mod.mean_field.auxiliary.misc import AttrDict

___all__ = [
    'SweepRunner',
    'SimRunner',
]

class SimRunner(ABC, AttrDict):

    def __init__(self, **params):
        super().__init__(**params)
        pass

    @abstractmethod
    def _gen_params(self, **params):
        pass

    @abstractmethod
    def _mft_params(self, **params):
        pass

    @abstractmethod
    def _settings(self):
        pass

    @abstractmethod
    def run_effective_rates_on_grid(self, focus_pops: list[int], *nu_vecs: np.ndarray):
        pass

    @abstractmethod
    def run(self, nu_init, *args, **kwargs):
        pass


class SweepRunner(ABC):

    def __init__(self, sweep_param: str, mode: str = 'new', **params):
        self.sweep_param = sweep_param
        self.mode = mode
        self.params = params.copy()
        self._params_origin = deepcopy(params)
        pass

    def _get_params(self, sweep_param_val: float) -> dict:
        sweep_param = {self.sweep_param: sweep_param_val}
        params = {**self.params, **sweep_param}
        return params

    @abstractmethod
    def run(self, sweep_params: np.ndarray | list[float], nu_init: np.ndarray, *args, **kwargs):
        pass

    @abstractmethod
    def _single_run(self, sweep_param_val: float, nu_init: np.ndarray, *args, **kwargs):
        pass

    def _apply_transformation(self, param_name: str, transformation_matrix: np.ndarray):
        self.params[param_name] = transformation_matrix.T @ self.params[param_name] @ transformation_matrix

    def _shift_params(self, param_name: str, shift: float):
        self.params[param_name] += shift

    def _reset_params(self):
        self.params = deepcopy(self._params_origin)
        