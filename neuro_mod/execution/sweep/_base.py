
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from collections.abc import Iterable

from neuro_mod.execution.stagers._base import _Stager
from neuro_mod.perturbations.vectorial import VectorialPerturbation


class _BaseSweepRunner(ABC):

    def __init__(self, *args, **kwargs):
        self._baseline_params = None
        self._stager: type[_Stager] | None = None
        self._sweep_object: _Stager | None = None
        self._save_outputs: list[str] | None = kwargs.get('save_outputs', None)
        self._dirs = {
            "main": Path(),
            "data": Path(),
            "plots": Path(),
            "configs": Path(),
            "metadata": Path(),
        }
        self.perturbator: VectorialPerturbation | None = None

    def _read_baseline_params(self,
                              baseline_params: dict | str):
        if isinstance(baseline_params, dict):
            self._baseline_params = baseline_params
        else:
            import yaml
            with open(baseline_params, 'rb') as f:
                self._baseline_params = yaml.safe_load(f)

    @abstractmethod
    def _step(self, *args, **kwargs):
        pass

    def _modify_params(
            self,
            param: list[str],
            param_val,
    ):
        d = self._baseline_params
        *prefix, last = param

        for key in prefix:
            d = d[key]

        if isinstance(param_val, Iterable) and not isinstance(param_val, (str, bytes)):
            d[last] = param_val
        else:
            d[last] = float(param_val)

    def execute(self,
                main_dir,
                baseline_params: dict | str,
                param: str | list[str],
                sweep_params: list,
                perturbation: bool = False,
                *args,
                **kwargs):
        self.set_dirs(main_dir)
        self._read_baseline_params(baseline_params)
        if perturbation:
            self._init_perturbator()
        param = param if isinstance(param, list) else [param]
        results = []
        for i, sweep_param in enumerate(sweep_params):
            self._modify_params(param, sweep_param)
            self._sweep_object = self._stager.from_dict(self._baseline_params)
            if perturbation:
                self._sweep_object.perturbator = self.perturbator
            results.append(self._step(param, i, sweep_param, **kwargs))
            config_path = self._dirs['configs'].joinpath(f"{i}.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(self._baseline_params, f, sort_keys=False)
        self.summary(results, sweep_params)

    @abstractmethod
    def _store(self, *args, **kwargs):
        pass

    def set_dirs(self, main_dir):
        base = Path(main_dir)
        self._dirs = {
            k: base if k == "main" else base / k
            for k in self._dirs
        }
        _ = [d.mkdir(parents=True, exist_ok=True) for d in self._dirs.values()]

    @abstractmethod
    def summary(self, *args, **kwargs):
        pass

    @abstractmethod
    def _summary_plot(self, *args, **kwargs):
        pass

    def repeat(self,
               directory: Path | str,
               n_trials: int,
               baseline_params: dict | str,
               param: str | list[str],
               sweep_params: list):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        dirs = []
        for i in range(n_trials):
            m_dir = directory.joinpath(f'trial_{i}')
            dirs.append(m_dir)
            self.execute(m_dir,
                         baseline_params=baseline_params,
                         param=param,
                         sweep_params=sweep_params)
        self.summarize_repeated_run(directory, sweep_params)

    @abstractmethod
    def summarize_repeated_run(self, *args, **kwargs):
        pass

    def _init_perturbator(self):
        copied = self._baseline_params['perturbation'].copy()
        length = self._baseline_params['architecture']['clusters']['total_pops']
        vectors = copied.pop('vectors', [])
        params = {
            **copied,
            "length": length,
        }
        self.perturbator = VectorialPerturbation(*vectors, **params)
