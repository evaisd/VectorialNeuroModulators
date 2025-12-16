
import yaml
from abc import ABC, abstractmethod
from pathlib import Path

from neuro_mod.execution.stagers._base import _Stager


class _BaseSweepRunner(ABC):

    def __init__(self, *args, **kwargs):
        self._baseline_params = None
        self._stager: type[_Stager] | None = None
        self._sweep_object: _Stager | None = None
        self._save_outputs: list[str] | None = kwargs.get('save_outputs', None)
        self._dirs = {
            "main": None,
            "data": None,
            "plots": None,
            "configs": None,
        }

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

    def _modify_params(self,
                       param: list[str],
                       param_val: float):
        d = self._baseline_params
        *prefix, last = param
        for key in prefix:
            d = d[key]  # walks down the nesting
        d[last] = float(param_val)

    def execute(self,
                main_dir,
                baseline_params: dict | str,
                param: str | list[str],
                sweep_params: list,
                *args,
                **kwargs):
        self.set_dirs(main_dir)
        self._read_baseline_params(baseline_params)
        param = param if isinstance(param, list) else [param]
        results = []
        for sweep_param in sweep_params:
            self._modify_params(param, sweep_param)
            self._sweep_object = self._stager.from_dict(self._baseline_params)
            results.append(self._step(param, sweep_param, **kwargs))
            config_path = self._dirs['configs'].joinpath(f"{sweep_param:.2f}.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(self._baseline_params, f, sort_keys=False)
        self.summary(results, sweep_params)

    @abstractmethod
    def _store(self, *args, **kwargs):
        pass

    def set_dirs(self, main_dir):
        self._dirs['main'] = Path(main_dir)
        self._dirs['data'] = Path(main_dir).joinpath("data")
        self._dirs['plots'] = Path(main_dir).joinpath("plots")
        self._dirs['configs'] = Path(main_dir).joinpath("configs")
        _ = [d.mkdir(parents=True, exist_ok=True) for d in self._dirs.values()]

    @abstractmethod
    def summary(self, *args, **kwargs):
        pass

    @abstractmethod
    def _summary_plot(self, *args, **kwargs):
        pass

    def repeat(self,
               directory: Path,
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
        self._summarize_repeated_run(directory, sweep_params)

    @abstractmethod
    def _summarize_repeated_run(self, *args, **kwargs):
        pass

