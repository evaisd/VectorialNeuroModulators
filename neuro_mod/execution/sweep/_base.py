"""Base classes for parameter sweep execution."""

import yaml
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy

from neuro_mod.execution.stagers._base import _Stager
from neuro_mod.execution.helpers import Logger
from neuro_mod.execution.helpers.sweep_helpers import (
    run_sweep_step,
    validate_process_pickling,
    run_process_pool,
)
from neuro_mod.core.perturbations.vectorial import VectorialPerturbation


class _BaseSweepRunner(ABC):

    def __init__(self, *args, **kwargs):
        self._baseline_params = None
        self._stager: type[_Stager] | None = None
        self._sweep_object: _Stager | None = None
        self._save_outputs: list[str] | None = kwargs.get('save_outputs', None)
        self.logger: Logger = kwargs.get('logger') or Logger(name=self.__class__.__name__)
        self._dirs = {
            "main": Path(),
            "data": Path(),
            "plots": Path(),
            "configs": Path(),
            "metadata": Path(),
        }
        self.perturbator: VectorialPerturbation | None = None
        self._perturbation_cfg = None

    def _read_baseline_params(self,
                              baseline_params: dict | str):
        if isinstance(baseline_params, dict):
            self._baseline_params = baseline_params
        else:
            import yaml
            with open(baseline_params, 'rb') as f:
                self._baseline_params = yaml.safe_load(f)

    @abstractmethod
    def _step(self,param, idx, sweep_param, *args, **kwargs):
        pass

    def _modify_params(
            self,
            baseline_params: dict,
            param: list[str],
            param_val,
            idx: int = None,
    ):
        d = baseline_params
        *prefix, last = param

        for key in prefix:
            d = d[key]

        if isinstance(param_val, Iterable) and not isinstance(param_val, (str, bytes)):
            pass
        else:
            param_val = float(param_val)
        if isinstance(d[last], Iterable):
            d[last][idx] = param_val
        else:
            d[last] = param_val


    def execute(
                self,
                main_dir,
                baseline_params: dict | str,
                param: str | list[str],
                sweep_params: list,
                param_idx: int = None,
                *args,
                parallel: bool = False,
                max_workers: int | None = None,
                executor: str = "thread",
                **kwargs):
        self.set_dirs(main_dir)
        self._read_baseline_params(baseline_params)
        perturbation = "perturbation" in self._baseline_params.keys()
        param = param if isinstance(param, list) else [param]
        results = []
        total_steps = len(sweep_params)
        self.logger.info(f"Starting sweep with {total_steps} steps.")
        if parallel and total_steps > 1:
            worker_desc = max_workers if max_workers is not None else "default"
            self.logger.info(
                f"Parallel sweep enabled with max_workers={worker_desc} "
                f"using executor={executor}."
            )
            results = self._execute_parallel(
                main_dir=main_dir,
                baseline_params=copy.deepcopy(self._baseline_params),
                param=param,
                sweep_params=sweep_params,
                param_idx=param_idx,
                max_workers=max_workers,
                executor=executor,
                step_kwargs=kwargs,
            )
        else:
            if perturbation:
                self._init_perturbator()
            for i, sweep_param in enumerate(sweep_params):
                self._modify_params(self._baseline_params, param, sweep_param, param_idx)
                init_kwargs = {}
                if perturbation:
                    perturbations = self._get_perturbation()
                    init_kwargs = {
                        f"{name}_perturbation": value
                        for name, value in perturbations.items()
                    }
                    self._log_perturbation_summary(perturbations)
                    self._save_perturbations(perturbations, i)
                self._sweep_object = self._stager.from_dict(self._baseline_params, **init_kwargs)
                if hasattr(self._sweep_object, "logger"):
                    self._sweep_object.logger = self.logger
                self.logger.info(f"Running sweep step {i + 1}/{total_steps} for {' '.join(param)}.")
                results.append(self._step(param, i, sweep_param, **kwargs))
                config_path = self._dirs['configs'].joinpath(f"{i}.yaml")
                with open(config_path, 'w') as f:
                    yaml.dump(self._baseline_params, f, sort_keys=False)
        self.summary(results, sweep_params)
        self.logger.info("Sweep complete.")

    def _execute_parallel(
            self,
            *,
            main_dir: Path | str,
            baseline_params: dict,
            param: list[str],
            sweep_params: list,
            param_idx: int | None,
            max_workers: int | None,
            executor: str,
            step_kwargs: dict,
    ) -> list:
        if executor not in {"thread", "process"}:
            raise ValueError(f"Unknown executor: {executor}")

        logger_settings = {
            "name": self.logger.name,
            "level": self.logger.level,
            "file_path": self.logger.file_path,
            "include_timestamp": self.logger.include_timestamp,
        }
        total_steps = len(sweep_params)
        results = [None] * total_steps
        if executor == "process":
            validate_process_pickling(runner_cls=type(self), step_kwargs=step_kwargs)
            executor, futures = run_process_pool(
                runner_cls=type(self),
                baseline_params=baseline_params,
                param=param,
                sweep_params=sweep_params,
                param_idx=param_idx,
                main_dir=main_dir,
                max_workers=max_workers,
                logger_settings=logger_settings,
                step_kwargs=step_kwargs,
                save_outputs=self._save_outputs,
            )
            with executor:
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()
            return results

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    run_sweep_step,
                    runner_cls=type(self),
                    baseline_params=baseline_params,
                    param=param,
                    sweep_param=sweep_param,
                    param_idx=param_idx,
                    idx=idx,
                    total_steps=total_steps,
                    main_dir=main_dir,
                    logger_settings=logger_settings,
                    step_kwargs=step_kwargs,
                    save_outputs=self._save_outputs,
                ): idx
                for idx, sweep_param in enumerate(sweep_params)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results

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
        if self.logger.file_path is None:
            self.logger.attach_file(str(self._dirs["metadata"] / "sweep.log"))

    @abstractmethod
    def summary(self, *args, **kwargs):
        pass

    @abstractmethod
    def _summary_plot(self, *args, **kwargs):
        pass

    def repeat(
               self,
               directory: Path | str,
               n_trials: int,
               baseline_params: dict | str,
               param: str | list[str],
               sweep_params: list,
               *,
               parallel: bool = False,
               max_workers: int | None = None,
               executor: str = "thread"):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        dirs = []
        for i in range(n_trials):
            m_dir = directory.joinpath(f'trial_{i}')
            dirs.append(m_dir)
            self.execute(
                m_dir,
                baseline_params=baseline_params,
                param=param,
                sweep_params=sweep_params,
                parallel=parallel,
                max_workers=max_workers,
                executor=executor,
            )
        self.summarize_repeated_run(directory, sweep_params)

    @abstractmethod
    def summarize_repeated_run(self, *args, **kwargs):
        pass

    def _init_perturbator(self):
        perturbation_cfg = self._baseline_params['perturbation']
        length = self._baseline_params.get("architecture").get("clusters").get("total_pops")
        self._perturbation_cfg = perturbation_cfg
        if "params" in perturbation_cfg or "vectors" in perturbation_cfg:
            copied = perturbation_cfg.copy()
            vectors = copied.pop('vectors', [])
            copied.pop('params', None)
            copied.pop('time_dependence', None)
            seed = copied.pop('seed', 256)
            copied["rng"] = np.random.default_rng(seed)
            params = {
                **copied,
                "length": length,
            }
            self.perturbator = {"rate": VectorialPerturbation(*vectors, **params)}
            return
        self.perturbator = {}
        for name, cfg in perturbation_cfg.items():
            if not isinstance(cfg, dict):
                continue
            copied = dict(cfg)
            vectors = copied.pop('vectors', [])
            copied.pop('params', None)
            copied.pop('time_dependence', None)
            seed = copied.pop('seed', 256)
            copied["rng"] = np.random.default_rng(seed)
            params = {
                **copied,
                "length": length,
            }
            self.perturbator[name] = VectorialPerturbation(*vectors, **params)

    def _get_perturbation(self):
        if isinstance(self.perturbator, dict) and "rate" in self.perturbator and (
            "params" in self._perturbation_cfg or "vectors" in self._perturbation_cfg
        ):
            config = self._perturbation_cfg
            coeffs = np.asarray(config["params"], dtype=float)
            time_vec = self._get_time_vector(config)
            if time_vec is not None:
                coeffs = np.outer(coeffs, time_vec)
            return {"rate": self.perturbator["rate"].get_perturbation(*coeffs)}

        out = {}
        for name, cfg in self._perturbation_cfg.items():
            if name not in self.perturbator:
                continue
            coeffs = np.asarray(cfg["params"], dtype=float)
            time_vec = self._get_time_vector(cfg)
            if time_vec is not None:
                coeffs = np.outer(coeffs, time_vec)
            out[name] = self.perturbator[name].get_perturbation(*coeffs)
        return out

    def _get_time_vector(self, config: dict) -> np.ndarray | None:
        """Build a time mask vector for perturbations.

        Args:
            config: Perturbation configuration containing time dependence.

        Returns:
            Time mask vector or None if not configured.
        """
        time_dependence = config.get("time_dependence")
        if time_dependence is None or "shape" not in time_dependence:
            return None
        dt = self._baseline_params["init_params"]["delta_t"]
        n_steps = int(self._baseline_params["init_params"]["duration_sec"] // dt)
        time_vec = np.zeros(n_steps)
        onset = int(time_dependence["onset_time"] // dt)
        offset = time_dependence.get("offset_time")
        offset = offset if offset is None else int(offset // dt)
        time_vec[slice(onset, offset)] = 1
        return time_vec

    def _log_perturbation_summary(self, perturbations: dict):
        """Log summary statistics for generated perturbations.

        Args:
            perturbations: Mapping of parameter names to perturbation arrays.
        """
        for name, values in perturbations.items():
            arr = np.asarray(values, dtype=float)
            self.logger.info(
                f"Perturbation {name}: shape={arr.shape} "
                f"mean={arr.mean():.4f} min={arr.min():.4f} max={arr.max():.4f}"
            )

    def _save_perturbations(self, perturbations: dict, idx: int):
        """Persist perturbations for reproducibility.

        Args:
            perturbations: Mapping of parameter names to perturbation arrays.
            idx: Sweep index for naming the output file.
        """
        if not self._dirs.get("metadata"):
            return
        file_path = self._dirs["metadata"] / f"perturbations_{idx}.npz"
        np.savez(file_path, **perturbations)
