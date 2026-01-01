"""Helper functions for sweep runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from concurrent.futures import ProcessPoolExecutor
import copy
import pickle

import yaml

from neuro_mod.execution.helpers.logger import Logger


def build_logger_from_settings(settings: dict) -> Logger:
    """Create a Logger instance from serialized settings."""
    return Logger(**settings)


def validate_process_pickling(*, runner_cls: type, step_kwargs: dict) -> None:
    """Ensure the runner class and step kwargs are pickleable."""
    for name, obj in (("runner_cls", runner_cls), ("step_kwargs", step_kwargs)):
        try:
            pickle.dumps(obj)
        except Exception as exc:
            raise ValueError(
                "Process execution requires pickleable inputs. "
                f"{name} is not pickleable: {exc}"
            ) from exc


def run_sweep_step(
    *,
    runner_cls: type,
    baseline_params: dict,
    param: str | list[str],
    sweep_param: Any,
    param_idx: int | None,
    idx: int,
    total_steps: int,
    main_dir: Path | str,
    logger_settings: dict,
    step_kwargs: dict,
    save_outputs: list[str] | None,
):
    """Execute a single sweep step in an isolated runner instance."""
    logger = build_logger_from_settings(logger_settings)
    runner = runner_cls()
    runner.logger = logger
    runner._save_outputs = save_outputs
    runner.set_dirs(main_dir)

    runner._baseline_params = copy.deepcopy(baseline_params)
    perturbation = "perturbation" in runner._baseline_params
    if perturbation:
        runner._init_perturbator()

    param_list = param if isinstance(param, list) else [param]
    runner._modify_params(runner._baseline_params, param_list, sweep_param, param_idx)

    init_kwargs = {}
    if perturbation:
        perturbations = runner._get_perturbation()
        init_kwargs = {
            f"{name}_perturbation": value
            for name, value in perturbations.items()
        }
        runner._log_perturbation_summary(perturbations)
        runner._save_perturbations(perturbations, idx)

    runner._sweep_object = runner._stager.from_dict(runner._baseline_params, **init_kwargs)
    if hasattr(runner._sweep_object, "logger"):
        runner._sweep_object.logger = runner.logger

    logger.info(f"Running sweep step {idx + 1}/{total_steps} for {' '.join(param_list)}.")
    result = runner._step(param_list, idx, sweep_param, **step_kwargs)

    config_path = runner._dirs["configs"].joinpath(f"{idx}.yaml")
    with open(config_path, "w") as f:
        yaml.dump(runner._baseline_params, f, sort_keys=False)

    return result


def run_process_pool(
    *,
    runner_cls: type,
    baseline_params: dict,
    param: str | list[str],
    sweep_params: list,
    param_idx: int | None,
    main_dir: Path | str,
    max_workers: int | None,
    logger_settings: dict,
    step_kwargs: dict,
    save_outputs: list[str] | None,
):
    """Submit sweep steps to a process pool and return the executor/futures."""
    total_steps = len(sweep_params)
    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures = {
        executor.submit(
            run_sweep_step,
            runner_cls=runner_cls,
            baseline_params=baseline_params,
            param=param,
            sweep_param=sweep_param,
            param_idx=param_idx,
            idx=idx,
            total_steps=total_steps,
            main_dir=main_dir,
            logger_settings=logger_settings,
            step_kwargs=step_kwargs,
            save_outputs=save_outputs,
        ): idx
        for idx, sweep_param in enumerate(sweep_params)
    }
    return executor, futures
