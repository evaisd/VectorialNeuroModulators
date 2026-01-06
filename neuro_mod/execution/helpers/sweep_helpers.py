"""Helper functions for sweep runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from concurrent.futures import ProcessPoolExecutor
import copy
import pickle

from neuro_mod.execution.helpers.logger import build_logger_from_settings


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
    param_list = param if isinstance(param, list) else [param]
    return runner._run_sweep_step(
        param_list,
        sweep_param,
        param_idx,
        idx,
        total_steps,
        step_kwargs=step_kwargs,
        save_outputs=save_outputs,
    )


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
