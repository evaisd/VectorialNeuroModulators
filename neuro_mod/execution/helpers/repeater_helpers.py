"""Helper functions for the repeatable simulation runner."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Any
from concurrent.futures import ProcessPoolExecutor
import functools
import inspect
import pickle
import traceback

import numpy as np

from neuro_mod.execution.helpers.logger import Logger


def default_run_fn(stager: Any) -> dict:
    """Default run callback for a stager."""
    return stager.run()


def default_save_outputs(
    outputs: dict,
    idx: int,
    stager: Any,
    data_dir: Path,
    plot_dir: Path,
    save_dir: Path,
) -> None:
    """Default output saver used by the repeater."""
    spikes = outputs.get("spikes") if isinstance(outputs, dict) else None
    if spikes is not None:
        np.save(data_dir / f"spikes_{idx}.npy", spikes)
        if hasattr(stager, "_plot"):
            stager._plot(spikes, plt_path=plot_dir / f"spikes_{idx}.png")

    clusters = outputs.get("clusters") if isinstance(outputs, dict) else None
    if clusters is not None and idx == 0:
        np.save(save_dir / "clusters.npy", clusters)


def build_logger_from_settings(settings: dict) -> Logger:
    """Create a Logger instance from serialized settings."""
    return Logger(**settings)


def call_stager_factory(stager_factory: Callable[..., Any], seed: int, logger: Logger) -> Any:
    """Call a stager factory with an optional logger kwarg."""
    try:
        signature = inspect.signature(stager_factory)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        params = signature.parameters
        if "logger" in params or any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
        ):
            return stager_factory(seed, logger=logger)
    return stager_factory(seed)


def process_step(
    seed: int,
    idx: int,
    n_repeats: int,
    stager_factory: Callable[..., Any],
    run_fn: Callable[[Any], dict],
    save_fn: Callable[[dict, int, Any], None],
    logger_settings: dict,
) -> None:
    """Execute one repeat step inside a process worker."""
    logger = build_logger_from_settings(logger_settings)
    logger.info(f"Starting repeat {idx + 1}/{n_repeats}.")
    try:
        stager = call_stager_factory(stager_factory, seed, logger)
        outputs = run_fn(stager)
        save_fn(outputs, idx, stager)
    except Exception:
        logger.error(
            f"Repeat {idx + 1}/{n_repeats} failed.\n{traceback.format_exc()}"
        )
        raise
    logger.info(f"Finished repeat {idx + 1}/{n_repeats}.")


def validate_process_pickling(
    *,
    stager_factory: Callable[..., Any],
    run_fn: Callable[[Any], dict],
    save_fn: Callable[[dict, int, Any], None],
) -> None:
    """Ensure callables are pickleable for process execution."""
    for name, obj in (("stager_factory", stager_factory), ("run_fn", run_fn), ("save_fn", save_fn)):
        try:
            pickle.dumps(obj)
        except Exception as exc:
            raise ValueError(
                "Process execution requires pickleable callables. "
                f"{name} is not pickleable: {exc}"
            ) from exc


def build_process_save_fn(
    save_fn: Callable[[dict, int, Any], None],
    *,
    use_default: bool,
    data_dir: Path,
    plot_dir: Path,
    save_dir: Path,
) -> Callable[[dict, int, Any], None]:
    """Return a pickleable save function for process execution."""
    if use_default:
        return functools.partial(
            default_save_outputs,
            data_dir=data_dir,
            plot_dir=plot_dir,
            save_dir=save_dir,
        )
    return save_fn


def run_process_pool(
    *,
    seeds: np.ndarray,
    n_repeats: int,
    max_workers: int | None,
    stager_factory: Callable[..., Any],
    run_fn: Callable[[Any], dict],
    save_fn: Callable[[dict, int, Any], None],
    logger_settings: dict,
):
    """Run repeats using a process pool and return the futures mapping."""
    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures = {
        executor.submit(
            process_step,
            seed,
            idx,
            n_repeats,
            stager_factory,
            run_fn,
            save_fn,
            logger_settings,
        ): (idx, seed)
        for idx, seed in enumerate(seeds)
    }
    return executor, futures
