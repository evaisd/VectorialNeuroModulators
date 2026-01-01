"""Factories for creating stagers with process-safe signatures."""

from __future__ import annotations

from pathlib import Path

from neuro_mod.execution.helpers.logger import Logger
from neuro_mod.execution.stagers import StageSNNSimulation


def make_snn_stager(
    seed: int,
    *,
    config_path: Path | str,
    logger: Logger | None = None,
) -> StageSNNSimulation:
    """Build a spiking network stager with an optional logger."""
    return StageSNNSimulation(config_path, random_seed=seed, logger=logger)


def make_perturbed_snn_stager(
    seed: int,
    *,
    config: dict,
    perturbations: dict,
    logger: Logger | None = None,
) -> StageSNNSimulation:
    """Build a perturbed spiking network stager from a config dict."""
    return StageSNNSimulation.from_dict(
        config,
        random_seed=seed,
        logger=logger,
        perturbations=perturbations,
    )
