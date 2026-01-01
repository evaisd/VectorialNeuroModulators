"""Sweep runner base class for spiking network simulations."""

from abc import abstractmethod, ABC
from neuro_mod.execution.sweep._base import _BaseSweepRunner
from neuro_mod.execution import StageSNNSimulation


class SNNBaseSweepRunner(_BaseSweepRunner, ABC):
    """Base sweep runner for spiking network stagers."""
    def __init__(self):
        """Initialize the sweep runner with the SNN stager."""
        super().__init__()
        self._stager = StageSNNSimulation
        self._sweep_object: StageSNNSimulation | None = None

    @abstractmethod
    def _step(self, *args, **kwargs):
        pass
