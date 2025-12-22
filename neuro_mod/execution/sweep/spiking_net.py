
from abc import abstractmethod, ABC
from neuro_mod.execution.sweep._base import _BaseSweepRunner
from neuro_mod.execution.stagers import StageSNNSimulation


class SNNBaseSweepRunner(_BaseSweepRunner, ABC):
    def __init__(self):
        super().__init__()
        self._stager = StageSNNSimulation
        self._sweep_object: StageSNNSimulation | None = None

    @abstractmethod
    def _step(self, *args, **kwargs):
        pass
