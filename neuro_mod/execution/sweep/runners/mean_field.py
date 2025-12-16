
from abc import ABC
from neuro_mod.execution.sweep._base import _BaseSweepRunner
from neuro_mod.execution.stagers.mean_field import FullMeanFieldStager, ReducedMeanFieldStager

__all__ = [
    'FullMeanFieldBaseSweepRunner',
    'ReducedMeanFieldSweepRunner',
]


class FullMeanFieldBaseSweepRunner(_BaseSweepRunner, ABC):

    def __init__(self):
        super().__init__()
        self._stager = FullMeanFieldStager
        self._sweep_object: FullMeanFieldStager | None = None

    def _step(self, param: str, sweep_param: float, nu_init, *args, **kwargs):
        pass


class ReducedMeanFieldSweepRunner(_BaseSweepRunner, ABC):

    def __init__(self):
        super().__init__()
        self._stager = ReducedMeanFieldStager
        self._sweep_object: ReducedMeanFieldStager | None = None

    def _step(
            self,
            focus_pops: list[int],
            grid_density: float,
            grid_lims: tuple[float, float] | list[tuple[float, float]],
            *args,
            **kwargs):
        pass
