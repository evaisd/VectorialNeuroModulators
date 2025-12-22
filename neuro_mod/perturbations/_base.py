
from abc import abstractmethod, ABC
import numpy as np


class BasePerturbator(ABC):

    def __init__(self,
                 *args,
                 **kwargs):
        self.rng = kwargs.pop('rng', np.random.default_rng(256))

    @abstractmethod
    def get_perturbation(self,
                         *params,
                         **kwargs) -> np.ndarray:
        pass
