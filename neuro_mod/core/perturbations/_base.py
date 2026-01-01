"""Base interfaces for perturbation generators."""

from abc import abstractmethod, ABC
import numpy as np


class BasePerturbator(ABC):
    """Abstract base class for perturbation generators."""

    def __init__(self,
                 *args,
                 **kwargs):
        """Initialize the perturbator.

        Args:
            *args: Ignored positional arguments for compatibility.
            **kwargs: Optional keyword arguments. Supports `rng` to pass a
                NumPy random generator.
        """
        self.rng = kwargs.pop('rng', np.random.default_rng(256))

    @abstractmethod
    def get_perturbation(self,
                         *params,
                         **kwargs) -> np.ndarray:
        """Compute a perturbation matrix or vector from parameters.

        Args:
            *params: Perturbation parameters (implementation-specific).
            **kwargs: Optional keyword arguments.

        Returns:
            A NumPy array representing the perturbation.
        """
        pass
