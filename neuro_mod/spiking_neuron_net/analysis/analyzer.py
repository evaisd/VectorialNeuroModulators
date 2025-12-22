
from pathlib import Path
import numpy as np


class Analyzer:

    def __init__(
            self,
            spikes: np.ndarray[bool] | str | Path,
            clusters: np.ndarray[int] | str | Path,
            dt: float = .5e-3
                 ):
        self.spikes = np.asarray(spikes)
        self.clusters = np.asarray(clusters)
        self.dt = dt
        self.total_time = len(spikes) * self.dt
        self._attractor_map = {}
        self.transition_matrix = self._get_transition_matrix()

    def get_neuron_spike_rate(self,
                              *args,
                              **kwargs):
        pass

    def get_cluster_spike_rate(self,
                               *args,
                               **kwargs):
        pass

    def get_attractors_matrix(self,
                              *args,
                              **kwargs):
        pass

    def get_attractor_idx(
            self,
            *clusters: int,
            **kwargs
    ) -> int:
        pass

    def get_mean_lifespan(
            self,
            attractor_idx: int,
            *args,
            **kwargs) -> float:
        pass

    def get_attractor_prob(self,
                           attractor_idx: int,
                           *args,
                           **kwargs) -> float:
        pass

    def get_transition_prob(self,
                            from_attractor_idx: int,
                            to_attractor_idx: int,
                            *args,
                            **kwargs) -> float:
        pass

    def _get_transition_matrix(self,
                               *args,
                               **kwargs) -> np.ndarray:
        pass