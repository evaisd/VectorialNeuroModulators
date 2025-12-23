
from pathlib import Path
from functools import lru_cache

import numpy as np
from neuro_mod.spiking_neuron_net.analysis.logic import activity


class Analyzer:

    def __init__(
            self,
            spikes_or_folder: str | Path,
            dt: float = .5e-3
                 ):
        self.spikes_path = Path(spikes_or_folder)
        self.session_length = len(list(self._files_walker()))
        self.dt = dt
        self.total_time = len(spikes_or_folder) * self.dt
        self._attractor_map = {}
        self.transition_matrix = self._get_transition_matrix()

    @staticmethod
    def _read_single(path: Path):
        data = np.load(path)
        spikes = data['spikes']
        clusters = data['clusters']
        return spikes, clusters

    def get_data(self):
        spikes, clusters = [], []
        for p in self._files_walker():
            s, c = self._read_single(p)
            spikes.append(s)
            clusters.append(c)
        return np.concat(spikes, dtype=bool), np.concat(clusters, dtype=np.uint8)

    def _get_data_generator(self):
        return (self._read_single(p) for p in self._files_walker())

    @lru_cache()
    def get_neuron_spike_rate(self,
                              **kwargs):
        spikes, _ = self.get_data()
        firing_rates = activity.get_firing_rates(
            spikes,
            dt_ms=self.dt / 1e-3,
            kernel_param=kwargs.get('kernel_param', 20.),
            kernel_type=kwargs.get('kernel_type', 'gaussian'),
        )
        return firing_rates

    @lru_cache()
    def get_cluster_spike_rate(self,
                               **kwargs):
        spikes, clusters = self.get_data()
        return activity.get_average_cluster_firing_rate(
            spikes,
            clusters,
            dt_ms=self.dt / 1e-3,
            kernel_param=kwargs.get('kernel_param', 20.),
            kernel_type=kwargs.get('kernel_type', 'gaussian'),
        )

    @lru_cache()
    def get_cluster_activity(
            self,
            **kwargs
    ):
        return activity.get_activity(self.get_cluster_spike_rate(**kwargs))

    def get_attractors_matrix(self,
                              *args,
                              **kwargs):
        pass

    def get_attractor_idx(
            self,
            *clusters: int,
            **kwargs
    ) -> int:
        idx = next(
            (key for
             key, value in
             self._attractor_map.items()
             if value == clusters),
            -1
        )
        if idx == -1:
            raise ValueError('No attractor found')
        return idx

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

    def _files_walker(self):
        if self.spikes_path.is_file():
            yield self.spikes_path
        else:
            for p in self.spikes_path.glob("*.npz"):
                yield p

    @lru_cache()
    def get_unique_attractors(self, **kwargs):
        activity_matrix = self.get_cluster_activity()[:18]
        minimal_length = kwargs.pop('minimal_length', 200)
        smoothed_activity = activity.smooth_cluster_activity(activity_matrix,
                                                             minimal_length=minimal_length,)
        self._attractor_map = activity.get_unique_attractors(smoothed_activity)
        return self._attractor_map