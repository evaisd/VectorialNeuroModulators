
from pathlib import Path
from functools import lru_cache

import numpy as np
from neuro_mod.spiking_neuron_net.analysis.logic import activity


class Analyzer:

    _attractor_map: dict
    total_sim_duration_ms: float
    attractors_data: dict
    _clustering_params: dict = dict(kernel_param=20., kernel_type='gaussian')
    _minimal_life_span_ms: float = 20.

    def __init__(
            self,
            spikes_or_folder: str | Path,
            clusters: str | Path = None,
            dt: float = .5e-3
                 ):
        self.spikes_path = Path(spikes_or_folder)
        self.clusters_path = clusters
        self.session_length = len(list(self._files_walker()))
        self.dt = dt
        self.attractors_data = self.get_attractors_data()

    def _read_single(self, path: Path):
        data = np.load(path)
        if path.suffix == '.npz':
            spikes = data['spikes']
        else:
            spikes = data
        if self.clusters_path is None:
            clusters = data['clusters']
        else:
            clusters = np.load(self.clusters_path)
        return spikes, clusters

    def get_data(self):
        spikes = []
        clusters = np.empty(())
        for p in self._files_walker():
            s, c = self._read_single(p)
            spikes.append(s)
            clusters = c
        concat_spikes = np.concatenate(spikes, axis=0)
        self.total_sim_duration_ms = self.dt * concat_spikes.shape[0] * 1e3
        return concat_spikes, clusters

    def _get_data_generator(self):
        return (self._read_single(p) for p in self._files_walker())

    @lru_cache()
    def get_neuron_spike_rate(self,
                              **kwargs):
        spikes, _ = self.get_data()
        params = self._clustering_params.copy()
        params.update(kwargs)
        params.setdefault('dt_ms', self.dt / 1e-3)
        firing_rates = activity.get_firing_rates(
            spikes,
            **params
        )
        return firing_rates.T

    @lru_cache()
    def get_cluster_spike_rate(self,
                               **kwargs):
        spikes, clusters = self.get_data()
        params = self._clustering_params.copy()
        params.update(kwargs)
        params.setdefault('dt_ms', self.dt / 1e-3)
        return activity.get_average_cluster_firing_rate(
            spikes,
            clusters,
            **params
        )

    @lru_cache()
    def get_cluster_activity(
            self,
            **kwargs
    ):
        return activity.get_activity(self.get_cluster_spike_rate(**kwargs))

    @lru_cache()
    def get_attractors_data(self,
                              **kwargs):
        minimal_time_ms = kwargs.pop('minimal_time_ms', self._minimal_life_span_ms)
        activity_matrix = self.get_cluster_activity(**kwargs)
        dt_ms = self.dt * 1e3
        attractors_data = activity.extract_attractors(activity_matrix, minimal_time_ms, dt_ms)
        self._attractor_map = {attractors_data[k]['idx']: k
                               for k
                               in attractors_data.keys()}
        return attractors_data

    def get_attractor_data(self,
                           *idx_or_identity: int | tuple[int, ...],):
        out = {}
        for idx in idx_or_identity:
            if isinstance(idx, tuple):
                out.update({idx: self.attractors_data[idx]})
            else:
                _idx = self._attractor_map[idx]
                out.update({idx: self.attractors_data[_idx]})
        return out

    def get_attractor_idx(
            self,
            *clusters: int,
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
            *idx_or_identities: int | tuple[int, ...],
            **kwargs):
        means, stds = [], []
        for idx in idx_or_identities:
            attractor_data = self.get_attractor_data(idx, **kwargs)[idx]
            starts = np.asarray(attractor_data['starts'])
            ends = np.asarray(attractor_data['ends'])
            diffs = self.dt * 1e3 * (ends - starts)
            means.append(diffs.mean().round(4))
            stds.append(diffs.std().round(4))
        return np.stack(means, axis=0), np.stack(stds, axis=0)

    def get_attractor_prob(self,
                           *idx_or_identity: int | tuple[int, ...],
                           **kwargs) -> np.ndarray:
        probs = []
        for idx in idx_or_identity:
            attractor_data = self.get_attractor_data(idx, **kwargs)[idx]
            duration = attractor_data['total_duration']
            probs.append(duration / self.total_sim_duration_ms)
        return np.stack(probs, axis=0)

    def get_transition_prob(self,
                            idx_or_identity_from: int | tuple[int, ...],
                            idx_or_identity_to: int | tuple[int, ...],
                            *args,
                            **kwargs) -> float:
        transition_matrix = self.get_transition_matrix()
        if isinstance(idx_or_identity_from, tuple):
            idx_or_identity_from = self.get_attractor_idx(*idx_or_identity_from)
        if isinstance(idx_or_identity_to, tuple):
            idx_or_identity_to = self.get_attractor_idx(*idx_or_identity_to)
        return transition_matrix[idx_or_identity_from, idx_or_identity_to]

    @lru_cache()
    def get_transition_matrix(self) -> np.ndarray:
        transition_matrix = activity.get_transition_matrix(
            self.attractors_data,
        )
        return transition_matrix

    def _files_walker(self):
        if self.spikes_path.is_file():
            yield self.spikes_path
        else:
            for p in self.spikes_path.glob("*.np[yz]"):
                yield p

    @lru_cache()
    def get_unique_attractors(self,):
        attractors = set(list(self.attractors_data.keys()))
        return attractors

    def get_sequence_probability(
            self,
            *idx_or_identity: int | tuple[int, ...],
    ):
        if len(idx_or_identity) == 1:
            raise ValueError('Enter atleast two attractors')
        probs = [self.get_transition_prob(att_a, att_b)
                 for att_a, att_b
                 in zip(idx_or_identity[:-1], idx_or_identity[1:])]
        return np.prod(probs)
