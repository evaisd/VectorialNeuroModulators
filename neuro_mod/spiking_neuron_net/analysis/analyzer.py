
from pathlib import Path
from functools import lru_cache
import logging

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
        self.dt = dt
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session_length = len(self._get_sessions())
        self.total_sim_duration_ms = self._get_total_duration_ms()
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

    @lru_cache()
    def _get_sessions(self):
        sessions = []
        clusters_ref = None
        for p in self._files_walker():
            spikes, clusters = self._read_single(p)
            if clusters_ref is None:
                clusters_ref = clusters
            elif clusters.shape != clusters_ref.shape or not np.array_equal(clusters, clusters_ref):
                raise ValueError("Cluster labels differ across simulations.")
            sessions.append((spikes, clusters))
        return tuple(sessions)

    @lru_cache()
    def _get_total_duration_ms(self):
        return self.dt * sum(spikes.shape[0] for spikes, _ in self._get_sessions()) * 1e3

    @staticmethod
    def _aggregate_series(series, axis: int):
        if not series:
            return np.empty(())
        if len(series) == 1:
            return series[0]
        return np.concatenate(series, axis=axis)

    def get_data(self):
        sessions = self._get_sessions()
        if not sessions:
            return np.empty(()), np.empty(())
        spikes = [spikes for spikes, _ in sessions]
        clusters = sessions[0][1]
        concat_spikes = np.concatenate(spikes, axis=0)
        self.total_sim_duration_ms = self._get_total_duration_ms()
        return concat_spikes, clusters

    def _get_data_generator(self):
        return (self._read_single(p) for p in self._files_walker())

    @lru_cache()
    def get_neuron_spike_rate(self,
                              **kwargs):
        params = self._clustering_params.copy()
        params.update(kwargs)
        params.setdefault('dt_ms', self.dt / 1e-3)
        firing_rates = []
        for spikes, _ in self._get_sessions():
            firing_rates.append(
                activity.get_firing_rates(
                    spikes,
                    **params
                ).T
            )
        return self._aggregate_series(firing_rates, axis=1)

    @lru_cache()
    def get_cluster_spike_rate(self,
                               **kwargs):
        firing_rates = self._get_session_cluster_spike_rates(**kwargs)
        return self._aggregate_series(firing_rates, axis=1)

    @lru_cache()
    def get_cluster_activity(
            self,
            **kwargs
    ):
        activity_mats = []
        for cluster_rates in self._get_session_cluster_spike_rates(**kwargs):
            activity_mats.append(activity.get_activity(cluster_rates))
        return self._aggregate_series(activity_mats, axis=1)

    @lru_cache()
    def get_attractors_data(self,
                              **kwargs):
        minimal_time_ms = kwargs.pop('minimal_time_ms', self._minimal_life_span_ms)
        session_attractors = self._get_session_attractors_data(minimal_time_ms, **kwargs)
        session_lengths = self._get_session_lengths_steps(**kwargs)
        self._validate_no_cross_simulation_attractors(session_attractors, session_lengths)
        attractors_data = self._merge_attractors_data(session_attractors)
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
        self.logger.info("get_attractor_data executed successfully.")
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
        session_attractors = self._get_session_attractors_data(self._minimal_life_span_ms)
        return activity.get_transition_matrix_session_aware(
            self.attractors_data,
            session_attractors,
        )

    def _files_walker(self):
        if self.spikes_path.is_file():
            yield self.spikes_path
        else:
            for p in sorted(self.spikes_path.glob("*.np[yz]")):
                yield p

    @lru_cache()
    def get_unique_attractors(self,):
        attractors = set(list(self.attractors_data.keys()))
        return attractors

    @lru_cache()
    def _get_unique_attractor_first_start_steps(
            self,
            minimal_time_ms: float,
            **kwargs,
    ) -> np.ndarray:
        session_attractors = self._get_session_attractors_data(minimal_time_ms, **kwargs)
        session_lengths = self._get_session_lengths_steps(**kwargs)
        offsets = np.cumsum([0] + session_lengths[:-1])
        first_starts = {}
        for session_data, offset in zip(session_attractors, offsets):
            for identity, entry in session_data.items():
                starts = entry.get("starts", [])
                if not starts:
                    continue
                global_start = offset + min(starts)
                prev = first_starts.get(identity)
                if prev is None or global_start < prev:
                    first_starts[identity] = global_start
        if not first_starts:
            return np.empty((0,), dtype=int)
        return np.fromiter(first_starts.values(), dtype=int, count=len(first_starts))

    @lru_cache()
    def _get_transition_pair_first_time_steps(
            self,
            minimal_time_ms: float,
            **kwargs,
    ) -> np.ndarray:
        session_attractors = self._get_session_attractors_data(minimal_time_ms, **kwargs)
        session_lengths = self._get_session_lengths_steps(**kwargs)
        offsets = np.cumsum([0] + session_lengths[:-1])
        first_times = {}
        for session_data, offset in zip(session_attractors, offsets):
            times = []
            labels = []
            for identity, entry in session_data.items():
                starts = np.asarray(entry.get("starts", []))
                if starts.size == 0:
                    continue
                times.append(starts)
                labels.extend([identity] * starts.size)
            if not times:
                continue
            times = np.concatenate(times)
            labels = np.array(labels, dtype=object)
            order = np.argsort(times)
            times = times[order]
            labels = labels[order]
            if labels.size < 2:
                continue
            src = labels[:-1]
            dst = labels[1:]
            transition_times = times[1:] + offset
            for s, d, t in zip(src, dst, transition_times):
                pair = (s, d)
                prev = first_times.get(pair)
                if prev is None or t < prev:
                    first_times[pair] = int(t)
        if not first_times:
            return np.empty((0,), dtype=int)
        return np.fromiter(first_times.values(), dtype=int, count=len(first_times))

    def get_unique_attractors_count_until_time(
            self,
            time_ms: float,
            **kwargs,
    ) -> int:
        if time_ms < 0:
            raise ValueError("time_ms must be non-negative.")
        dt_ms = self.dt * 1e3
        time_steps = int(np.floor(time_ms / dt_ms))
        minimal_time_ms = kwargs.pop('minimal_time_ms', self._minimal_life_span_ms)
        first_starts = self._get_unique_attractor_first_start_steps(
            minimal_time_ms,
            **kwargs,
        )
        if first_starts.size == 0:
            return 0
        return int(np.count_nonzero(first_starts <= time_steps))

    def get_transition_density_until_time(
            self,
            time_ms: float,
            **kwargs,
    ) -> float:
        if time_ms < 0:
            raise ValueError("time_ms must be non-negative.")
        dt_ms = self.dt * 1e3
        time_steps = int(np.floor(time_ms / dt_ms))
        minimal_time_ms = kwargs.pop('minimal_time_ms', self._minimal_life_span_ms)
        first_times = self._get_transition_pair_first_time_steps(
            minimal_time_ms,
            **kwargs,
        )
        if first_times.size == 0:
            return 0.0
        nonzero = int(np.count_nonzero(first_times <= time_steps))
        n_attractors = self.get_unique_attractors_count_until_time(
            time_ms,
            minimal_time_ms=minimal_time_ms,
            **kwargs,
        )
        total_entries = n_attractors * n_attractors
        if total_entries == 0:
            return 0.0
        return nonzero / total_entries

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

    @lru_cache()
    def _get_session_cluster_spike_rates(self, **kwargs):
        params = self._clustering_params.copy()
        params.update(kwargs)
        params.setdefault('dt_ms', self.dt / 1e-3)
        rates = []
        for spikes, clusters in self._get_sessions():
            rates.append(
                activity.get_average_cluster_firing_rate(
                    spikes,
                    clusters,
                    **params
                )
            )
        return rates

    def _get_session_cluster_activity(self, **kwargs):
        activity_mats = []
        for cluster_rates in self._get_session_cluster_spike_rates(**kwargs):
            activity_mats.append(activity.get_activity(cluster_rates))
        return activity_mats

    def _get_session_attractors_data(self, minimal_time_ms: float, **kwargs):
        dt_ms = self.dt * 1e3
        session_attractors = []
        for activity_matrix in self._get_session_cluster_activity(**kwargs):
            session_attractors.append(
                activity.extract_attractors(activity_matrix, minimal_time_ms, dt_ms)
            )
        return session_attractors

    @lru_cache()
    def _get_session_lengths_steps(self, **kwargs):
        return [mat.shape[1] for mat in self._get_session_cluster_activity(**kwargs)]

    def _validate_no_cross_simulation_attractors(self, session_attractors, session_lengths):
        if len(session_attractors) != len(session_lengths):
            raise ValueError("Session lengths do not match attractor sessions.")
        offsets = np.cumsum([0] + session_lengths[:-1]).tolist()
        for session_idx, attractors_data in enumerate(session_attractors):
            session_len = session_lengths[session_idx]
            offset = offsets[session_idx]
            for identity, entry in attractors_data.items():
                starts = entry.get("starts", [])
                ends = entry.get("ends", [])
                if len(starts) != len(ends):
                    raise ValueError(f"Mismatched starts/ends for attractor {identity}.")
                prev_end = None
                for start, end in zip(starts, ends):
                    if start < 0 or end > session_len or end <= start:
                        raise ValueError(
                            f"Invalid start/end for attractor {identity} in session {session_idx}."
                        )
                    global_start = start + offset
                    global_end = end + offset
                    if global_start < offset or global_end > offset + session_len:
                        raise ValueError(
                            f"Attractor {identity} crosses simulation boundary in session {session_idx}."
                        )
                    if prev_end is not None and start < prev_end:
                        raise ValueError(
                            f"Attractor {identity} has overlapping occurrences in session {session_idx}."
                        )
                    prev_end = end

    def _merge_attractors_data(self, session_attractors):
        merged = {}
        idx = 0
        for attractors_data in session_attractors:
            for identity, entry in attractors_data.items():
                if identity not in merged:
                    merged[identity] = {
                        "idx": idx,
                        "#": 0,
                        "starts": [],
                        "ends": [],
                        "occurrence_durations": [],
                        "total_duration": 0,
                        "clusters": identity,
                    }
                    idx += 1
                merged_entry = merged[identity]
                merged_entry["#"] += entry["#"]
                merged_entry["starts"].extend(entry["starts"])
                merged_entry["ends"].extend(entry["ends"])
                merged_entry["occurrence_durations"].extend(entry["occurrence_durations"])
                merged_entry["total_duration"] += entry["total_duration"]
        return merged
