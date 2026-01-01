"""Analysis helpers for spiking network simulation outputs."""

from pathlib import Path
import json
from functools import lru_cache
import logging

import numpy as np
from neuro_mod.core.spiking_net.analysis.logic import activity
from neuro_mod.core.spiking_net.analysis.logic import time_window


class Analyzer:
    """Analyze spiking activity to extract attractors and transitions."""

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
        """Initialize the analyzer for a spikes file or folder.

        Args:
            spikes_or_folder: Path to a spikes file or directory of sessions.
            clusters: Optional path to cluster labels.
            dt: Time step in seconds.
        """
        self.spikes_path = Path(spikes_or_folder)
        self.clusters_path = clusters
        self.dt = dt
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session_length = len(self._get_sessions())
        self.total_sim_duration_ms = self._get_total_duration_ms()
        self.attractors_data = self.get_attractors_data()

    def _read_single(self, path: Path):
        data = np.load(path, allow_pickle=True)
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
        """Load and concatenate spikes across sessions.

        Returns:
            Tuple `(spikes, clusters)` where `spikes` is concatenated across
            sessions and `clusters` is the shared cluster label array.
        """
        sessions = self._get_sessions()
        if not sessions:
            return np.empty(()), np.empty(())
        spikes = [spikes for spikes, _ in sessions]
        clusters = sessions[0][1]
        concat_spikes = np.concatenate(spikes, axis=0)
        self.total_sim_duration_ms = self._get_total_duration_ms()
        return concat_spikes, clusters

    @lru_cache()
    def get_neuron_spike_rate(self,
                              **kwargs):
        """Compute firing rates for individual neurons.

        Args:
            **kwargs: Overrides for clustering parameters.

        Returns:
            Array of firing rates with shape `(n_neurons, T)`.
        """
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
        """Compute firing rates aggregated by cluster.

        Args:
            **kwargs: Overrides for clustering parameters.

        Returns:
            Array of firing rates with shape `(n_clusters, T)`.
        """
        firing_rates = self._get_session_cluster_spike_rates(**kwargs)
        return self._aggregate_series(firing_rates, axis=1)

    @lru_cache()
    def get_cluster_activity(
            self,
            **kwargs
    ):
        """Compute binary activity matrices per cluster.

        Args:
            **kwargs: Overrides for clustering parameters.

        Returns:
            Boolean activity array with shape `(n_clusters, T)`.
        """
        activity_mats = []
        for cluster_rates in self._get_session_cluster_spike_rates(**kwargs):
            activity_mats.append(activity.get_activity(cluster_rates))
        return self._aggregate_series(activity_mats, axis=1)

    @lru_cache()
    def get_attractors_data(self,
                              t_from: float | None = None,
                              t_to: float | None = None,
                              **kwargs):
        """Extract and merge attractors across sessions.

        Args:
            **kwargs: Parameters forwarded to attractor extraction.

        Returns:
            Mapping from attractor identity to summary dicts.
        """
        if t_from is not None or t_to is not None:
            return self._get_attractors_data_between(t_from, t_to)
        if self._can_use_loaded_attractors(kwargs):
            self._ensure_attractor_map()
            return self.attractors_data
        minimal_time_ms = kwargs.pop('minimal_time_ms', self._minimal_life_span_ms)
        session_attractors = self._get_session_attractors_data(minimal_time_ms, **kwargs)
        session_lengths = self._get_session_lengths_steps(**kwargs)
        self._validate_no_cross_simulation_attractors(session_attractors, session_lengths)
        attractors_data = self._merge_attractors_data(session_attractors, session_lengths)
        self._attractor_map = {attractors_data[k]['idx']: k
                               for k
                               in attractors_data.keys()}
        return attractors_data

    def _get_attractors_data_between(self, t_from: float | None, t_to: float | None) -> dict:
        """Return attractor data filtered to occurrences starting within [t_from, t_to].

        Args:
            t_from: Start time in seconds. Negative values are treated as offsets
                from the end of the simulation. None defaults to start.
            t_to: End time in seconds. Negative values are treated as offsets
                from the end of the simulation. None defaults to end.

        Returns:
            Filtered attractor data mapping.
        """
        total_duration_s = self.total_sim_duration_ms / 1e3
        start_steps, end_steps = time_window.get_time_bounds_steps(
            total_duration_s,
            self.dt,
            t_from,
            t_to,
        )
        filtered = {}
        for identity, entry in self.attractors_data.items():
            starts = entry.get("starts", [])
            if not starts:
                continue
            keep_idx = [i for i, s in enumerate(starts) if start_steps <= s <= end_steps]
            if not keep_idx:
                continue
            ends = entry.get("ends", [])
            durations = entry.get("occurrence_durations", [])
            filtered_entry = {
                "idx": entry.get("idx"),
                "#": len(keep_idx),
                "starts": [starts[i] for i in keep_idx],
                "ends": [ends[i] for i in keep_idx],
                "occurrence_durations": [durations[i] for i in keep_idx],
                "total_duration": float(np.sum([durations[i] for i in keep_idx])),
                "clusters": entry.get("clusters", identity),
            }
            filtered[identity] = filtered_entry
        return filtered

    @staticmethod
    def _get_attractor_identities_in_order(attractors_data: dict) -> list[tuple[int, ...]]:
        return [
            identity
            for identity, entry in sorted(
                attractors_data.items(),
                key=lambda item: item[1].get("idx", 0),
            )
        ]

    def _can_use_loaded_attractors(self, kwargs: dict) -> bool:
        if not hasattr(self, "attractors_data"):
            return False
        if not kwargs:
            return True
        if set(kwargs.keys()) == {"minimal_time_ms"}:
            return kwargs["minimal_time_ms"] == self._minimal_life_span_ms
        return False

    def _ensure_attractor_map(self) -> None:
        if hasattr(self, "_attractor_map"):
            return
        self._attractor_map = {
            self.attractors_data[k]["idx"]: k
            for k in self.attractors_data.keys()
        }

    def get_attractor_data(self,
                           *idx_or_identity: int | tuple[int, ...],
                           t_from: float | None = None,
                           t_to: float | None = None,):
        """Fetch attractor summaries by index or identity.

        Args:
            *idx_or_identity: Attractor indices or identity tuples.

        Returns:
            Dictionary of attractor data keyed by the input identifiers.
        """
        out = {}
        data = self.get_attractors_data(t_from=t_from, t_to=t_to)
        for idx in idx_or_identity:
            if isinstance(idx, tuple):
                if idx not in data:
                    raise ValueError("No attractor data for the requested time range.")
                out.update({idx: data[idx]})
            else:
                _idx = self._attractor_map[idx]
                if _idx not in data:
                    raise ValueError("No attractor data for the requested time range.")
                out.update({idx: data[_idx]})
        self.logger.info("get_attractor_data executed successfully.")
        return out

    def get_attractor_idx(
            self,
            *clusters: int,
    ) -> int:
        """Resolve an attractor identity to its index.

        Args:
            *clusters: Cluster indices describing the attractor identity.

        Returns:
            The index of the attractor.

        Raises:
            ValueError: If the attractor is not found.
        """
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
            t_from: float | None = None,
            t_to: float | None = None,
            **kwargs):
        """Compute mean and standard deviation of attractor lifespans.

        Args:
            *idx_or_identities: Attractor indices or identity tuples.
            **kwargs: Forwarded to `get_attractor_data`.

        Returns:
            Tuple `(means, stds)` as arrays in milliseconds.
        """
        means, stds = [], []
        for idx in idx_or_identities:
            attractor_data = self.get_attractor_data(idx, t_from=t_from, t_to=t_to, **kwargs)[idx]
            starts = np.asarray(attractor_data['starts'])
            ends = np.asarray(attractor_data['ends'])
            diffs = self.dt * 1e3 * (ends - starts)
            means.append(diffs.mean().round(4))
            stds.append(diffs.std().round(4))
        return np.stack(means, axis=0), np.stack(stds, axis=0)

    def get_attractor_prob(self,
                           *idx_or_identity: int | tuple[int, ...],
                           t_from: float | None = None,
                           t_to: float | None = None,
                           **kwargs) -> np.ndarray:
        """Compute occurrence probabilities for attractors.

        Args:
            *idx_or_identity: Attractor indices or identity tuples.
            **kwargs: Forwarded to `get_attractor_data`.

        Returns:
            Array of probabilities for each attractor.
        """
        probs = []
        total_duration_s = self.total_sim_duration_ms / 1e3
        t_from_s, t_to_s = time_window.resolve_time_bounds_s(
            total_duration_s,
            t_from,
            t_to,
        )
        total_duration_ms = (t_to_s - t_from_s) * 1e3
        if total_duration_ms <= 0:
            return np.zeros((len(idx_or_identity),), dtype=float)
        for idx in idx_or_identity:
            attractor_data = self.get_attractor_data(idx, t_from=t_from, t_to=t_to, **kwargs)[idx]
            duration = attractor_data['total_duration']
            probs.append(duration / total_duration_ms)
        return np.stack(probs, axis=0)

    def get_num_states(self,
                       t_from: float | None = None,
                       t_to: float | None = None,) -> int:
        """Return the number of detected attractor states."""
        return len(self.get_attractors_data(t_from=t_from, t_to=t_to))

    def get_life_spans(self,
                       t_from: float | None = None,
                       t_to: float | None = None,):
        """Return mean and std of lifespans for all attractors."""
        data = self.get_attractors_data(t_from=t_from, t_to=t_to)
        identities = self._get_attractor_identities_in_order(data)
        return self.get_mean_lifespan(*identities, t_from=t_from, t_to=t_to)

    def get_occurrences(self,
                        t_from: float | None = None,
                        t_to: float | None = None,) -> np.ndarray:
        """Return occurrence counts for each attractor."""
        att = self.get_attractors_data(t_from=t_from, t_to=t_to)
        identities = self._get_attractor_identities_in_order(att)
        return np.array([att[k]["#"] for k in identities])

    def get_num_clusters(self,
                         t_from: float | None = None,
                         t_to: float | None = None,) -> np.ndarray:
        """Return number of clusters participating in each attractor."""
        att = self.get_attractors_data(t_from=t_from, t_to=t_to)
        identities = self._get_attractor_identities_in_order(att)
        return np.array([len(k) for k in identities])

    def get_attractor_probs(self,
                            t_from: float | None = None,
                            t_to: float | None = None,) -> np.ndarray:
        """Return probability of all attractors in order of indices."""
        data = self.get_attractors_data(t_from=t_from, t_to=t_to)
        identities = self._get_attractor_identities_in_order(data)
        if not identities:
            return np.array([], dtype=float)
        return self.get_attractor_prob(*identities, t_from=t_from, t_to=t_to).flatten()

    def save_analysis(
            self,
            folder: str | Path,
            *,
            attractors_filename: str = "attractors.npy",
            transition_filename: str = "transition_matrix.npy",
            config_filename: str = "analysis_config.json",
    ) -> Path:
        """Persist analysis artifacts to disk.

        Args:
            folder: Directory to write analysis artifacts.
            attractors_filename: Filename for attractor data.
            transition_filename: Filename for transition matrix.
            config_filename: Filename for config metadata.

        Returns:
            Path to the analysis folder.
        """
        self.logger.info("Saving analysis artifacts.")
        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)
        attractors_path = folder_path / attractors_filename
        transition_path = folder_path / transition_filename
        config_path = folder_path / config_filename

        np.save(attractors_path, self.attractors_data, allow_pickle=True)
        np.save(transition_path, self.get_transition_matrix())

        config = {
            "spikes_path": str(self.spikes_path),
            "clusters_path": str(self.clusters_path) if self.clusters_path is not None else None,
            "dt": self.dt,
            "minimal_life_span_ms": self._minimal_life_span_ms,
            "clustering_params": self._clustering_params,
            "session_length": self.session_length,
            "session_lengths_steps": self._get_session_lengths_steps(),
            "total_sim_duration_ms": self.total_sim_duration_ms,
            "files": {
                "attractors": attractors_filename,
                "transition_matrix": transition_filename,
            },
        }
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True))
        self.logger.info(
            "Analysis saved.",
            extra={
                "analysis_dir": str(folder_path),
                "attractors_file": str(attractors_path),
                "transition_file": str(transition_path),
                "config_file": str(config_path),
            },
        )
        return folder_path

    @classmethod
    def load_analysis(
            cls,
            folder: str | Path,
            *,
            config_filename: str = "analysis_config.json",
    ) -> "Analyzer":
        """Load a saved analysis from disk.

        Args:
            folder: Analysis directory.
            config_filename: Name of the config file.

        Returns:
            An `Analyzer` instance populated with saved data.
        """
        folder_path = Path(folder)
        config_path = folder_path / config_filename
        config = json.loads(config_path.read_text())

        files = config.get("files", {})
        attractors_path = folder_path / files.get("attractors", "attractors.npy")
        transition_path = folder_path / files.get("transition_matrix", "transition_matrix.npy")

        obj = cls.__new__(cls)
        obj.spikes_path = Path(config["spikes_path"])
        obj.clusters_path = (
            Path(config["clusters_path"])
            if config.get("clusters_path") is not None
            else None
        )
        obj.dt = float(config["dt"])
        obj.logger = logging.getLogger(cls.__name__)
        obj.session_length = int(config.get("session_length", 0))
        obj.total_sim_duration_ms = float(config.get("total_sim_duration_ms", 0.0))
        obj._minimal_life_span_ms = float(config.get("minimal_life_span_ms", cls._minimal_life_span_ms))
        obj._clustering_params = config.get("clustering_params", cls._clustering_params)
        obj._session_lengths_steps = config.get("session_lengths_steps")

        obj.attractors_data = np.load(attractors_path, allow_pickle=True).item()
        obj._attractor_map = {obj.attractors_data[k]["idx"]: k for k in obj.attractors_data.keys()}
        obj._transition_matrix = np.load(transition_path)
        obj.logger.info(
            "Analysis loaded from disk.",
            extra={
                "analysis_dir": str(folder_path),
                "attractors_file": str(attractors_path),
                "transition_file": str(transition_path),
                "config_file": str(config_path),
            },
        )
        return obj

    def get_transition_prob(self,
                            idx_or_identity_from: int | tuple[int, ...],
                            idx_or_identity_to: int | tuple[int, ...],
                            t_from: float | None = None,
                            t_to: float | None = None,
                            *args,
                            **kwargs) -> float:
        """Compute transition probability between two attractors.

        Args:
            idx_or_identity_from: Source attractor index or identity.
            idx_or_identity_to: Destination attractor index or identity.
            *args: Ignored positional arguments.
            **kwargs: Ignored keyword arguments.

        Returns:
            Transition probability from source to destination.
        """
        transition_matrix = self.get_transition_matrix(t_from=t_from, t_to=t_to)
        if t_from is None and t_to is None:
            if isinstance(idx_or_identity_from, tuple):
                idx_or_identity_from = self.get_attractor_idx(*idx_or_identity_from)
            if isinstance(idx_or_identity_to, tuple):
                idx_or_identity_to = self.get_attractor_idx(*idx_or_identity_to)
            return transition_matrix[idx_or_identity_from, idx_or_identity_to]
        data = self.get_attractors_data(t_from=t_from, t_to=t_to)
        keys = self._get_attractor_identities_in_order(data)
        key_to_row = {k: i for i, k in enumerate(keys)}
        if isinstance(idx_or_identity_from, tuple):
            identity_from = idx_or_identity_from
        else:
            identity_from = self._attractor_map[idx_or_identity_from]
        if isinstance(idx_or_identity_to, tuple):
            identity_to = idx_or_identity_to
        else:
            identity_to = self._attractor_map[idx_or_identity_to]
        if identity_from not in key_to_row or identity_to not in key_to_row:
            raise ValueError("No transition data for the requested time range.")
        return transition_matrix[key_to_row[identity_from], key_to_row[identity_to]]

    @lru_cache()
    def get_transition_matrix(self,
                              t_from: float | None = None,
                              t_to: float | None = None,) -> np.ndarray:
        """Return the transition matrix between attractors."""
        if t_from is None and t_to is None and hasattr(self, "_transition_matrix"):
            return self._transition_matrix
        if t_from is None and t_to is None:
            session_attractors = self._get_session_attractors_data(self._minimal_life_span_ms)
            return activity.get_transition_matrix_session_aware(
                self.attractors_data,
                session_attractors,
            )
        attractors_data = self.get_attractors_data(t_from=t_from, t_to=t_to)
        if not attractors_data:
            return np.zeros((0, 0), dtype=float)
        keys = self._get_attractor_identities_in_order(attractors_data)
        key_to_row = {k: i for i, k in enumerate(keys)}
        n = len(keys)
        occ = np.zeros(n, dtype=float)
        times, labels = activity.get_ordered_occurrences(attractors_data)
        if times.size == 0:
            return np.zeros((n, n), dtype=float)
        for identity in keys:
            entry = attractors_data[identity]
            occ[key_to_row[identity]] = entry.get("#", 0)
        if labels.size < 2:
            return np.zeros((n, n), dtype=float)
        session_end_steps = self._get_session_end_steps()
        counts = activity.get_transition_counts_from_occurrences(
            times,
            labels,
            key_to_row,
            session_end_steps=session_end_steps if session_end_steps else None,
        )
        occ[occ == 0] = 1.0
        return counts / occ[:, None]

    def _files_walker(self):
        if self.spikes_path.is_file():
            yield self.spikes_path
        else:
            for p in sorted(self.spikes_path.glob("*.np[yz]")):
                yield p

    @lru_cache()
    def get_unique_attractors(self,
                              t_from: float | None = None,
                              t_to: float | None = None,):
        """Return the set of unique attractor identities."""
        attractors = set(list(self.get_attractors_data(t_from=t_from, t_to=t_to).keys()))
        return attractors

    @lru_cache()
    def _get_unique_attractor_first_start_steps_from_attractors(self) -> np.ndarray:
        first_starts = []
        for entry in self.attractors_data.values():
            starts = entry.get("starts", [])
            if not starts:
                continue
            first_starts.append(min(starts))
        if not first_starts:
            return np.empty((0,), dtype=int)
        return np.asarray(first_starts, dtype=int)

    def _get_session_end_steps(self) -> list[int]:
        lengths = getattr(self, "_session_lengths_steps", None)
        if lengths is None:
            if self._get_sessions():
                lengths = self._get_session_lengths_steps()
            else:
                return []
        return np.cumsum(lengths).tolist()

    def get_unique_attractors_count_until_time(
            self,
            time_ms: float,
            **kwargs,
    ) -> int:
        """Count unique attractors observed up to a time threshold.

        Args:
            time_ms: Time threshold in milliseconds.
            **kwargs: Parameters forwarded to attractor extraction.

        Returns:
            Number of unique attractors observed up to `time_ms`.
        """
        if time_ms < 0:
            raise ValueError("time_ms must be non-negative.")
        dt_ms = self.dt * 1e3
        time_steps = int(np.floor(time_ms / dt_ms))
        kwargs.pop('minimal_time_ms', None)
        first_starts = self._get_unique_attractor_first_start_steps_from_attractors()
        first_starts = np.asarray(first_starts, dtype=int)
        if first_starts.size == 0:
            return 0
        return int(np.count_nonzero(first_starts <= time_steps))

    def get_transition_density_until_time(
            self,
            time_ms: float,
            **kwargs,
    ) -> float:
        """Compute transition density up to a time threshold.

        Args:
            time_ms: Time threshold in milliseconds.
            **kwargs: Parameters forwarded to attractor extraction.

        Returns:
            Fraction of observed transitions among possible pairs.
        """
        if time_ms < 0:
            raise ValueError("time_ms must be non-negative.")
        dt_ms = self.dt * 1e3
        time_steps = int(np.floor(time_ms / dt_ms))
        kwargs.pop('minimal_time_ms', None)
        times, labels = activity.get_ordered_occurrences(self.attractors_data)
        if times.size == 0:
            return 0.0
        session_end_steps = self._get_session_end_steps()
        before_mask = times < time_steps
        if not np.any(before_mask):
            return 0.0
        times = times[before_mask]
        labels = labels[before_mask]
        if labels.size < 2:
            return 0.0
        pairs = activity.get_transition_pairs(
            times,
            labels,
            session_end_steps=session_end_steps if session_end_steps else None,
        )
        if not pairs:
            return 0.0
        n_attractors = self.get_unique_attractors_count_until_time(time_ms)
        total_entries = n_attractors * n_attractors
        if total_entries == 0:
            return 0.0
        return len(pairs) / total_entries

    def get_sequence_probability(
            self,
            *idx_or_identity: int | tuple[int, ...],
            t_from: float | None = None,
            t_to: float | None = None,
    ):
        """Compute probability of a sequence of attractor transitions.

        Args:
            *idx_or_identity: Sequence of attractor indices or identities.

        Returns:
            Probability of the sequence under the transition matrix.

        Raises:
            ValueError: If fewer than two attractors are provided.
        """
        if len(idx_or_identity) == 1:
            raise ValueError('Enter atleast two attractors')
        probs = [self.get_transition_prob(att_a, att_b, t_from=t_from, t_to=t_to)
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

    def _merge_attractors_data(self, session_attractors, session_lengths):
        merged = {}
        idx = 0
        offsets = np.cumsum([0] + session_lengths[:-1]).tolist()
        for session_idx, attractors_data in enumerate(session_attractors):
            offset = offsets[session_idx]
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
                merged_entry["starts"].extend([s + offset for s in entry["starts"]])
                merged_entry["ends"].extend([e + offset for e in entry["ends"]])
                merged_entry["occurrence_durations"].extend(entry["occurrence_durations"])
                merged_entry["total_duration"] += entry["total_duration"]
        for entry in merged.values():
            starts = entry["starts"]
            if len(starts) <= 1:
                continue
            order = sorted(range(len(starts)), key=starts.__getitem__)
            entry["starts"] = [starts[i] for i in order]
            entry["ends"] = [entry["ends"][i] for i in order]
            entry["occurrence_durations"] = [entry["occurrence_durations"][i] for i in order]
        return merged
