"""Analysis helpers for spiking network simulation outputs."""

from pathlib import Path
import json
from functools import lru_cache
import logging

import numpy as np
from neuro_mod.core.spiking_net.analysis.logic import activity
from neuro_mod.core.spiking_net.analysis.logic import analyzer_helpers as helpers
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

    @lru_cache()
    def _get_sessions(self):
        return helpers.load_sessions(self.spikes_path, self.clusters_path)

    @lru_cache()
    def _get_total_duration_ms(self):
        return helpers.get_total_duration_ms(self._get_sessions(), self.dt)

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
        return helpers.aggregate_series(firing_rates, axis=1)

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
        return helpers.aggregate_series(firing_rates, axis=1)

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
        return helpers.aggregate_series(activity_mats, axis=1)

    @lru_cache()
    def get_attractors_data(self,
                              t_from: float | None = None,
                              t_to: float | None = None,
                              **kwargs):
        """Extract and merge attractors across sessions.

        Args:
            **kwargs: Parameters forwarded to attractor extraction.

        Returns:
            Mapping from attractor identity to summary dicts with starts/ends in seconds.
        """
        if t_from is not None or t_to is not None:
            total_duration_s = self.total_sim_duration_ms / 1e3
            return helpers.filter_attractors_data_between(
                self.attractors_data,
                total_duration_s,
                t_from,
                t_to,
            )
        if helpers.can_use_loaded_attractors(
                hasattr(self, "attractors_data"),
                kwargs,
                self._minimal_life_span_ms,
        ):
            if not hasattr(self, "_attractor_map"):
                self._attractor_map = helpers.build_attractor_map(self.attractors_data)
            return self.attractors_data
        minimal_time_ms = kwargs.pop('minimal_time_ms', self._minimal_life_span_ms)
        session_cluster_rates = self._get_session_cluster_spike_rates(**kwargs)
        session_cluster_activity = helpers.get_session_cluster_activity(session_cluster_rates)
        session_attractors = helpers.get_session_attractors_data(
            session_cluster_activity,
            minimal_time_ms,
            self.dt * 1e3,
        )
        session_lengths = helpers.get_session_lengths_steps(session_cluster_activity)
        helpers.validate_no_cross_simulation_attractors(session_attractors, session_lengths)
        attractors_data = helpers.merge_attractors_data(
            session_attractors,
            session_lengths,
            self.dt,
        )
        self._attractor_map = helpers.build_attractor_map(attractors_data)
        return attractors_data

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
        for identifier in idx_or_identity:
            if isinstance(identifier, tuple):
                if identifier not in data:
                    raise ValueError("No attractor data for the requested time range.")
                out.update({identifier: data[identifier]})
            else:
                mapped_identity = self._attractor_map[identifier]
                if mapped_identity not in data:
                    raise ValueError("No attractor data for the requested time range.")
                out.update({identifier: data[mapped_identity]})
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
            median: bool = False,
            **kwargs):
        """Compute mean and standard deviation of attractor lifespans.

        Args:
            *idx_or_identities: Attractor indices or identity tuples.
            **kwargs: Forwarded to `get_attractor_data`.

        Returns:
            Tuple `(means, stds)` as arrays in milliseconds.
        """
        means, stds = [], []
        for identifier in idx_or_identities:
            attractor_data = self.get_attractor_data(
                identifier,
                t_from=t_from,
                t_to=t_to,
                **kwargs,
            )[identifier]
            starts = np.asarray(attractor_data['starts'])
            ends = np.asarray(attractor_data['ends'])
            diffs = (ends - starts) * 1e3
            if not median:
                means.append(diffs.mean().round(4))
            else:
                means.append(np.median(diffs).round(4))
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
        window_duration_ms = (t_to_s - t_from_s) * 1e3
        if window_duration_ms <= 0:
            return np.zeros((len(idx_or_identity),), dtype=float)
        for identifier in idx_or_identity:
            attractor_data = self.get_attractor_data(
                identifier,
                t_from=t_from,
                t_to=t_to,
                **kwargs,
            )[identifier]
            duration = attractor_data['total_duration']
            probs.append(duration / window_duration_ms)
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
        identities = helpers.get_attractor_identities_in_order(data)
        return self.get_mean_lifespan(*identities, t_from=t_from, t_to=t_to)

    def get_occurrences(self,
                        t_from: float | None = None,
                        t_to: float | None = None,) -> np.ndarray:
        """Return occurrence counts for each attractor."""
        att = self.get_attractors_data(t_from=t_from, t_to=t_to)
        identities = helpers.get_attractor_identities_in_order(att)
        return np.array([att[k]["#"] for k in identities])

    def get_num_clusters(self,
                         t_from: float | None = None,
                         t_to: float | None = None,) -> np.ndarray:
        """Return number of clusters participating in each attractor."""
        att = self.get_attractors_data(t_from=t_from, t_to=t_to)
        identities = helpers.get_attractor_identities_in_order(att)
        return np.array([len(k) for k in identities])

    def get_attractor_probs(self,
                            t_from: float | None = None,
                            t_to: float | None = None,) -> np.ndarray:
        """Return probability of all attractors in order of indices."""
        data = self.get_attractors_data(t_from=t_from, t_to=t_to)
        identities = helpers.get_attractor_identities_in_order(data)
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
            "starts_ends_unit": "seconds",
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
        unit = config.get("starts_ends_unit", "steps")
        if unit == "steps":
            obj.attractors_data = helpers.convert_attractors_data_steps_to_seconds(
                obj.attractors_data,
                obj.dt,
            )
        obj._attractor_map = helpers.build_attractor_map(obj.attractors_data)
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
        keys = helpers.get_attractor_identities_in_order(data)
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
        session_end_times = self._get_session_end_times_s()
        return activity.get_transition_matrix_from_data(
            attractors_data,
            session_end_times=session_end_times if session_end_times else None,
        )

    @lru_cache()
    def get_unique_attractors(self,
                              t_from: float | None = None,
                              t_to: float | None = None,):
        """Return the set of unique attractor identities."""
        attractors = set(list(self.get_attractors_data(t_from=t_from, t_to=t_to).keys()))
        return attractors

    @lru_cache()
    def _get_unique_attractor_first_start_times_from_attractors(self) -> np.ndarray:
        return helpers.get_unique_attractor_first_start_times(self.attractors_data)

    def _get_session_end_times_s(self) -> list[float]:
        lengths = getattr(self, "_session_lengths_steps", None)
        if lengths is None:
            if self._get_sessions():
                lengths = self._get_session_lengths_steps()
            else:
                return []
        return helpers.get_session_end_times_s(lengths, self.dt)

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
        time_s = time_ms / 1e3
        kwargs.pop('minimal_time_ms', None)
        first_starts = self._get_unique_attractor_first_start_times_from_attractors()
        first_starts = np.asarray(first_starts, dtype=float)
        if first_starts.size == 0:
            return 0
        return int(np.count_nonzero(first_starts <= time_s))

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
        time_s = time_ms / 1e3
        kwargs.pop('minimal_time_ms', None)
        times, labels = activity.get_ordered_occurrences(self.attractors_data)
        if times.size == 0:
            return 0.0
        session_end_times = self._get_session_end_times_s()
        within_window = times < time_s
        if not np.any(within_window):
            return 0.0
        times = times[within_window]
        labels = labels[within_window]
        if labels.size < 2:
            return 0.0
        pairs = activity.get_transition_pairs(
            times,
            labels,
            session_end_times=session_end_times if session_end_times else None,
        )
        if not pairs:
            return 0.0
        n_attractors = self.get_unique_attractors_count_until_time(time_ms)
        total_entries = n_attractors * n_attractors
        if total_entries == 0:
            return 0.0
        return len(pairs) / total_entries

    def get_transition_matrix_l2_norms_until_time(
            self,
            t: float | None = None,
            dt: float | None = None,
            num_steps: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute L2 norms between consecutive transition matrices up to time t.

        Args:
            t: End time in seconds. Defaults to full session duration.
            dt: Step size in seconds. Defaults to max(1e-3, t / 100).
            num_steps: If provided, sets dt to t / num_steps.

        Returns:
            Tuple of (times, norms) where times correspond to the second time in each pair.
        """
        if t is None:
            t = self.total_sim_duration_ms / 1e3
        if t < 0:
            raise ValueError("t must be non-negative.")
        if num_steps is not None:
            if num_steps <= 0:
                raise ValueError("num_steps must be positive.")
            dt = t / num_steps if t > 0 else 1e-3
        if dt is None:
            dt = max(1e-3, t / 100) if t > 0 else 1e-3
        if dt <= 0:
            raise ValueError("dt must be positive.")
        times = np.arange(0.0, t + 1e-12, dt, dtype=float)
        if times.size < 2:
            return np.empty((0,), dtype=float), np.empty((0,), dtype=float)
        base_attractors = self.get_attractors_data(t_from=0, t_to=t)
        if not base_attractors:
            return times[1:], np.zeros(times.size - 1, dtype=float)
        base_keys = helpers.get_attractor_identities_in_order(base_attractors)
        base_idx_to_row = {
            base_attractors[k].get("idx", k): i
            for i, k in enumerate(base_keys)
        }
        n_base = len(base_keys)
        session_end_times = self._get_session_end_times_s()
        session_end_times = session_end_times if session_end_times else None

        def embed_matrix(current_data: dict) -> np.ndarray:
            if not current_data:
                return np.zeros((n_base, n_base), dtype=float)
            current_keys = helpers.get_attractor_identities_in_order(current_data)
            tm_current = activity.get_transition_matrix_from_data(
                current_data,
                session_end_times=session_end_times,
            )
            if tm_current.size == 0:
                return np.zeros((n_base, n_base), dtype=float)
            base_rows = []
            current_indices = []
            for idx_current, key in enumerate(current_keys):
                idx = current_data[key].get("idx", key)
                if idx in base_idx_to_row:
                    base_rows.append(base_idx_to_row[idx])
                    current_indices.append(idx_current)
            if not base_rows or not current_indices:
                return np.zeros((n_base, n_base), dtype=float)
            mat = np.zeros((n_base, n_base), dtype=float)
            sub_tm = tm_current[np.ix_(current_indices, current_indices)]
            mat[np.ix_(base_rows, base_rows)] = sub_tm
            return mat

        tm_prev = embed_matrix(self.get_attractors_data(t_from=0, t_to=float(times[0])))
        norms = np.zeros(times.size - 1, dtype=float)
        for i in range(1, times.size):
            tm_next = embed_matrix(self.get_attractors_data(t_from=0, t_to=float(times[i])))
            norms[i - 1] = np.linalg.norm(tm_next - tm_prev)
            tm_prev = tm_next
        return times[1:], norms

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
        return helpers.get_session_cluster_spike_rates(
            self._get_sessions(),
            params,
            self.dt,
        )

    def _get_session_cluster_activity(self, **kwargs):
        return helpers.get_session_cluster_activity(
            self._get_session_cluster_spike_rates(**kwargs)
        )

    def _get_session_attractors_data(self, minimal_time_ms: float, **kwargs):
        return helpers.get_session_attractors_data(
            self._get_session_cluster_activity(**kwargs),
            minimal_time_ms,
            self.dt * 1e3,
        )

    @lru_cache()
    def _get_session_lengths_steps(self, **kwargs):
        return helpers.get_session_lengths_steps(self._get_session_cluster_activity(**kwargs))
