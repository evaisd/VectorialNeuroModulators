"""Analyzer for processed SNN attractor data."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neuro_mod.analysis import BaseAnalyzer, MetricResult, manipulation, metric, reader
from neuro_mod.core.spiking_net.analysis.logic import transitions
from neuro_mod.core.spiking_net.analysis import helpers
from neuro_mod.core.spiking_net.analysis.logic import time_window


class SNNAnalyzer(BaseAnalyzer):
    """Analyze processed SNN attractor data to extract metrics.

    This class loads attractors_data (from SNNProcessor or file) and
    provides methods for computing metrics like lifespans, probabilities,
    transition matrices, and more.
    """

    def __init__(
            self,
            processed_data: dict | Path | str,
            *,
            config: dict | None = None,
    ) -> None:
        """Initialize the SNN analyzer.

        Args:
            processed_data: Either attractors_data dict or path to saved data.
            config: Optional configuration dict (loaded from file if path given).
        """
        if isinstance(processed_data, (str, Path)):
            path = Path(processed_data)
            if config is None:
                config = helpers.load_config(path)
            self._processed_data = helpers.load_from_path(path)
        else:
            self._processed_data = processed_data

        self._config = config or {}
        self._attractor_map: dict | None = None
        self._transition_matrix: np.ndarray | None = None
        self.processed_time_df = pd.DataFrame()
        self._df: pd.DataFrame | None = None

        # Extract config values
        self.dt = self._config.get("dt", 0.5e-3)
        self.total_duration_ms = self._config.get("total_duration_ms", 0.0)
        self.minimal_life_span_ms = self._config.get("minimal_life_span_ms", 20.0)
        self.session_lengths_steps = self._config.get("session_lengths_steps", [])
        self.repeat_durations_ms = self._config.get("repeat_durations_ms", [])
        self.n_runs = self._config.get("n_runs")


    @property
    def processed_data(self) -> dict:
        """Return the processed data dictionary."""
        return self._processed_data

    @property
    def attractors_data(self) -> dict:
        """Return the attractors_data dictionary."""
        return self._processed_data

    @property
    def attractor_map(self) -> dict:
        """Return mapping from index to identity."""
        if self._attractor_map is None:
            self._attractor_map = helpers.build_attractor_map(self.attractors_data)
        return self._attractor_map

    @property
    def df(self) -> pd.DataFrame:
        """Base occurrence-level DataFrame."""
        if self._df is None:
            self._df = self._build_dataframe()
        return self._df

    @reader("attractors_data")
    def _build_dataframe(self) -> pd.DataFrame:
        """Convert attractors_data to a pandas DataFrame.

        Returns a DataFrame indexed by occurrences sorted by time, with columns:
            - idx: Global occurrence index (sorted by t_start)
            - clusters: Cluster pattern identity
            - attractor_idx: Integer ID for this attractor
            - t_start: Start time of occurrence
            - t_end: End time of occurrence
            - duration: Duration of occurrence (ms)
            - occurrence: Per-attractor occurrence number (1st, 2nd, ... time this attractor appeared)
            - prev_attractor_idx: Attractor idx of the temporally previous occurrence
            - next_attractor_idx: Attractor idx of the temporally next occurrence
            - num_clusters: Number of active clusters in this attractor

        If batch processing was used, also includes: repeat, seed, sweep_value, sweep_idx.
        """
        rows = []
        for identity, entry in self.attractors_data.items():
            attractor_id = entry.get("idx", 0)
            starts = entry.get("starts", [])
            ends = entry.get("ends", [])
            durations = entry.get("occurrence_durations", [])

            # Count clusters in identity
            if isinstance(identity, (frozenset, set)):
                num_clusters = len(identity)
            elif isinstance(identity, (tuple, list)):
                num_clusters = len(identity)
            else:
                num_clusters = 1

            # Check for batch processing metadata
            repeat_indices = entry.get("repeat_indices", [])
            seeds = entry.get("seeds", [])
            sweep_values = entry.get("sweep_values", [])
            sweep_indices = entry.get("sweep_indices", [])
            has_metadata = bool(repeat_indices or seeds or sweep_values)

            for i, (start, end, dur) in enumerate(zip(starts, ends, durations)):
                row = {
                    "clusters": identity,
                    "attractor_idx": attractor_id,
                    "t_start": start,
                    "t_end": end,
                    "duration": dur,
                    "num_clusters": num_clusters,
                    "_per_attractor_idx": i,  # Temporary, for computing occurrence
                }

                # Add metadata columns if present
                if has_metadata:
                    if i < len(repeat_indices):
                        row["repeat"] = repeat_indices[i]
                    if i < len(seeds):
                        row["seed"] = seeds[i]
                    if i < len(sweep_values) and sweep_values[i] is not None:
                        row["sweep_value"] = sweep_values[i]
                    if i < len(sweep_indices) and sweep_indices[i] is not None:
                        row["sweep_idx"] = sweep_indices[i]

                rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Sort by time and assign global index
        df = df.sort_values("t_start").reset_index(drop=True)
        df.index.name = "idx"
        df = df.reset_index()

        # Compute per-attractor occurrence number (1-indexed)
        df["occurrence"] = df.groupby("attractor_idx").cumcount() + 1

        # Compute prev/next attractor idx (temporal neighbors)
        df["prev_attractor_idx"] = df["attractor_idx"].shift(1)
        df["next_attractor_idx"] = df["attractor_idx"].shift(-1)

        # Convert to nullable int for prev/next (NaN at boundaries)
        df["prev_attractor_idx"] = df["prev_attractor_idx"].astype("Int64")
        df["next_attractor_idx"] = df["next_attractor_idx"].astype("Int64")

        # Drop temporary column and reorder
        df = df.drop(columns=["_per_attractor_idx"])

        # Define column order
        base_cols = [
            "idx",
            "clusters",
            "attractor_idx",
            "t_start",
            "t_end",
            "duration",
            "occurrence",
            "prev_attractor_idx",
            "next_attractor_idx",
            "num_clusters",
        ]
        # Add metadata columns at the end if present
        extra_cols = [c for c in df.columns if c not in base_cols]
        df = df[base_cols + extra_cols]

        return df

    # --- Manipulations ---

    @manipulation("per_attractor")
    def per_attractor(self) -> pd.DataFrame:
        df = self.df
        if df.empty:
            return pd.DataFrame()
        agg = df.groupby("attractor_idx").agg(
            occurrences=("idx", "count"),
            total_duration=("duration", "sum"),
            mean_duration=("duration", "mean"),
            std_duration=("duration", "std"),
            num_clusters=("num_clusters", "first"),
            first_start=("t_start", "min"),
            last_end=("t_end", "max"),
        ).reset_index()
        agg["std_duration"] = agg["std_duration"].fillna(0)
        return agg

    @manipulation("transitions")
    def transitions_matrix(
        self,
        *,
        t_from: float | None = None,
        t_to: float | None = None,
    ) -> pd.DataFrame:
        matrix = self.get_transition_matrix(t_from=t_from, t_to=t_to)
        indices = self._get_attractor_indices_in_order(t_from=t_from, t_to=t_to)

        if indices and matrix.shape[0] == len(indices):
            df = pd.DataFrame(matrix, index=indices, columns=indices)
            mapping = pd.Series(self.attractor_map).sort_values()
            ordered = [idx for idx in mapping.index if idx in df.index]
            if ordered and len(ordered) == df.shape[0]:
                df = df.loc[ordered, ordered]
            return df
        return pd.DataFrame(matrix)

    @manipulation("time_evolution")
    def time_evolution(
        self,
        *,
        dt: float | None = None,
        num_steps: int | None = None,
    ) -> pd.DataFrame:
        return self.get_time_evolution_dataframe(dt=dt, num_steps=num_steps)

    @manipulation("filtered")
    def filtered(
        self,
        *,
        t_from: float | None = None,
        t_to: float | None = None,
        min_duration: float | None = None,
    ) -> pd.DataFrame:
        df = self.df
        if t_from is not None:
            df = df[df["t_start"] >= t_from]
        if t_to is not None:
            df = df[df["t_end"] <= t_to]
        if min_duration is not None:
            df = df[df["duration"] >= min_duration]
        return df.copy()

    # --- Metrics ---

    @metric("tpm_heatmap", expects="transitions")
    def tpm_heatmap(self, df: pd.DataFrame) -> MetricResult:
        metadata = {
            "index": df.index.to_numpy(),
            "columns": df.columns.to_numpy(),
            "colorbar_label": "probability",
        }
        return MetricResult(
            x=df.values,
            y=np.array([]),
            metadata=metadata
        )

    def get_summary_metrics(self) -> dict[str, Any]:
        """Extract summary metrics from the attractors_data.

        Returns:
            Dictionary with summary statistics.
        """
        num_states = self.get_num_states()
        lifespans_mean, lifespans_std = self.get_life_spans()
        probs = self.get_attractor_probs()
        occurrences = self.get_occurrences()
        total_duration_ms = self.total_duration_ms
        if total_duration_ms == 0.0 and self.session_lengths_steps:
            total_duration_ms = sum(self.session_lengths_steps) * self.dt * 1e3

        return {
            "num_states": num_states,
            "total_duration_s": total_duration_ms / 1e3,
            "mean_lifespan_ms": float(lifespans_mean.mean()) if lifespans_mean.size > 0 else 0.0,
            "total_occurrences": int(occurrences.sum()) if occurrences.size > 0 else 0,
            "mean_probability": float(probs.mean()) if probs.size > 0 else 0.0,
        }

    # --- Attractor data access methods ---

    def get_attractors_data(
            self,
            t_from: float | None = None,
            t_to: float | None = None,
    ) -> dict:
        """Get attractors_data, optionally filtered to a time window.

        Args:
            t_from: Start of time window (seconds).
            t_to: End of time window (seconds).

        Returns:
            Attractors_data dict (filtered if time window specified).
        """
        if t_from is not None or t_to is not None:
            total_duration_s = self.total_duration_ms / 1e3
            return helpers.filter_attractors_data_between(
                self.attractors_data,
                total_duration_s,
                t_from,
                t_to,
            )
        return self.attractors_data

    def get_attractor_data(
            self,
            *idx_or_identity: int | tuple[int, ...],
            t_from: float | None = None,
            t_to: float | None = None,
    ) -> dict:
        """Fetch attractor summaries by index or identity.

        Args:
            *idx_or_identity: Attractor indices or identity tuples.
            t_from: Start of time window (seconds).
            t_to: End of time window (seconds).

        Returns:
            Dictionary of attractor data keyed by the input identifiers.
        """
        out = {}
        data = self.get_attractors_data(t_from=t_from, t_to=t_to)
        for identifier in idx_or_identity:
            if isinstance(identifier, tuple):
                if identifier not in data:
                    raise ValueError("No attractor data for the requested time range.")
                out[identifier] = data[identifier]
            else:
                mapped_identity = self.attractor_map[identifier]
                if mapped_identity not in data:
                    raise ValueError("No attractor data for the requested time range.")
                out[identifier] = data[mapped_identity]
        return out

    def get_attractor_idx(self, *clusters: int) -> int:
        """Resolve an attractor identity to its index.

        Args:
            *clusters: Cluster indices describing the attractor identity.

        Returns:
            The index of the attractor.

        Raises:
            ValueError: If the attractor is not found.
        """
        idx = next(
            (key for key, value in self.attractor_map.items() if value == clusters),
            -1
        )
        if idx == -1:
            raise ValueError("No attractor found")
        return idx

    def get_num_states(
            self,
            t_from: float | None = None,
            t_to: float | None = None,
    ) -> int:
        """Return the number of detected attractor states."""
        return len(self.get_attractors_data(t_from=t_from, t_to=t_to))

    @lru_cache()
    def get_unique_attractors(
            self,
            t_from: float | None = None,
            t_to: float | None = None,
    ) -> set:
        """Return the set of unique attractor identities."""
        return set(self.get_attractors_data(t_from=t_from, t_to=t_to).keys())

    # --- Lifespan methods ---

    def get_mean_lifespan(
            self,
            *idx_or_identities: int | tuple[int, ...],
            t_from: float | None = None,
            t_to: float | None = None,
            median: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute mean/median and std of attractor lifespans.

        Args:
            *idx_or_identities: Attractor indices or identity tuples.
            t_from: Start of time window (seconds).
            t_to: End of time window (seconds).
            median: If True, compute median instead of mean.

        Returns:
            Tuple (means_or_medians, stds) as arrays in milliseconds.
        """
        means, stds = [], []
        for identifier in idx_or_identities:
            attractor_data = self.get_attractor_data(
                identifier,
                t_from=t_from,
                t_to=t_to,
            )[identifier]
            starts = np.asarray(attractor_data["starts"])
            ends = np.asarray(attractor_data["ends"])
            diffs = (ends - starts) * 1e3
            if not median:
                means.append(diffs.mean().round(4))
            else:
                means.append(np.median(diffs).round(4))
            stds.append(diffs.std().round(4))
        return np.stack(means, axis=0), np.stack(stds, axis=0)

    def get_life_spans(
            self,
            t_from: float | None = None,
            t_to: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return mean and std of lifespans for all attractors."""
        data = self.get_attractors_data(t_from=t_from, t_to=t_to)
        identities = helpers.get_attractor_identities_in_order(data)
        if not identities:
            return np.array([]), np.array([])
        return self.get_mean_lifespan(*identities, t_from=t_from, t_to=t_to)


    def get_occurrences(
            self,
            t_from: float | None = None,
            t_to: float | None = None,
    ) -> np.ndarray:
        """Return occurrence counts for each attractor."""
        att = self.get_attractors_data(t_from=t_from, t_to=t_to)
        identities = helpers.get_attractor_identities_in_order(att)
        return np.array([att[k]["#"] for k in identities])

    def get_num_clusters(
            self,
            t_from: float | None = None,
            t_to: float | None = None,
    ) -> np.ndarray:
        """Return number of clusters participating in each attractor."""
        att = self.get_attractors_data(t_from=t_from, t_to=t_to)
        identities = helpers.get_attractor_identities_in_order(att)
        return np.array([len(k) for k in identities])

    # --- Probability methods ---

    def get_attractor_prob(
            self,
            *idx_or_identity: int | tuple[int, ...],
            t_from: float | None = None,
            t_to: float | None = None,
    ) -> np.ndarray:
        """Compute occurrence probabilities for attractors.

        Args:
            *idx_or_identity: Attractor indices or identity tuples.
            t_from: Start of time window (seconds).
            t_to: End of time window (seconds).

        Returns:
            Array of probabilities for each attractor.
        """
        probs = []
        total_duration_s = self.total_duration_ms / 1e3
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
            )[identifier]
            duration = attractor_data["total_duration"]
            probs.append(duration / window_duration_ms)
        return np.stack(probs, axis=0)

    def get_attractor_probs(
            self,
            t_from: float | None = None,
            t_to: float | None = None,
    ) -> np.ndarray:
        """Return probability of all attractors in index order."""
        data = self.get_attractors_data(t_from=t_from, t_to=t_to)
        identities = helpers.get_attractor_identities_in_order(data)
        if not identities:
            return np.array([], dtype=float)
        return self.get_attractor_prob(*identities, t_from=t_from, t_to=t_to).flatten()

    # --- Transition methods ---

    @lru_cache()
    def get_transition_matrix(
            self,
            t_from: float | None = None,
            t_to: float | None = None,
    ) -> np.ndarray:
        """Return the transition matrix between attractors."""
        if t_from is None and t_to is None and self._transition_matrix is not None:
            return self._transition_matrix

        attractors_data = self.get_attractors_data(t_from=t_from, t_to=t_to)
        session_end_times = self._get_session_end_times_s()

        return transitions.get_transition_matrix_from_data(
            attractors_data,
            session_end_times if session_end_times else None,
        )

    def get_transition_prob(
            self,
            idx_or_identity_from: int | tuple[int, ...],
            idx_or_identity_to: int | tuple[int, ...],
            t_from: float | None = None,
            t_to: float | None = None,
    ) -> float:
        """Compute transition probability between two attractors.

        Args:
            idx_or_identity_from: Source attractor index or identity.
            idx_or_identity_to: Destination attractor index or identity.
            t_from: Start of time window (seconds).
            t_to: End of time window (seconds).

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
            identity_from = self.attractor_map[idx_or_identity_from]
        if isinstance(idx_or_identity_to, tuple):
            identity_to = idx_or_identity_to
        else:
            identity_to = self.attractor_map[idx_or_identity_to]

        if identity_from not in key_to_row or identity_to not in key_to_row:
            raise ValueError("No transition data for the requested time range.")
        return transition_matrix[key_to_row[identity_from], key_to_row[identity_to]]

    def get_sequence_probability(
            self,
            *idx_or_identity: int | tuple[int, ...],
            t_from: float | None = None,
            t_to: float | None = None,
    ) -> float:
        """Compute probability of a sequence of attractor transitions.

        Args:
            *idx_or_identity: Sequence of attractor indices or identities.
            t_from: Start of time window (seconds).
            t_to: End of time window (seconds).

        Returns:
            Probability of the sequence under the transition matrix.

        Raises:
            ValueError: If fewer than two attractors are provided.
        """
        if len(idx_or_identity) < 2:
            raise ValueError("Enter at least two attractors")
        probs = [
            self.get_transition_prob(att_a, att_b, t_from=t_from, t_to=t_to)
            for att_a, att_b in zip(idx_or_identity[:-1], idx_or_identity[1:])
        ]
        return np.prod(probs)

    # --- Time-evolution metrics ---

    def _get_unique_attractor_first_start_times(self) -> np.ndarray:
        """Get first start time for each unique attractor."""
        return helpers.get_unique_attractor_first_start_times(self.attractors_data)

    def get_unique_attractors_count_until_time(self, time_ms: float) -> int:
        """Count unique attractors observed up to a time threshold.

        Args:
            time_ms: Time threshold in milliseconds.

        Returns:
            Number of unique attractors observed up to time_ms.
        """
        if time_ms < 0:
            raise ValueError("time_ms must be non-negative.")
        time_s = time_ms / 1e3
        first_starts = self._get_unique_attractor_first_start_times()
        if first_starts.size == 0:
            return 0
        return int(np.count_nonzero(first_starts <= time_s))

    def get_transition_matrix_l2_norms_until_time(
            self,
            t: float | None = None,
            dt: float | None = None,
            num_steps: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute L2 norms between consecutive transition matrices up to time t.

        Args:
            t: End time in seconds. Defaults to full session duration.
            dt: Step size in seconds.
            num_steps: If provided, sets dt = t / num_steps.

        Returns:
            Tuple (times, norms) where times correspond to the second time in each pair.
        """
        if t is None:
            t = self.total_duration_ms / 1e3
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
            tm_current = transitions.get_transition_matrix_from_data(
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
            num = np.linalg.norm(tm_next - tm_prev)
            den = np.linalg.norm(tm_next) + np.linalg.norm(tm_prev)
            norms[i - 1] = 2.0 * num / (den + 1e-12)
            tm_prev = tm_next
        return times[1:], norms

    def get_time_evolution_dataframe(
        self,
        *,
        t: float | None = None,
        dt: float | None = None,
        num_steps: int | None = None,
    ) -> pd.DataFrame:
        """Return time-evolution metrics as a DataFrame.

        Columns:
            time_ms: Time points in milliseconds.
            transition_l2_norm: L2 norm between consecutive transition matrices.
            unique_attractors_count: Unique attractors observed up to time.
        """
        times_s, norms = self.get_transition_matrix_l2_norms_until_time(
            t=t,
            dt=dt,
            num_steps=num_steps,
        )
        if times_s.size == 0:
            self.processed_time_df = pd.DataFrame(
                columns=[
                    "time_ms",
                    "transition_l2_norm",
                    "unique_attractors_count",
                    "discovery_rate_per_s",
                ]
            )
            return self.processed_time_df
        times_ms = times_s * 1e3
        unique_counts = np.array(
            [self.get_unique_attractors_count_until_time(t_ms) for t_ms in times_ms],
            dtype=int,
        )
        discovery_rate = np.zeros_like(times_ms, dtype=float)
        if times_ms.size > 1:
            time_s = times_ms / 1e3
            dt_s = np.diff(time_s)
            new_attractors = np.diff(unique_counts)
            discovery_rate[1:] = np.divide(
                new_attractors,
                dt_s,
                out=np.zeros_like(dt_s, dtype=float),
                where=dt_s > 0,
            )
        self.processed_time_df = pd.DataFrame(
            {
                "time_ms": times_ms,
                "transition_l2_norm": norms,
                "unique_attractors_count": unique_counts,
                "discovery_rate_per_s": discovery_rate,
            }
        )
        return self.processed_time_df

    # --- Helper methods ---

    def _get_session_end_times_s(self) -> list[float]:
        """Get session end times in seconds."""
        if not self.session_lengths_steps:
            return []
        return helpers.get_session_end_times_s(self.session_lengths_steps, self.dt)

    def _get_attractor_indices_in_order(
        self,
        t_from: float | None = None,
        t_to: float | None = None,
    ) -> list[Any]:
        from neuro_mod.core.spiking_net.analysis.helpers import get_attractor_indices_in_order
        data = self.get_attractors_data(t_from=t_from, t_to=t_to)
        return get_attractor_indices_in_order(data)
