"""Processor for spiking neural network simulation outputs."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from neuro_mod.data_processing.base_processor import _BaseSimProcessor
from neuro_mod.core.spiking_net.processing.logic import attractors
from neuro_mod.core.spiking_net.processing.logic import detection
from neuro_mod.core.spiking_net.processing.logic import firing_rates as fr
from neuro_mod.core.spiking_net.processing.logic import session_helpers as helpers


class SNNBatchProcessor:
    """Batch processor for unified processing of multiple SNN simulation outputs.

    Processes all runs together to ensure consistent attractor identities
    across repeated simulations with the same dynamics.
    """

    def __init__(
        self,
        raw_outputs: list[dict[str, Any]],
        metadata: list[dict[str, Any]],
        *,
        dt: float = 0.5e-3,
        clustering_params: dict | None = None,
        minimal_life_span_ms: float = 20.0,
    ) -> None:
        """Initialize the batch processor.

        Args:
            raw_outputs: List of raw simulation outputs, each with 'spikes_path'.
            metadata: List of metadata dicts with seed, repeat_idx, etc.
            dt: Time step in seconds.
            clustering_params: Parameters for firing rate computation.
            minimal_life_span_ms: Minimum attractor duration in milliseconds.
        """
        self.raw_outputs = raw_outputs
        self.metadata = metadata
        self.dt = dt
        self.clustering_params = clustering_params or SNNProcessor.DEFAULT_CLUSTERING_PARAMS.copy()
        self.minimal_life_span_ms = minimal_life_span_ms
        self._processed_data: dict | None = None
        self._total_duration_ms: float = 0.0
        self._session_lengths_steps: list[int] = []
        self._repeat_durations_ms: list[float] = []

    @property
    def processed_data(self) -> dict | None:
        """Return the processed data, or None if not yet processed."""
        return self._processed_data

    @property
    def total_duration_ms(self) -> float:
        """Return total simulation duration in milliseconds."""
        return self._total_duration_ms

    @property
    def session_lengths_steps(self) -> list[int]:
        """Return session lengths in time steps."""
        return self._session_lengths_steps

    @property
    def repeat_durations_ms(self) -> list[float]:
        """Return per-repeat durations in milliseconds."""
        return self._repeat_durations_ms

    def get_config(self) -> dict[str, Any]:
        """Return configuration dict for use with analyzer."""
        return {
            "dt": self.dt,
            "total_duration_ms": self._total_duration_ms,
            "minimal_life_span_ms": self.minimal_life_span_ms,
            "session_lengths_steps": self._session_lengths_steps,
            "repeat_durations_ms": self._repeat_durations_ms,
            "n_runs": len(self.raw_outputs),
        }

    def process_batch(
        self,
        raw_outputs: list[dict[str, Any]],
        metadata: list[dict[str, Any]],
    ) -> dict:
        """Process multiple raw outputs together with metadata.

        Loads all sessions, processes them together for consistent attractor
        identification, and embeds metadata (repeat_idx, seed, sweep_value)
        into the attractor occurrences.

        Args:
            raw_outputs: List of raw simulation outputs.
            metadata: List of metadata dicts for each output.

        Returns:
            Unified processed data with metadata embedded in occurrences.
        """
        # Collect per-session attractors without retaining full raw sessions.
        session_attractors = []
        session_lengths = []
        session_metadata = []
        repeat_durations_ms = []
        total_duration_ms = 0.0

        params = self.clustering_params.copy()
        params.setdefault("dt_ms", self.dt / 1e-3)

        for raw, meta in zip(raw_outputs, metadata):
            spikes_path = Path(raw["spikes_path"])
            clusters_path = Path(raw["clusters_path"]) if raw.get("clusters_path") else None

            sessions = helpers.load_sessions(spikes_path, clusters_path)
            repeat_duration_ms = 0.0
            for spikes, clusters in sessions:
                rates = fr.get_average_cluster_firing_rate(
                    spikes,
                    clusters,
                    **params,
                )
                activity = detection.get_activity(rates)
                session_attractors.append(
                    attractors.extract_attractors(
                        activity,
                        self.minimal_life_span_ms,
                        self.dt * 1e3,
                    )
                )
                session_lengths.append(activity.shape[1])
                session_metadata.append(meta.copy())
                session_duration_ms = self.dt * spikes.shape[0] * 1e3
                total_duration_ms += session_duration_ms
                repeat_duration_ms += session_duration_ms
            repeat_durations_ms.append(repeat_duration_ms)

        if not session_attractors:
            self._processed_data = {}
            self._repeat_durations_ms = repeat_durations_ms
            self._session_lengths_steps = []
            self._total_duration_ms = 0.0
            return self._processed_data

        # Validate and merge
        helpers.validate_no_cross_simulation_attractors(session_attractors, session_lengths)
        attractors_data = helpers.merge_attractors_data(
            session_attractors,
            session_lengths,
            self.dt,
        )

        # Embed metadata into attractor occurrences
        attractors_data = self._embed_metadata(
            attractors_data,
            session_lengths,
            session_metadata,
        )

        # Store session lengths and compute total duration
        self._session_lengths_steps = session_lengths
        self._total_duration_ms = total_duration_ms
        self._repeat_durations_ms = repeat_durations_ms

        self._processed_data = attractors_data
        return attractors_data

    def _embed_metadata(
        self,
        attractors_data: dict,
        session_lengths: list[int],
        session_metadata: list[dict[str, Any]],
    ) -> dict:
        """Embed metadata into attractor occurrence data.

        For each occurrence of each attractor, determine which session it
        came from and embed the corresponding metadata.

        Args:
            attractors_data: The merged attractor data.
            session_lengths: Length in steps of each session.
            session_metadata: Metadata for each session.

        Returns:
            attractors_data with metadata lists added.
        """
        # Compute session boundaries in seconds
        session_boundaries = [0.0]
        cumulative_steps = 0
        for length in session_lengths:
            cumulative_steps += length
            session_boundaries.append(cumulative_steps * self.dt)

        # For each attractor, add metadata for each occurrence
        for identity, data in attractors_data.items():
            starts = data.get("starts", [])
            n_occurrences = len(starts)

            # Initialize metadata lists
            data["repeat_indices"] = []
            data["seeds"] = []
            data["sweep_values"] = []
            data["sweep_indices"] = []

            for start_time in starts:
                # Find which session this occurrence belongs to
                session_idx = 0
                for i, (start, end) in enumerate(
                    zip(session_boundaries[:-1], session_boundaries[1:])
                ):
                    if start <= start_time < end:
                        session_idx = i
                        break

                # Get metadata for this session
                if session_idx < len(session_metadata):
                    meta = session_metadata[session_idx]
                    data["repeat_indices"].append(meta.get("repeat_idx"))
                    data["seeds"].append(meta.get("seed"))
                    data["sweep_values"].append(meta.get("sweep_value"))
                    data["sweep_indices"].append(meta.get("sweep_idx"))
                else:
                    data["repeat_indices"].append(None)
                    data["seeds"].append(None)
                    data["sweep_values"].append(None)
                    data["sweep_indices"].append(None)

        return attractors_data

    def save(self, path: Path) -> Path:
        """Save processed data to disk.

        Args:
            path: Directory where files will be saved.

        Returns:
            Path to the save directory.
        """
        if self._processed_data is None:
            raise ValueError("No processed data to save. Call process_batch() first.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save attractors data
        np.save(path / "attractors.npy", self._processed_data, allow_pickle=True)

        # Save config (compatible with SNNAnalyzer._load_config)
        config = {
            "n_runs": len(self.raw_outputs),
            "dt": self.dt,
            "clustering_params": self.clustering_params,
            "minimal_life_span_ms": self.minimal_life_span_ms,
            "total_duration_ms": self._total_duration_ms,
            "session_lengths_steps": self._session_lengths_steps,
            "repeat_durations_ms": self._repeat_durations_ms,
            "starts_ends_unit": "seconds",
            "metadata": self.metadata,
            "files": {
                "attractors": "attractors.npy",
            },
        }
        # Save as processor_config.json for SNNAnalyzer compatibility
        (path / "processor_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True)
        )
        batch_config = {
            "dt": self.dt,
            "total_duration_ms": self._total_duration_ms,
            "n_runs": len(self.raw_outputs),
            "repeats": [
                {
                    "repeat_idx": i,
                    "duration_ms": duration_ms,
                    "seed": self.metadata[i].get("seed") if i < len(self.metadata) else None,
                }
                for i, duration_ms in enumerate(self._repeat_durations_ms)
            ],
        }
        (path / "batch_config.json").write_text(
            json.dumps(batch_config, indent=2, sort_keys=True)
        )

        return path

    @classmethod
    def load_processed(cls, path: Path) -> dict:
        """Load previously processed data from disk."""
        path = Path(path)
        return np.load(path / "attractors.npy", allow_pickle=True).item()


class SNNBatchProcessorFactory:
    """Factory for creating SNNBatchProcessor instances.

    Conforms to the BatchProcessorFactory protocol for use with Pipeline.
    """

    def __init__(
        self,
        *,
        dt: float = 0.5e-3,
        clustering_params: dict | None = None,
        minimal_life_span_ms: float = 20.0,
    ) -> None:
        """Initialize the factory with processing parameters.

        Args:
            dt: Time step in seconds.
            clustering_params: Parameters for firing rate computation.
            minimal_life_span_ms: Minimum attractor duration in milliseconds.
        """
        self.dt = dt
        self.clustering_params = clustering_params
        self.minimal_life_span_ms = minimal_life_span_ms

    def __call__(
        self,
        raw_outputs: list[dict[str, Any]],
        metadata: list[dict[str, Any]],
        **kwargs: Any,
    ) -> SNNBatchProcessor:
        """Create a batch processor instance.

        Args:
            raw_outputs: List of raw simulation outputs.
            metadata: List of metadata dicts for each output.
            **kwargs: Additional configuration (overrides factory defaults).

        Returns:
            SNNBatchProcessor instance.
        """
        return SNNBatchProcessor(
            raw_outputs,
            metadata,
            dt=kwargs.get("dt", self.dt),
            clustering_params=kwargs.get("clustering_params", self.clustering_params),
            minimal_life_span_ms=kwargs.get("minimal_life_span_ms", self.minimal_life_span_ms),
        )

    @property
    def supports_batch(self) -> bool:
        """Return True to indicate batch processing support."""
        return True


class SNNProcessor(_BaseSimProcessor):
    """Process raw SNN simulation outputs into attractors_data.

    This class handles the transformation of raw spike data into structured
    attractor data that can be saved and later analyzed.
    """

    DEFAULT_CLUSTERING_PARAMS: dict = {
        "kernel_param": 20.0,
        "kernel_type": "gaussian",
    }
    DEFAULT_MINIMAL_LIFE_SPAN_MS: float = 20.0
    DEFAULT_DT: float = 0.5e-3

    def __init__(
            self,
            spikes_path: str | Path,
            clusters_path: str | Path | None = None,
            *,
            dt: float = DEFAULT_DT,
            clustering_params: dict | None = None,
            minimal_life_span_ms: float = DEFAULT_MINIMAL_LIFE_SPAN_MS,
    ) -> None:
        """Initialize the SNN processor.

        Args:
            spikes_path: Path to spike data file or directory of sessions.
            clusters_path: Optional path to cluster labels file.
            dt: Time step in seconds.
            clustering_params: Parameters for firing rate computation.
            minimal_life_span_ms: Minimum attractor duration in milliseconds.
        """
        super().__init__()
        self.spikes_path = Path(spikes_path)
        self.clusters_path = Path(clusters_path) if clusters_path else None
        self.dt = dt
        self.clustering_params = clustering_params or self.DEFAULT_CLUSTERING_PARAMS.copy()
        self.minimal_life_span_ms = minimal_life_span_ms

        # Lazy-loaded attributes
        self._sessions: tuple | None = None
        self._session_lengths_steps: list[int] | None = None
        self._total_duration_ms: float | None = None

    def _load_raw_data(self) -> tuple:
        """Load sessions from spike files.

        Returns:
            Tuple of (spikes, clusters) tuples per session.
        """
        if self._sessions is None:
            self._sessions = helpers.load_sessions(self.spikes_path, self.clusters_path)
        return self._sessions

    @property
    def sessions(self) -> tuple:
        """Return loaded sessions, loading if necessary."""
        return self._load_raw_data()

    @property
    def num_sessions(self) -> int:
        """Return the number of sessions."""
        return len(self.sessions)

    @property
    def total_duration_ms(self) -> float:
        """Return total simulation duration in milliseconds."""
        if self._total_duration_ms is None:
            self._total_duration_ms = helpers.get_total_duration_ms(self.sessions, self.dt)
        return self._total_duration_ms

    def get_config(self) -> dict[str, Any]:
        """Return configuration dict for use with analyzer."""
        return {
            "spikes_path": str(self.spikes_path),
            "clusters_path": str(self.clusters_path) if self.clusters_path else None,
            "dt": self.dt,
            "clustering_params": self.clustering_params,
            "minimal_life_span_ms": self.minimal_life_span_ms,
            "num_sessions": self.num_sessions,
            "session_lengths_steps": self._get_session_lengths_steps(),
            "total_duration_ms": self.total_duration_ms,
            "starts_ends_unit": "seconds",
        }

    @lru_cache()
    def _get_session_cluster_spike_rates(self) -> list[np.ndarray]:
        """Compute cluster firing rates for each session."""
        return helpers.get_session_cluster_spike_rates(
            self.sessions,
            self.clustering_params,
            self.dt,
        )

    def _get_session_cluster_activity(self) -> list[np.ndarray]:
        """Compute binary activity matrices for each session."""
        return helpers.get_session_cluster_activity(
            self._get_session_cluster_spike_rates()
        )

    def _get_session_attractors_data(self) -> list[dict]:
        """Extract attractors for each session."""
        return helpers.get_session_attractors_data(
            self._get_session_cluster_activity(),
            self.minimal_life_span_ms,
            self.dt * 1e3,
        )

    def _get_session_lengths_steps(self) -> list[int]:
        """Get session lengths in time steps."""
        if self._session_lengths_steps is None:
            self._session_lengths_steps = helpers.get_session_lengths_steps(
                self._get_session_cluster_activity()
            )
        return self._session_lengths_steps

    def process(self) -> dict:
        """Process raw spike data into attractors_data.

        This is the main entry point for the processing pipeline.
        Runs the full transformation from raw spikes to merged
        attractor data across all sessions.

        Returns:
            The attractors_data dictionary with times in seconds.
        """
        self.logger.info("Starting SNN processing pipeline.")

        # Get per-session attractors
        session_attractors = self._get_session_attractors_data()
        session_lengths = self._get_session_lengths_steps()

        # Validate no cross-session attractors
        helpers.validate_no_cross_simulation_attractors(session_attractors, session_lengths)

        # Merge across sessions
        attractors_data = helpers.merge_attractors_data(
            session_attractors,
            session_lengths,
            self.dt,
        )

        self._processed_data = attractors_data
        self.logger.info(
            f"Processing complete. Found {len(attractors_data)} unique attractors."
        )
        return attractors_data

    def save(
            self,
            path: Path,
            *,
            attractors_filename: str = "attractors.npy",
            config_filename: str = "processor_config.json",
    ) -> Path:
        """Save processed attractors_data and configuration to disk.

        Args:
            path: Directory where files will be saved.
            attractors_filename: Name for the attractors data file.
            config_filename: Name for the config file.

        Returns:
            Path to the save directory.
        """
        if self._processed_data is None:
            raise ValueError("No processed data to save. Call process() first.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save attractors data
        attractors_path = path / attractors_filename
        np.save(attractors_path, self._processed_data, allow_pickle=True)

        # Build config
        config = {
            "spikes_path": str(self.spikes_path),
            "clusters_path": str(self.clusters_path) if self.clusters_path else None,
            "dt": self.dt,
            "clustering_params": self.clustering_params,
            "minimal_life_span_ms": self.minimal_life_span_ms,
            "num_sessions": self.num_sessions,
            "session_lengths_steps": self._get_session_lengths_steps(),
            "total_duration_ms": self.total_duration_ms,
            "starts_ends_unit": "seconds",
            "files": {
                "attractors": attractors_filename,
            },
        }

        # Save config
        config_path = path / config_filename
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True))

        self.logger.info(f"Saved processed data to {path}")
        return path

    @classmethod
    def load_processed(
            cls,
            path: Path,
            *,
            config_filename: str = "processor_config.json",
    ) -> dict:
        """Load previously processed attractors_data from disk.

        Args:
            path: Directory containing saved files.
            config_filename: Name of the config file.

        Returns:
            The attractors_data dictionary.
        """
        path = Path(path)
        config_path = path / config_filename

        if config_path.exists():
            config = json.loads(config_path.read_text())
            files = config.get("files", {})
            attractors_filename = files.get("attractors", "attractors.npy")
        else:
            attractors_filename = "attractors.npy"

        attractors_path = path / attractors_filename
        return np.load(attractors_path, allow_pickle=True).item()

    @classmethod
    def load_config(
            cls,
            path: Path,
            *,
            config_filename: str = "processor_config.json",
    ) -> dict[str, Any]:
        """Load processing configuration from disk.

        Args:
            path: Directory containing saved files.
            config_filename: Name of the config file.

        Returns:
            The configuration dictionary.
        """
        path = Path(path)
        config_path = path / config_filename
        return json.loads(config_path.read_text())

    def get_cluster_spike_rate(self) -> np.ndarray:
        """Get concatenated cluster firing rates across all sessions.

        Returns:
            Array of firing rates with shape (n_clusters, T).
        """
        rates = self._get_session_cluster_spike_rates()
        return helpers.aggregate_series(rates, axis=1)

    def get_cluster_activity(self) -> np.ndarray:
        """Get concatenated binary activity matrices across all sessions.

        Returns:
            Boolean activity array with shape (n_clusters, T).
        """
        activity_mats = self._get_session_cluster_activity()
        return helpers.aggregate_series(activity_mats, axis=1)

    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Load and concatenate spikes across sessions.

        Returns:
            Tuple (spikes, clusters) where spikes is concatenated across
            sessions and clusters is the shared cluster label array.
        """
        sessions = self.sessions
        if not sessions:
            return np.empty(()), np.empty(())
        spikes = [spikes for spikes, _ in sessions]
        clusters = sessions[0][1]
        concat_spikes = np.concatenate(spikes, axis=0)
        return concat_spikes, clusters
