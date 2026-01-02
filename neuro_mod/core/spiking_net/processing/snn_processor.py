"""Processor for spiking neural network simulation outputs."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from neuro_mod.data_processing.base_processor import _BaseSimProcessor
from neuro_mod.core.spiking_net.processing.logic import session_helpers as helpers


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
