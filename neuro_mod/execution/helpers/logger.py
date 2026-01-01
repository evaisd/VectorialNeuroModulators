"""Simple logger utility for simulations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TextIO
import sys
import threading


_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
}


@dataclass
class Logger:
    """Simple structured logger with optional file output."""
    name: str = "simulation"
    level: str = "INFO"
    stream: TextIO = sys.stdout
    file_path: str | None = None
    include_timestamp: bool = True

    def __post_init__(self) -> None:
        """Normalize level and open file handle if provided."""
        normalized = self.level.upper()
        if normalized not in _LEVELS:
            raise ValueError(f"Unknown log level: {self.level}")
        self.level = normalized
        self._file_handle = open(self.file_path, "a", encoding="utf-8") if self.file_path else None
        self._lock = threading.Lock()

    def close(self) -> None:
        """Close any open file handle."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def attach_file(self, file_path: str) -> None:
        """Attach a log file for output.

        Args:
            file_path: Path to the file to append logs to.
        """
        if self._file_handle:
            self._file_handle.close()
        self.file_path = file_path
        self._file_handle = open(self.file_path, "a", encoding="utf-8")

    def log(self, level: str, message: str) -> None:
        """Log a message at the specified level.

        Args:
            level: Log level name (e.g., "INFO").
            message: Message to write.
        """
        normalized = level.upper()
        if normalized not in _LEVELS:
            raise ValueError(f"Unknown log level: {level}")
        if _LEVELS[normalized] < _LEVELS[self.level]:
            return

        prefix_parts = []
        if self.include_timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            prefix_parts.append(f"[{timestamp}]")
        prefix_parts.append(f"[{normalized}]")
        prefix_parts.append(f"{self.name}:")
        prefix = " ".join(prefix_parts)
        line = f"{prefix} {message}\n"

        with self._lock:
            self.stream.write(line)
            self.stream.flush()
            if self._file_handle:
                self._file_handle.write(line)
                self._file_handle.flush()

    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.log("DEBUG", message)

    def info(self, message: str) -> None:
        """Log an info message."""
        self.log("INFO", message)

    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.log("WARNING", message)

    def error(self, message: str) -> None:
        """Log an error message."""
        self.log("ERROR", message)

    def __enter__(self) -> "Logger":
        """Enter a logging context manager."""
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        """Exit a logging context manager and close resources."""
        self.close()
