"""CLI helpers for execution scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import shlex
import sys


def resolve_path(root: Path, value: Path | str) -> Path:
    """Return an absolute path, resolving relative paths against root."""
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def save_cmd(metadata_dir: Path, argv: Sequence[str] | None = None, executable: str | None = None) -> Path:
    """Save the invoking command to metadata/cmd.txt and return the path."""
    metadata_dir.mkdir(parents=True, exist_ok=True)
    argv = list(argv) if argv is not None else list(sys.argv)
    executable = executable or sys.executable
    cmd = shlex.join([executable, *argv])
    cmd_path = metadata_dir / "cmd.txt"
    cmd_path.write_text(f"{cmd}\n")
    return cmd_path
