#!/usr/bin/env python3
"""Download or copy a LaTeX template into a project directory.

Examples:
  python scripts/fetch_template.py --source https://example.com/template.zip --dest templates --extract
  python scripts/fetch_template.py --source /path/to/template.zip --dest templates --extract
  python scripts/fetch_template.py --source /path/to/revtex --dest templates
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse


def is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def as_local_path(value: str) -> Path:
    parsed = urlparse(value)
    if parsed.scheme == "file":
        return Path(parsed.path)
    return Path(value)


def download(url: str, dest: Path, timeout: int) -> None:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        dest.write_bytes(response.read())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch a LaTeX template from URL or local path.")
    parser.add_argument("--source", required=True, help="URL or local path (file, zip, or directory).")
    parser.add_argument("--dest", required=True, help="Destination directory to write into.")
    parser.add_argument("--name", help="Optional output name (file or directory).")
    parser.add_argument("--extract", action="store_true", help="Extract zip archives into destination.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--timeout", type=int, default=30, help="Download timeout in seconds.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dest_dir = Path(args.dest)
    ensure_dir(dest_dir)

    if is_http_url(args.source):
        filename = args.name or Path(urlparse(args.source).path).name or "template.zip"
        target_file = dest_dir / filename
        if target_file.exists() and not args.force:
            print(f"Refusing to overwrite existing file: {target_file}", file=sys.stderr)
            return 2
        download(args.source, target_file, args.timeout)
        source_path = target_file
    else:
        source_path = as_local_path(args.source)
        if not source_path.exists():
            print(f"Source not found: {source_path}", file=sys.stderr)
            return 2

    if source_path.is_dir():
        target_dir = dest_dir / (args.name or source_path.name)
        if target_dir.exists() and not args.force:
            print(f"Refusing to overwrite existing directory: {target_dir}", file=sys.stderr)
            return 2
        shutil.copytree(source_path, target_dir, dirs_exist_ok=args.force)
        print(f"Copied directory to {target_dir}")
        return 0

    if args.extract and source_path.suffix.lower() == ".zip":
        extract_dir = dest_dir / (args.name or "")
        ensure_dir(extract_dir)
        with zipfile.ZipFile(source_path, "r") as archive:
            archive.extractall(extract_dir)
        print(f"Extracted archive to {extract_dir}")
        return 0

    target_file = dest_dir / (args.name or source_path.name)
    if target_file.exists() and not args.force:
        print(f"Refusing to overwrite existing file: {target_file}", file=sys.stderr)
        return 2
    shutil.copy2(source_path, target_file)
    print(f"Copied file to {target_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
