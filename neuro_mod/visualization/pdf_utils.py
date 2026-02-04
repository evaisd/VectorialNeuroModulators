"""Utilities for exporting image folders to PDF."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def folder_plots_to_pdf(
    folder_path: str | Path,
    output_path: str | Path | None = None,
    extensions: Optional[Iterable[str]] = None,
    sort_key: Optional[Callable[[Path], str]] = None,
) -> Path:
    """Combine plot images in a folder into a single multi-page PDF.

    Args:
        folder_path: Directory containing images.
        output_path: Output PDF path. Defaults to "<folder>/plots.pdf".
        extensions: Iterable of file extensions (with or without dot).
        sort_key: Optional key function to sort paths (defaults to name).

    Returns:
        Path to the generated PDF file.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"Folder does not exist: {folder}")

    if output_path is None:
        output_path = folder / "plots.pdf"

    if extensions is None:
        extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    normalized_exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}"
                       for ext in extensions}

    image_paths = [
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in normalized_exts
    ]
    if not image_paths:
        raise ValueError(f"No image files found in {folder} with {normalized_exts}")

    image_paths.sort(key=sort_key or (lambda p: p.name))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        for image_path in image_paths:
            image = mpimg.imread(image_path)
            height, width = image.shape[:2]
            dpi = 150  # Keep a reasonable size without exploding file size.
            fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
            ax.imshow(image)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

    return output_path


def image_to_pdf(
    image_path: str | Path,
    output_path: str | Path,
    *,
    dpi: int = 150,
) -> Path:
    """Save a raster image into a single-page PDF (rasterized)."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise ValueError(f"Image does not exist: {image_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = mpimg.imread(image_path)
    height, width = image.shape[:2]
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi, layout="tight")
    ax.imshow(image)
    ax.axis("off")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path
