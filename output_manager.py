"""
output_manager.py
Helpers for saving generated figures and LaTeX table files into structured
phase-specific output folders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt


def get_phase_output_dirs(output_root: str | Path, phase_name: str) -> tuple[Path, Path]:
    """Create and return the figure/table directories for one phase."""
    phase_root = Path(output_root) / phase_name
    figure_dir = phase_root / "figures"
    table_dir = phase_root / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir, table_dir


def save_figures(
    figures: Sequence[tuple[str, plt.Figure]],
    output_dir: str | Path,
    dpi: int = 300,
) -> None:
    """Save named Matplotlib figures as PNG files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for filename, figure in figures:
        figure.savefig(output_path / filename, dpi=dpi, bbox_inches="tight")


def write_text_outputs(files: Mapping[str, str], output_dir: str | Path) -> None:
    """Write named text artifacts such as LaTeX snippets."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        (output_path / filename).write_text(content, encoding="utf-8")
