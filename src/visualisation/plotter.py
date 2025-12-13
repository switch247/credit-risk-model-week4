"""Plotting utility with auto-save and inline display."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns

from src.config.settings import settings


class Plotter:
    """Lightweight plotting helpers for notebooks and scripts.

    - Always displays plots inline.
    - Automatically saves PNG files to the configured figures directory using a slugified title.
    - Caller only needs to pass the title; no manual paths.
    """

    def __init__(self, figures_dir: Optional[Path] = None):
        sns.set_theme(style="whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)
        self.figures_dir = Path(figures_dir) if figures_dir else settings.figures_dir
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _slugify(title: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", title.strip()).strip("_").lower()
        return slug or "figure"

    def _finalize(self, title: str | None, xlabel: str | None, ylabel: str | None):
        if title:
            plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        plt.tight_layout()

        if title:
            filename = f"{self._slugify(title)}.png"
            out_path = self.figures_dir / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=300)

        plt.show()
        plt.close()

    def plot_histogram(self, df, column, title=None, xlabel=None, ylabel="Count", bins=20):
        """Plot and save a histogram with KDE."""
        plt.figure()
        sns.histplot(data=df, x=column, bins=bins, kde=True)
        self._finalize(title or f"Distribution of {column}", xlabel or column, ylabel)

    def plot_bar(self, df, x, y, title=None, xlabel=None, ylabel=None):
        """Plot and save a bar chart."""
        plt.figure()
        sns.barplot(data=df, x=x, y=y)
        plt.xticks(rotation=45)
        self._finalize(title or f"{y} by {x}", xlabel or x, ylabel or y)

    def plot_time_series(self, df, date_col, value_col, title=None, xlabel=None, ylabel=None):
        """Plot and save a time series line chart."""
        plt.figure()
        sns.lineplot(data=df, x=date_col, y=value_col, marker="o")
        plt.xticks(rotation=45)
        self._finalize(title or f"{value_col} over Time", xlabel or date_col, ylabel or value_col)

    def plot_box(self, df, y, x=None, title=None, xlabel=None, ylabel=None):
        """Plot and save a boxplot (optionally grouped by x)."""
        plt.figure()
        sns.boxplot(data=df, x=x, y=y)
        plt.xticks(rotation=45)
        self._finalize(title or f"Distribution of {y}", xlabel or (x if x else ""), ylabel or y)
