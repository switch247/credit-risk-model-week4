"""Central configuration for the insurance risk analytics project."""
from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project_name: str = "Insurance Risk Analytics"

    # Base paths
    root_dir: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = root_dir / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    outputs_dir: Path = root_dir / "outputs"
    figures_dir: Path = outputs_dir / "figures"
    reports_dir: Path = outputs_dir / "reports"

    # File names
    raw_data_file: str = "MachineLearningRating_v3.txt"
    processed_data_file: str = "insurance_clean.csv"

    @property
    def raw_data_path(self) -> Path:
        return self.raw_data_dir / self.raw_data_file

    @property
    def processed_data_path(self) -> Path:
        return self.processed_data_dir / self.processed_data_file


settings = Settings()

# Backward-compatible mapping used by legacy modules
DATA_PATHS = {
    "raw": str(settings.raw_data_dir),
    "processed": str(settings.processed_data_dir),
    "raw_reviews": str(settings.raw_data_path),
    "processed_reviews": str(settings.processed_data_path),
    "figures": str(settings.figures_dir),
    "reports": str(settings.reports_dir),
}
