"""Configuration loading from YAML."""

from pathlib import Path

import yaml


def load_config(config_path: str | Path = "configs/experiment.yaml") -> dict:
    """Load experiment configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent.parent
