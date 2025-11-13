"""
Configuration module for plant classifier.

This module handles all configuration loading and validation.
It has no dependencies on other application modules to avoid circular imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from plant_classifier.configs import DataConfig, DeployConfig, TrainConfig


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Load the configuration files
data_config = load_config(Path("plant_classifier/configs/data.yaml"))
train_config = load_config(Path("plant_classifier/configs/train.yaml"))
deploy_config = load_config(Path("plant_classifier/configs/deploy.yaml"))

# Validate the configuration using Pydantic models
DATA_CONFIG = DataConfig.model_validate(data_config)
TRAIN_CONFIG = TrainConfig.model_validate(train_config)
DEPLOY_CONFIG = DeployConfig.model_validate(deploy_config)


__all__ = [
    "DATA_CONFIG",
    "TRAIN_CONFIG",
    "DEPLOY_CONFIG",
]
