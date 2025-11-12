"""
Configuration module for plant classifier.

This module handles all configuration loading and validation.
It has no dependencies on other application modules to avoid circular imports.
"""

from __future__ import annotations

from pathlib import Path

from plant_classifier.configs import DataConfig, TrainConfig
from plant_classifier.utils.utils import load_config

# Load the configuration files
data_config = load_config(Path("plant_classifier/configs/data.yaml"))
train_config = load_config(Path("plant_classifier/configs/train.yaml"))

# Validate the configuration using Pydantic models
DATA_CONFIG = DataConfig.model_validate(data_config)
TRAIN_CONFIG = TrainConfig.model_validate(train_config)

__all__ = [
    "DATA_CONFIG",
    "TRAIN_CONFIG",
]
