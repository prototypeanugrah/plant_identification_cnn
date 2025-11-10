from pathlib import Path

import evaluate
from transformers import ViTImageProcessor

from plant_classifier.configs import DataConfig, TrainConfig
from plant_classifier.utils.utils import load_config

# Load the configuration
data_config = load_config(Path("plant_classifier/configs/data.yaml"))
train_config = load_config(Path("plant_classifier/configs/train.yaml"))

# Validate the configuration
DATA_CONFIG = DataConfig.model_validate(data_config)
TRAIN_CONFIG = TrainConfig.model_validate(train_config)

# Load the processor
PROCESSOR = ViTImageProcessor.from_pretrained(DATA_CONFIG.model_path)

# Load the evaluation metrics
ACCURACY = evaluate.load("accuracy")
PRECISION = evaluate.load("precision")
RECALL = evaluate.load("recall")
F1 = evaluate.load("f1")


__all__ = [
    "PROCESSOR",
    "DATA_CONFIG",
    "TRAIN_CONFIG",
    "ACCURACY",
    "PRECISION",
    "RECALL",
    "F1",
]
