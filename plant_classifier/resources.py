"""
Resources module for plant classifier.

This module handles initialization of heavy resources like model processors
and evaluation metrics. It depends on the config module but has no other
application dependencies to avoid circular imports.
"""

from __future__ import annotations

import evaluate
from transformers import ViTImageProcessor

from plant_classifier.config import DATA_CONFIG

# Load the processor
PROCESSOR = ViTImageProcessor.from_pretrained(DATA_CONFIG.model_path)

# Load the evaluation metrics
ACCURACY = evaluate.load("accuracy")
PRECISION = evaluate.load("precision")
RECALL = evaluate.load("recall")
F1 = evaluate.load("f1")

__all__ = [
    "PROCESSOR",
    "ACCURACY",
    "PRECISION",
    "RECALL",
    "F1",
]
