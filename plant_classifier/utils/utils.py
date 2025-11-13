import base64
import io
from typing import Any, Dict, Tuple

import pandas as pd
from datasets import Dataset
from mlflow.models import ModelSignature, infer_signature
from mlflow.transformers import generate_signature_output
from PIL import Image
from transformers import Pipeline, Trainer, pipeline

from plant_classifier.config import TRAIN_CONFIG


def class_names(dataset: Dataset) -> Dict[int, str]:
    """
    Get the class names from the dataset.

    Args:
        dataset (Dataset): The dataset to get the class names from.

    Returns:
        Dict[int, str]: A dictionary mapping class IDs to class names.
    """
    labels = dataset.features["label"].names
    id2label = {i: label for i, label in enumerate(labels)}

    # for key, value in id2label.items():
    #     print(f"Id: {key}, Label: {value}")

    return id2label


def create_signature(
    sample_image: Image.Image, trainer: Trainer, image_processor: Any
) -> Tuple[Pipeline, ModelSignature]:
    """
    Create a signature for the model.

    Args:
        sample_image (PIL.Image.Image): Raw sample image.
        trainer (Trainer): The trainer.

    Returns:
        ModelSignature: The signature.
    """

    if image_processor is None:
        msg = "image_processor must be provided to create_signature to avoid circular imports."
        raise ValueError(msg)

    # Create a pipeline for signature generation
    image_classifier_pipeline = pipeline(
        task=TRAIN_CONFIG.task,
        model=trainer.model,
        image_processor=image_processor,
    )

    # Encode the sample image as base64 so the signature records real inference payloads
    buffered = io.BytesIO()
    sample_image.convert("RGB").save(buffered, format="PNG")
    sample_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Wrap the encoded image in a DataFrame to capture the column name in the schema
    sample_input_df = pd.DataFrame({"image": [sample_image_base64]})

    # Generate signature using MLflow's transformer-specific function
    # This properly handles the pipeline output format (list of dicts)
    signature_output = generate_signature_output(
        image_classifier_pipeline, sample_image_base64
    )

    # Create signature from the same raw input type the pyfunc wrapper expects
    signature = infer_signature(sample_input_df, signature_output)

    return image_classifier_pipeline, signature
