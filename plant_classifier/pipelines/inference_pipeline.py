from typing import Dict

import mlflow
from PIL import Image

from plant_classifier.config import TRAIN_CONFIG


def inference_pipeline(image: Image.Image) -> Dict:
    """
    Inference pipeline.

    Args:
        image (Image.Image): The image to predict. Pipeline will make
        predictions on the image.

    Returns:
        Dict: A dictionary containing the predicted labels and the confidence scores.
    """
    # Set MLflow tracking URI to match training
    mlflow.set_tracking_uri(TRAIN_CONFIG.mlflow_tracking_uri)

    registered_model_name = "PlantClassifierHfTraining"
    model_version = "latest"
    model_uri = f"models:/{registered_model_name}/{model_version}"

    # Load the model as a pipeline (preprocessing is included)
    pipeline = mlflow.transformers.load_model(
        model_uri=model_uri,
    )

    # Make prediction on the image
    pred = pipeline(image)
    return pred


if __name__ == "__main__":
    inference_pipeline()
