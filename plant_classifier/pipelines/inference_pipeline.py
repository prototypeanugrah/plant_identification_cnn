import pickle
from pathlib import Path
from typing import Dict

import mlflow
import torch
from PIL import Image

from plant_classifier.config import DATA_CONFIG, TRAIN_CONFIG
from plant_classifier.entities.drift_detector import DriftDetector
from plant_classifier.mlflow_databricks_integration import setup_mlflow


def inference_pipeline(image: Image.Image, include_drift: bool = True) -> Dict:
    """
    Inference pipeline with optional drift detection.

    Args:
        image (Image.Image): The image to predict. Pipeline will make
            predictions on the image.
        include_drift (bool): Whether to include drift detection. Default is True.

    Returns:
        Dict: A dictionary containing:
            - predictions: List of predicted labels and confidence scores
            - drift_metrics: Drift detection results (if include_drift=True)
    """

    print("=" * 80)
    print(f"Inference pipeline started with run name: {TRAIN_CONFIG.run_name}")
    print("=" * 80)

    # Setup MLFlow
    setup_mlflow()

    # Load the model as a pipeline (preprocessing is included)
    pipeline = mlflow.transformers.load_model(
        model_uri=f"runs:/{TRAIN_CONFIG.run_name}/{TRAIN_CONFIG.name}",
    )

    # Make prediction on the image
    pred = pipeline(image)

    result = {"predictions": pred}

    # Add drift detection if requested
    if include_drift:
        try:
            # Load reference statistics
            drift_ref_path = (
                Path(DATA_CONFIG.save_dir) / "drift_reference" / "reference_stats.pkl"
            )

            if drift_ref_path.exists():
                with open(drift_ref_path, "rb") as f:
                    reference_stats = pickle.load(f)

                # Create drift detector
                drift_detector = DriftDetector(reference_stats)

                # Determine device
                device = "cuda" if torch.cuda.is_available() else "cpu"

                # Extract model and processor from pipeline
                model = pipeline.model
                processor = pipeline.image_processor

                # Compute drift metrics
                drift_metrics = drift_detector.detect_drift(
                    image=image,
                    model=model,
                    processor=processor,
                    prediction_confidence=pred[0]["score"],
                    device=device,
                )

                result["drift_metrics"] = drift_metrics
            else:
                result["drift_metrics"] = {
                    "error": "Reference statistics not found. Run scripts/compute_drift_reference.py first."
                }
        except Exception as e:
            result["drift_metrics"] = {"error": f"Drift detection failed: {str(e)}"}

    return result


if __name__ == "__main__":
    inference_pipeline()
