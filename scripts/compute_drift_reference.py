"""
Script to compute reference statistics for drift detection.

This script should be run after training to establish baseline statistics
from validation data for drift monitoring in production.
"""

import pickle
from pathlib import Path

import mlflow
import torch
from datasets import load_from_disk

from plant_classifier.config import DATA_CONFIG, TRAIN_CONFIG
from plant_classifier.entities.drift_detector import extract_reference_statistics


def main():
    """Compute and save reference statistics from validation data."""
    print("=" * 80)
    print("Computing Reference Statistics for Drift Detection")
    print("=" * 80)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load validation data
    print("\nLoading validation data...")
    splits_dir = Path(DATA_CONFIG.save_dir) / "data_splits"
    val_path = splits_dir / "validation"

    if not val_path.exists():
        raise FileNotFoundError(
            f"Validation data not found at {val_path}. "
            "Please run training pipeline first to generate data splits."
        )

    validation_data = load_from_disk(str(val_path))
    print(f"Loaded {len(validation_data)} validation samples")

    # Load model from MLflow
    print("\nLoading model from MLflow...")
    mlflow.set_tracking_uri(TRAIN_CONFIG.mlflow_tracking_uri)

    registered_model_name = "PlantClassifierHfTraining"
    model_version = "latest"
    model_uri = f"models:/{registered_model_name}/{model_version}"

    # Load the model
    pipeline = mlflow.transformers.load_model(model_uri=model_uri)

    # Extract the actual model and processor from the pipeline
    # The pipeline is a transformers ImageClassificationPipeline
    model = pipeline.model
    processor = pipeline.image_processor

    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Processor loaded: {processor.__class__.__name__}")

    # Get images and compute predictions for confidence scores
    print("\nComputing predictions to extract confidence scores...")
    images = []
    confidences = []

    # Use a subset of validation data for efficiency (adjust as needed)
    num_samples = min(500, len(validation_data))  # Use up to 500 samples
    print(f"Using {num_samples} samples for reference statistics")

    for idx in range(num_samples):
        sample = validation_data[idx]
        image = sample["image"]
        images.append(image)

        # Get prediction
        pred = pipeline(image)
        confidences.append(pred[0]["score"])

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{num_samples} samples")

    # Extract reference statistics
    print("\nExtracting reference statistics...")
    reference_stats = extract_reference_statistics(
        model=model,
        processor=processor,
        images=images,
        confidences=confidences,
        device=device,
    )

    # Save reference statistics
    output_dir = Path(DATA_CONFIG.save_dir) / "drift_reference"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "reference_stats.pkl"

    print(f"\nSaving reference statistics to {output_file}...")
    with open(output_file, "wb") as f:
        pickle.dump(reference_stats, f)

    print("\n" + "=" * 80)
    print("Reference Statistics Summary:")
    print("=" * 80)
    print(
        f"Pixel distributions: R, G, B channels ({len(reference_stats['pixel_dist_r'])} samples each)"
    )
    print(f"Feature mean shape: {reference_stats['feature_mean'].shape}")
    print(f"Feature std shape: {reference_stats['feature_std'].shape}")
    print(f"Confidence mean: {reference_stats['confidence_mean']:.4f}")
    print(f"Confidence std: {reference_stats['confidence_std']:.4f}")
    print("\n" + "=" * 80)
    print("✓ Reference statistics saved successfully!")
    print(f"✓ Location: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
