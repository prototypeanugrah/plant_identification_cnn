"""
Data drift detection module using Wasserstein distance.

This module provides functionality to detect drift in production images
compared to training data by measuring:
1. Visual drift: pixel distribution changes
2. Feature drift: model embedding distribution changes
3. Performance drift: prediction confidence monitoring
"""

from typing import Dict

import numpy as np
import torch
from PIL import Image
from scipy.stats import wasserstein_distance
from transformers import ViTForImageClassification, ViTImageProcessor


class DriftDetector:
    """Detects data drift using Wasserstein distance."""

    def __init__(
        self,
        reference_stats: Dict[str, np.ndarray],
        thresholds: Dict[str, float] | None = None,
    ):
        """
        Initialize drift detector with reference statistics.

        Args:
            reference_stats: Dictionary containing reference distributions:
                - pixel_dist_r, pixel_dist_g, pixel_dist_b: RGB pixel distributions
                - feature_mean: mean of feature embeddings
                - feature_std: std of feature embeddings
                - confidence_mean: mean prediction confidence
                - confidence_std: std prediction confidence
            thresholds: Optional custom thresholds for drift alerts
        """
        self.reference_stats = reference_stats

        # Calibrated thresholds based on validation data (95th percentile)
        # These were computed using scripts/calibrate_drift_thresholds.py
        self.thresholds = thresholds or {
            "pixel_drift": 0.5005,  # Normalized pixel drift threshold
            "feature_drift": 0.8523,  # Feature embedding drift threshold
            "confidence_drop": 0.2562,  # Confidence drop threshold (~25%)
        }

    def compute_pixel_drift(
        self, image: Image.Image, processor: ViTImageProcessor
    ) -> Dict[str, float]:
        """
        Compute visual drift using pixel distributions.

        Args:
            image: Input PIL image
            processor: ViT image processor for preprocessing

        Returns:
            Dictionary with per-channel Wasserstein distances and overall drift
        """
        # Use the processor to preprocess the image (same as model input)
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"][0]  # Shape: (3, 224, 224)

        # Convert to numpy and extract per-channel distributions
        pixel_values = pixel_values.numpy()
        pixel_r = pixel_values[0].flatten()
        pixel_g = pixel_values[1].flatten()
        pixel_b = pixel_values[2].flatten()

        # Compute Wasserstein distance for each channel
        drift_r = wasserstein_distance(pixel_r, self.reference_stats["pixel_dist_r"])
        drift_g = wasserstein_distance(pixel_g, self.reference_stats["pixel_dist_g"])
        drift_b = wasserstein_distance(pixel_b, self.reference_stats["pixel_dist_b"])

        # Overall pixel drift (average across channels)
        overall_drift = (drift_r + drift_g + drift_b) / 3

        return {
            "drift_r": float(drift_r),
            "drift_g": float(drift_g),
            "drift_b": float(drift_b),
            "overall_pixel_drift": float(overall_drift),
            "pixel_drift_alert": overall_drift > self.thresholds["pixel_drift"],
        }

    def compute_feature_drift(
        self,
        model: ViTForImageClassification,
        processor: ViTImageProcessor,
        image: Image.Image,
        device: str = "cpu",
    ) -> Dict[str, float]:
        """
        Compute feature drift using model embeddings.

        Args:
            model: ViT model
            processor: Image processor
            image: Input PIL image
            device: Device to run inference on

        Returns:
            Dictionary with feature drift metrics
        """
        model.eval()
        model.to(device)

        # Process image and extract features
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            # Get hidden states (embeddings from last layer before classifier)
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Last layer hidden states

            # Use CLS token embedding (first token)
            cls_embedding = hidden_states[:, 0, :].cpu().numpy().flatten()

        # Compute Wasserstein distance using feature statistics
        # Since we have mean/std from reference, we approximate with normal distribution
        ref_mean = self.reference_stats["feature_mean"]
        ref_std = self.reference_stats["feature_std"]

        # Simple drift metric: normalized distance from reference distribution
        # Using z-score based approach
        feature_drift = np.mean(np.abs((cls_embedding - ref_mean) / (ref_std + 1e-8)))

        return {
            "feature_drift": float(feature_drift),
            "feature_drift_alert": feature_drift > self.thresholds["feature_drift"],
        }

    def compute_confidence_drift(
        self, confidence: float
    ) -> Dict[str, float | bool]:
        """
        Compute performance drift based on prediction confidence.

        Args:
            confidence: Current prediction confidence score

        Returns:
            Dictionary with confidence drift metrics
        """
        ref_confidence = self.reference_stats["confidence_mean"]

        # Calculate relative drop in confidence
        confidence_drop = (ref_confidence - confidence) / ref_confidence

        return {
            "current_confidence": float(confidence),
            "reference_confidence": float(ref_confidence),
            "confidence_drop": float(confidence_drop),
            "confidence_alert": confidence_drop > self.thresholds["confidence_drop"],
        }

    def detect_drift(
        self,
        image: Image.Image,
        model: ViTForImageClassification,
        processor: ViTImageProcessor,
        prediction_confidence: float,
        device: str = "cpu",
    ) -> Dict[str, Dict[str, float | bool]]:
        """
        Perform complete drift detection.

        Args:
            image: Input PIL image
            model: ViT model
            processor: Image processor
            prediction_confidence: Confidence score from prediction
            device: Device to run inference on

        Returns:
            Comprehensive drift report
        """
        pixel_drift = self.compute_pixel_drift(image, processor)
        feature_drift = self.compute_feature_drift(model, processor, image, device)
        confidence_drift = self.compute_confidence_drift(prediction_confidence)

        # Overall drift assessment
        any_alert = (
            pixel_drift["pixel_drift_alert"]
            or feature_drift["feature_drift_alert"]
            or confidence_drift["confidence_alert"]
        )

        return {
            "visual_drift": pixel_drift,
            "feature_drift": feature_drift,
            "performance_drift": confidence_drift,
            "overall_alert": any_alert,
        }


def extract_reference_statistics(
    model: ViTForImageClassification,
    processor: ViTImageProcessor,
    images: list[Image.Image],
    confidences: list[float],
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """
    Extract reference statistics from training/validation data.

    Args:
        model: ViT model
        processor: Image processor
        images: List of training/validation images
        confidences: List of confidence scores from validation predictions
        device: Device to run inference on

    Returns:
        Dictionary of reference statistics
    """
    model.eval()
    model.to(device)

    # Collect pixel distributions
    all_pixels_r, all_pixels_g, all_pixels_b = [], [], []
    all_features = []

    print(f"Extracting reference statistics from {len(images)} images...")

    for idx, image in enumerate(images):
        if idx % 100 == 0:
            print(f"Processing image {idx}/{len(images)}...")

        # Pixel distributions using processor (same as production)
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"][0]  # Shape: (3, 224, 224)
        pixel_values = pixel_values.numpy()

        all_pixels_r.extend(pixel_values[0].flatten())
        all_pixels_g.extend(pixel_values[1].flatten())
        all_pixels_b.extend(pixel_values[2].flatten())

        # Feature embeddings
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            cls_embedding = hidden_states[:, 0, :].cpu().numpy().flatten()
            all_features.append(cls_embedding)

    # Compute statistics
    all_features = np.array(all_features)

    reference_stats = {
        # Sample pixel distributions (use subset for efficiency)
        "pixel_dist_r": np.array(all_pixels_r[::10]),  # Subsample
        "pixel_dist_g": np.array(all_pixels_g[::10]),
        "pixel_dist_b": np.array(all_pixels_b[::10]),
        # Feature statistics
        "feature_mean": np.mean(all_features, axis=0),
        "feature_std": np.std(all_features, axis=0),
        # Confidence statistics
        "confidence_mean": np.mean(confidences),
        "confidence_std": np.std(confidences),
    }

    print("Reference statistics extracted successfully!")
    return reference_stats
