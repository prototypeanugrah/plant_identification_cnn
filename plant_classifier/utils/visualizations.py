import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from matplotlib.figure import Figure
from PIL import Image

from plant_classifier.pipelines.inference_pipeline import inference_pipeline


def analyze_image_dimensions(
    dataset: Dataset,
    split: str,
) -> Figure:
    """
    Analyze and plot the distribution of image dimensions in the dataset.

    Args:
        dataset (Dataset): The HuggingFace dataset containing images
        split (str): The split name (e.g., "train", "val", "test")

    Returns:
        Figure: The generated figure object
    """
    # Extract dimensions from all images
    widths = []
    heights = []
    aspect_ratios = []

    for _, item in enumerate(dataset):
        img = item["image"]
        w, h = img.size
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h)

    # Convert to numpy arrays for statistics
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Image Dimension Analysis - {split.capitalize()} Dataset",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Width vs Height scatter
    axes[0].scatter(widths, heights, alpha=0.5, color="green", s=30)
    axes[0].set_xlabel("Width (pixels)")
    axes[0].set_ylabel("Height (pixels)")
    axes[0].set_title("Width vs Height Scatter Plot")
    axes[0].grid(True, alpha=0.3)

    # Add diagonal line for square images
    max_dim = max(widths.max(), heights.max())
    axes[0].plot(
        [0, max_dim], [0, max_dim], "r--", linewidth=2, label="Square (1:1)", alpha=0.7
    )
    axes[0].legend()

    # Plot 2: Aspect ratio distribution
    axes[1].hist(aspect_ratios, bins=30, color="purple", edgecolor="black", alpha=0.7)
    axes[1].axvline(
        aspect_ratios.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {aspect_ratios.mean():.2f}",
    )
    axes[1].axvline(
        1.0, color="orange", linestyle=":", linewidth=2, label="Square (1.0)", alpha=0.7
    )
    axes[1].set_xlabel("Aspect Ratio (Width/Height)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution of Aspect Ratios")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


def dist_labels(dataset: Dataset, split: str) -> Figure:
    """
    Plot the distribution of the labels.

    Args:
        dataset (Dataset): The HuggingFace dataset containing labels
        split (str): The split name (e.g., "train", "val", "test")

    Returns:
        Figure: The generated figure object
    """
    # Get label names if available
    label_names = (
        dataset.features["label"].names
        if hasattr(dataset.features["label"], "names")
        else None
    )

    # Plot with label names
    label_counts = pd.Series(dataset["label"]).value_counts().sort_index()
    if label_names:
        label_counts.index = [label_names[i] for i in label_counts.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    label_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Labels in {split.capitalize()} Dataset")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    return fig


def visualize_data(
    dataset: Dataset,
    id2label: dict,
    ncols: int = 4,
) -> Figure:
    """
    Visualize one sample from each class in the dataset with its label.

    Args:
        dataset: HuggingFace Dataset
        id2label: dict mapping label indices to class names
        ncols: number of columns in the plot grid

    Returns:
        Figure: The generated figure object
    """
    seen = set()
    samples = []
    for item in dataset:
        # Handle raw HuggingFace dataset (dict)
        if isinstance(item, dict):
            img = item["image"]
            label = item["label"]
        else:
            img, label = item
        if label not in seen:
            samples.append((img, label))
            seen.add(label)
        if len(seen) == len(id2label):
            break
    n_classes = len(samples)
    nrows = int(np.ceil(n_classes / ncols))
    fig = plt.figure(figsize=(ncols * 3, nrows * 3))
    for i, (img, label) in enumerate(samples):
        plt.subplot(nrows, ncols, i + 1)
        # Handle PIL Image
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        # Handle torch.Tensor
        elif isinstance(img, torch.Tensor):
            img_np = img.numpy()
            if img_np.shape[0] in [1, 3]:  # C,H,W -> H,W,C
                img_np = np.transpose(img_np, (1, 2, 0))
            if img_np.shape[-1] == 1:
                img_np = img_np.squeeze(-1)
        # Handle numpy array
        elif isinstance(img, np.ndarray):
            img_np = img
        else:
            # Fallback: try to convert to numpy array
            img_np = np.array(img)
        plt.imshow(img_np)
        plt.title(id2label[label])
        plt.axis("off")
    plt.tight_layout()

    return fig


def visualize_test_predictions(
    dataset: Dataset,
    id2label: dict,
    n_images: int = 12,
    nrows: int = 3,
    ncols: int = 4,
    pipeline=None,
) -> Figure:
    """
    Visualize predictions on test set images in a grid layout.

    Args:
        dataset: HuggingFace Dataset containing test images
        id2label: dict mapping label indices to class names
        n_images: number of images to visualize (default: 12)
        nrows: number of rows in grid (default: 3)
        ncols: number of columns in grid (default: 4)
        pipeline: Optional pretrained pipeline to use for predictions.
                 If None, will load from MLflow registry via inference_pipeline.

    Returns:
        Figure: The generated figure object
    """
    # Sample n_images from the dataset
    total_samples = len(dataset)
    if n_images > total_samples:
        n_images = total_samples

    # Get evenly spaced indices to sample diverse images
    indices = np.linspace(0, total_samples - 1, n_images, dtype=int)

    # Create figure
    fig = plt.figure(figsize=(ncols * 3, nrows * 3.5))
    fig.suptitle(
        "Test Set Predictions: Predicted vs Actual Labels",
        fontsize=16,
        fontweight="bold",
    )

    correct_count = 0

    for idx, sample_idx in enumerate(indices):
        item = dataset[int(sample_idx)]
        image = item["image"]
        true_label = item["label"]

        # Get prediction - use provided pipeline or load from MLflow
        if pipeline is not None:
            # Use the provided pipeline directly
            pred = pipeline(image)
            pred_dict = pred[0] if isinstance(pred, list) else pred
            pred_label_name = pred_dict["label"]
            confidence = pred_dict["score"]
        else:
            # Load from MLflow registry via inference_pipeline
            result = inference_pipeline(image, include_drift=False)
            predictions = result["predictions"]
            pred_dict = predictions[0] if isinstance(predictions, list) else predictions
            pred_label_name = pred_dict["label"]
            confidence = pred_dict["score"]

        # Convert predicted label name back to index
        label2id = {v: k for k, v in id2label.items()}
        pred_label_idx = label2id.get(pred_label_name)

        # Check if prediction is correct
        is_correct = pred_label_idx == true_label
        if is_correct:
            correct_count += 1

        # Plot image
        plt.subplot(nrows, ncols, idx + 1)
        img_np = np.array(image)
        plt.imshow(img_np)

        # Create title with prediction and true label
        true_name = id2label[true_label]
        title = f"Pred: {pred_label_name} ({confidence:.2%})\nTrue: {true_name}"

        # Color code: green for correct, red for incorrect
        title_color = "green" if is_correct else "red"
        plt.title(title, color=title_color, fontweight="bold", fontsize=9)
        plt.axis("off")

    plt.tight_layout()

    return fig
