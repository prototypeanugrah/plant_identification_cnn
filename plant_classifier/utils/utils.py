import base64
import io
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from mlflow.models import ModelSignature, infer_signature
from mlflow.transformers import generate_signature_output
from PIL import Image
from transformers import Pipeline, Trainer, pipeline


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


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def visualize_data(dataset, id2label, save_dir: str, ncols=4):
    """
    Visualize one sample from each class in the dataset with its label.
    Args:
        dataset: torch.utils.data.Dataset or HuggingFace Dataset (expects __getitem__ to return (image, label) or dict)
        id2label: dict mapping label indices to class names
        save_dir: directory to save the visualization
        ncols: number of columns in the plot grid
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
    plt.figure(figsize=(ncols * 3, nrows * 3))
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
    # plt.show()
    plt.savefig(save_dir)
    plt.close()


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
    print("Generating model signature...")
    image_classifier_pipeline = pipeline(
        task="image-classification",
        model=trainer.model,
        image_processor=image_processor,
    )

    # Encode the sample image as base64 so the signature records real inference payloads
    buffered = io.BytesIO()
    sample_image.convert("RGB").save(buffered, format="PNG")
    sample_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Wrap the encoded image in a DataFrame to capture the column name in the schema
    sample_input_df = pd.DataFrame({"image": [sample_image_base64]})

    # Use generate_signature_output for proper format
    sample_output = generate_signature_output(
        image_classifier_pipeline, sample_input_df
    )

    # Create signature from the same raw input type the pyfunc wrapper expects
    signature = infer_signature(sample_input_df, sample_output)

    return image_classifier_pipeline, signature
