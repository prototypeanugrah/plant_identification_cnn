from typing import Any, Dict, List

import torch


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate the batch.

    Args:
        batch (list): List of dictionaries containing the pixel values and labels.

    Returns:
        dict: Dictionary containing the collated batch.
    """
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }
