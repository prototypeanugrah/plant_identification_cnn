from typing import Any, Dict

from plant_classifier import PROCESSOR


def preprocess(
    example: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run the ViT image processor on a batch of PIL images.

    HF `datasets.map(batched=True)` passes in a *dict of lists*.
    We run the processor on the whole list, then return **one item
    per original example** so the dataset stays aligned.

    Returns
    -------
    dict with keys:
        pixel_values : list[Tensor]  # each of shape (3, 224, 224)
        label        : original labels (unchanged)
    """
    inputs = PROCESSOR(example["image"], return_tensors="pt")["pixel_values"]
    return {"pixel_values": [x for x in inputs], "labels": example["label"]}
