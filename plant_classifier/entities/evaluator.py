from transformers import EvalPrediction

from plant_classifier.resources import ACCURACY, F1, PRECISION, RECALL


def compute_metrics(p: EvalPrediction):
    """
    Compute metrics for the model.

    Args:
        p (EvalPrediction): Prediction output from the model.

    Returns:
        dict: Dictionary containing the metrics.
    """

    preds = p.predictions.argmax(axis=-1)
    labels = p.label_ids

    acc = ACCURACY.compute(predictions=preds, references=labels)
    prec = PRECISION.compute(
        predictions=preds, references=labels, average="macro"
    )  # macro for multiclass
    rec = RECALL.compute(
        predictions=preds, references=labels, average="macro"
    )  # macro for multiclass
    f1_metric = F1.compute(
        predictions=preds, references=labels, average="macro"
    )  # macro for multiclass
    return {
        "accuracy": acc["accuracy"],
        "precision": prec["precision"],
        "recall": rec["recall"],
        "f1": f1_metric["f1"],
    }
