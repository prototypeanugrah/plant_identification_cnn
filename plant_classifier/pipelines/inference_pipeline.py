from transformers import (
    Trainer,
    ViTForImageClassification,
)

from plant_classifier import DATA_CONFIG, PROCESSOR
from plant_classifier.entities.load_data import load_data


def inference_pipeline():
    """
    Inference pipeline.
    """

    _, _, test_data, id2label = load_data()

    model = ViTForImageClassification.from_pretrained(DATA_CONFIG.save_dir)

    eval_trainer = Trainer(model=model, processing_class=PROCESSOR)

    predictions = eval_trainer.predict(test_data)
    pred_labels = predictions.predictions.argmax(axis=-1)
    mapped_preds = [id2label[pred] for pred in pred_labels]
    actual_labels = [
        test_data.features["label"].names[test_data[i]["labels"]]
        for i in range(test_data.num_rows)
    ]
    print(f"Predicted labels: {mapped_preds[:10]}")
    print(f"Actual labels: {actual_labels[:10]}")
