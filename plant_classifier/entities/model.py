from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
)

from plant_classifier import DATA_CONFIG, PROCESSOR, TRAIN_CONFIG
from plant_classifier.entities.collator import collate_fn
from plant_classifier.entities.evaluator import compute_metrics


def train_model(
    train_data: Dataset,
    validation_data: Dataset,
):
    """
    Train the model.

    Args:
        config (PipelineConfig): The pipeline configuration.
        train_data (Dataset): The train data.
        validation_data (Dataset): The validation data.
    """
    labels = train_data.features["label"].names
    model = ViTForImageClassification.from_pretrained(
        TRAIN_CONFIG.model_path, num_labels=len(labels)
    )

    training_args = TrainingArguments(
        run_name="vit-base-plants-demo-v2",
        output_dir="./experiments",
        per_device_train_batch_size=DATA_CONFIG.batch_size,
        per_device_eval_batch_size=DATA_CONFIG.batch_size,
        eval_strategy="steps",
        logging_strategy="steps",
        num_train_epochs=TRAIN_CONFIG.epochs,
        # fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=TRAIN_CONFIG.lr,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        dataloader_num_workers=DATA_CONFIG.num_workers,
        report_to="mlflow",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_data,
        eval_dataset=validation_data,
        processing_class=PROCESSOR,
        compute_metrics=compute_metrics,
    )

    return trainer
