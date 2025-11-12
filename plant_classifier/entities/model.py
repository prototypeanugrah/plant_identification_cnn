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
    run_name: str,
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

    # Get the id2label and label2id mappings
    labels = train_data.features["label"].names
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}

    model = ViTForImageClassification.from_pretrained(
        TRAIN_CONFIG.model_path,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        run_name=run_name,
        output_dir=DATA_CONFIG.save_dir,
        per_device_train_batch_size=DATA_CONFIG.batch_size,
        per_device_eval_batch_size=DATA_CONFIG.batch_size,
        eval_strategy=TRAIN_CONFIG.eval_strategy,
        logging_strategy=TRAIN_CONFIG.logging_strategy,
        num_train_epochs=TRAIN_CONFIG.epochs,
        # fp16=True,
        # save_steps=TRAIN_CONFIG.save_steps,
        eval_steps=TRAIN_CONFIG.eval_steps,
        logging_steps=TRAIN_CONFIG.logging_steps,
        learning_rate=TRAIN_CONFIG.lr,
        remove_unused_columns=TRAIN_CONFIG.remove_unused_columns,
        push_to_hub=TRAIN_CONFIG.push_to_hub,
        load_best_model_at_end=TRAIN_CONFIG.load_best_model_at_end,
        dataloader_num_workers=DATA_CONFIG.num_workers,
        # report_to=TRAIN_CONFIG.report_to,
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
