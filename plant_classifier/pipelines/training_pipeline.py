import mlflow

from plant_classifier import DATA_CONFIG, PROCESSOR
from plant_classifier.entities.load_data import load_data
from plant_classifier.entities.model import train_model
from plant_classifier.entities.preproces import preprocess


def training_pipeline():
    """
    Training pipeline.
    """
    train_data, validation_data, test_data, id2label = load_data()

    train_data = train_data.with_transform(preprocess)
    validation_data = validation_data.with_transform(preprocess)
    test_data = test_data.with_transform(preprocess)

    trainer = train_model(
        train_data=train_data,
        validation_data=validation_data,
    )

    # Train the model
    print("Training the model...")
    with mlflow.start_run():
        train_results = trainer.train()
        metrics = trainer.evaluate(test_data)
        model_info = mlflow.transformers.log_model(
            transformers_model={"model": trainer.model, "tokenizer": PROCESSOR},
            task="image-classification",
            artifact_path="image_classifier",
            input_example=train_data[0]["image"],
        )

    # Save the trained model
    print("Saving the trained model...")
    trainer.save_model(DATA_CONFIG.save_dir)
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    # Evaluate the model
    print("Evaluating the model...")
    metrics = trainer.evaluate(test_data)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
