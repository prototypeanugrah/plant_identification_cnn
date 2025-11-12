import mlflow

from plant_classifier import PROCESSOR, TRAIN_CONFIG
from plant_classifier.entities.load_data import load_data
from plant_classifier.entities.model import train_model
from plant_classifier.entities.preproces import preprocess
from plant_classifier.utils.utils import create_signature


def training_pipeline():
    """
    Training pipeline.
    """
    train_data, validation_data, test_data, _ = load_data()

    # Keep original dataset for signature generation (needs original images, not preprocessed)
    sample_train_data = train_data[0]["image"]

    train_data = train_data.with_transform(preprocess)
    validation_data = validation_data.with_transform(preprocess)
    test_data = test_data.with_transform(preprocess)

    trainer = train_model(
        run_name=TRAIN_CONFIG.run_name,
        train_data=train_data,
        validation_data=validation_data,
    )

    # Train the model
    print("Training started...")
    with mlflow.start_run() as run:
        trainer.train()

    # Log the model with signature
    with mlflow.start_run(run_id=run.info.run_id):
        image_classifier_pipeline, signature = create_signature(
            sample_image=sample_train_data,
            trainer=trainer,
            image_processor=PROCESSOR,
        )

        # Log model with signature
        mlflow.transformers.log_model(
            transformers_model=image_classifier_pipeline,
            task="image-classification",
            name="image_classifier",
            signature=signature,
            registered_model_name="PlantClassifierHfTraining",
        )

        # Evaluate the trained model
        print("Evaluating the model...")
        mlflow.log_metrics(
            trainer.evaluate(
                eval_dataset=test_data,
                metric_key_prefix="test",
            ),
            # run_id=run.info.run_id,
        )
    print("Training completed...")
