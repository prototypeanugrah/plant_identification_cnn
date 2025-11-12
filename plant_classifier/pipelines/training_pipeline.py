import mlflow

from plant_classifier.config import TRAIN_CONFIG
from plant_classifier.entities.load_data import load_data
from plant_classifier.entities.model import train_model
from plant_classifier.entities.preproces import preprocess
from plant_classifier.resources import PROCESSOR
from plant_classifier.utils.utils import create_signature
from plant_classifier.utils.visualizations import (
    analyze_image_dimensions,
    dist_labels,
    visualize_data,
    visualize_test_predictions,
)


def training_pipeline():
    """
    Training pipeline.
    """
    # Initialize MLFlow
    mlflow.set_tracking_uri(TRAIN_CONFIG.mlflow_tracking_uri)
    mlflow.set_experiment(TRAIN_CONFIG.mlflow_experiment_name)

    train_data, validation_data, test_data, id2label = load_data()

    # Create the plots for the data
    train_dims_plot = analyze_image_dimensions(train_data, "train")
    val_dims_plot = analyze_image_dimensions(validation_data, "validation")
    test_dims_plot = analyze_image_dimensions(test_data, "test")
    train_labels_plot = dist_labels(train_data, "train")
    val_labels_plot = dist_labels(validation_data, "validation")
    test_labels_plot = dist_labels(test_data, "test")
    train_viz_plot = visualize_data(train_data, id2label)

    # Keep original dataset for signature generation (needs original images, not preprocessed)
    sample_train_data = train_data[0]["image"]

    transformed_train_data = train_data.with_transform(preprocess)
    transformed_val_data = validation_data.with_transform(preprocess)
    transformed_test_data = test_data.with_transform(preprocess)

    trainer = train_model(
        run_name=TRAIN_CONFIG.run_name,
        train_data=transformed_train_data,
        validation_data=transformed_val_data,
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
                eval_dataset=transformed_test_data,
                metric_key_prefix="test",
            ),
        )

        # Create test predictions visualization
        print("Generating test predictions visualization...")
        test_predictions_plot = visualize_test_predictions(
            dataset=test_data,
            id2label=id2label,
        )

        # Log the EDA figures to MLFlow

        # Log the image dimension analysis figures
        mlflow.log_figure(
            figure=train_dims_plot,
            artifact_file="analyze_image_dimensions_train.png",
        )
        mlflow.log_figure(
            figure=val_dims_plot,
            artifact_file="analyze_image_dimensions_validation.png",
        )
        mlflow.log_figure(
            figure=test_dims_plot,
            artifact_file="analyze_image_dimensions_test.png",
        )

        # Log the label distribution figures
        mlflow.log_figure(
            figure=train_labels_plot,
            artifact_file="dist_labels_train.png",
        )
        mlflow.log_figure(
            figure=val_labels_plot,
            artifact_file="dist_labels_validation.png",
        )
        mlflow.log_figure(
            figure=test_labels_plot,
            artifact_file="dist_labels_test.png",
        )

        # Log the data visualization figures
        mlflow.log_figure(
            figure=train_viz_plot,
            artifact_file="visualize_train_data.png",
        )
        mlflow.log_figure(
            figure=test_predictions_plot,
            artifact_file="visualize_test_predictions.png",
        )

    print("Training completed...")
