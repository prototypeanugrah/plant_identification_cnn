from plant_classifier.pipelines.data_ingestion import data_pipeline
from plant_classifier.pipelines.deployment_pipeline import deployment_pipeline
from plant_classifier.pipelines.inference_pipeline import inference_pipeline
from plant_classifier.pipelines.training_pipeline import training_pipeline

__all__ = [
    "data_pipeline",
    "training_pipeline",
    "inference_pipeline",
    "deployment_pipeline",
]
