from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    dataset_name: str = Field(..., description="The name of the dataset to load")
    model_path: str = Field(..., description="The path to the model to use")
    batch_size: int = Field(..., description="The batch size to use")
    num_workers: int = Field(..., description="The number of workers to use")
    save_dir: str = Field(..., description="The directory to save the artifacts")


class TrainConfig(BaseModel):
    model_path: str = Field(..., description="The path to the model to use")
    epochs: int = Field(..., description="The number of epochs to train for")
    lr: float = Field(..., description="The learning rate to use")
    device: str = Field(..., description="The device to use")
    run_name: str = Field(..., description="The name of the run")
    eval_strategy: str = Field(..., description="The evaluation strategy to use")
    logging_strategy: str = Field(..., description="The logging strategy to use")
    save_steps: int = Field(..., description="The number of steps to save the model")
    eval_steps: int = Field(
        ..., description="The number of steps to evaluate the model"
    )
    logging_steps: int = Field(
        ..., description="The number of steps to log the metrics"
    )
    remove_unused_columns: bool = Field(
        ..., description="Whether to remove unused columns"
    )
    push_to_hub: bool = Field(..., description="Whether to push the model to the hub")
    load_best_model_at_end: bool = Field(
        ..., description="Whether to load the best model at the end"
    )
    report_to: str = Field(..., description="The report to use")
    mlflow_tracking_uri: str = Field(..., description="The tracking URI to use")
    mlflow_experiment_name: str = Field(..., description="The experiment name to use")
    mlflow_registered_name: str = Field(..., description="The registered name to use")
    task: str = Field(..., description="The task to use")
    name: str = Field(..., description="The name/artifact path to use")


class DeployConfig(BaseModel):
    min_precision: float = Field(..., description="The minimum precision to use")
    min_recall: float = Field(..., description="The minimum recall to use")
    min_f1: float = Field(..., description="The minimum F1 score to use")
    min_accuracy: float = Field(..., description="The minimum accuracy to use")
