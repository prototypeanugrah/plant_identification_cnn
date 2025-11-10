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
