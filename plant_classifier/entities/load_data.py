from pathlib import Path
from typing import Dict, Tuple

from datasets import Dataset, load_dataset

from plant_classifier import DATA_CONFIG
from plant_classifier.utils.utils import class_names, visualize_data


def load_data() -> Tuple[
    Dataset,  # train data
    Dataset,  # validation data
    Dataset,  # test data
    Dict[int, str],  # id2label
]:
    """
    Load the data and split it into train, validation and test sets.

    Args:
        None

    Returns:
        Tuple[Dataset, Dataset, Dataset, Dict[int, str]]: The train,
            validation and test sets and the id2label mapping.
    """

    # Load the dataset
    dataset = load_dataset(DATA_CONFIG.dataset_name, split="train")

    # Split the dataset into train, validation and test sets

    # Split the dataset into train and test sets. Train: 80%, Test: 20%
    data_train_test = dataset.train_test_split(
        test_size=0.2, stratify_by_column="label"
    )

    # Split the test set into validation and test sets. Validation: 50%, Test: 50%
    data_train = data_train_test["test"].train_test_split(
        test_size=0.5, stratify_by_column="label"
    )

    # Get the train, validation and test sets
    train_data = data_train_test["train"]
    validation_data = data_train["train"]
    test_data = data_train["test"]

    print(f"Train size: {len(train_data)}")
    print(f"Validation size: {len(validation_data)}")
    print(f"Test size: {len(test_data)}")

    # Get the class names
    id2label = class_names(train_data)
    print(f"Class names: \n{id2label}")

    # Visualize the data
    visualize_data(
        dataset=train_data,
        id2label=id2label,
        save_dir=Path(DATA_CONFIG.save_dir) / "visualization.png",
    )

    return train_data, validation_data, test_data, id2label
