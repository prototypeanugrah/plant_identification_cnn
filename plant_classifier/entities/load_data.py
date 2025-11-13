from pathlib import Path
from typing import Dict, Tuple

from datasets import Dataset, load_dataset, load_from_disk
from rich import print

from plant_classifier.config import DATA_CONFIG
from plant_classifier.utils.utils import class_names


def load_data() -> Tuple[
    Dataset,  # train data
    Dataset,  # validation data
    Dataset,  # test data
    Dict[int, str],  # id2label
]:
    """
    Load the data and split it into train, validation and test sets.

    If splits already exist on disk, load them for consistency.
    Otherwise, create new splits and save them.

    Args:
        None

    Returns:
        Tuple[Dataset, Dataset, Dataset, Dict[int, str]]: The train,
            validation and test sets and the id2label mapping.
    """

    # Define paths for saved splits
    splits_dir = Path(DATA_CONFIG.save_dir) / "data_splits"
    train_path = splits_dir / "train"
    val_path = splits_dir / "validation"
    test_path = splits_dir / "test"

    # Check if splits already exist
    if train_path.exists() and val_path.exists() and test_path.exists():
        print("[yellow]Loading existing data splits from disk...[/yellow]")
        train_data = load_from_disk(str(train_path))
        validation_data = load_from_disk(str(val_path))
        test_data = load_from_disk(str(test_path))
    else:
        print("[yellow]Creating new data splits...[/yellow]")
        # Load the dataset
        dataset = load_dataset(DATA_CONFIG.dataset_name, split="train")

        # Split the dataset into train, validation and test sets
        # Use fixed seed for reproducibility
        # Final split: Train: 80%, Validation: 10%, Test: 10%

        # Split the dataset into train and test sets. Train: 80%, Test: 20%
        data_train_test = dataset.train_test_split(
            test_size=0.2, stratify_by_column="label", seed=42
        )

        # Split the test set into validation and test sets.
        # Validation: 50%, Test: 50%
        data_train = data_train_test["test"].train_test_split(
            test_size=0.5, stratify_by_column="label", seed=42
        )

        # Get the train, validation and test sets
        train_data = data_train_test["train"]
        validation_data = data_train["train"]
        test_data = data_train["test"]

        # Save splits to disk for future consistency
        print(f"[yellow]Saving data splits to {splits_dir}...[/yellow]")
        splits_dir.mkdir(parents=True, exist_ok=True)
        train_data.save_to_disk(str(train_path))
        validation_data.save_to_disk(str(val_path))
        test_data.save_to_disk(str(test_path))

    print(f"[blue]Train size: {len(train_data)}[/blue]")
    print(f"[blue]Validation size: {len(validation_data)}[/blue]")
    print(f"[blue]Test size: {len(test_data)}[/blue]")

    # Get the class names
    id2label = class_names(train_data)
    print(f"[blue]Class names:[/blue] \n{id2label}")

    return train_data, validation_data, test_data, id2label


if __name__ == "__main__":
    train_data, validation_data, test_data, id2label = load_data()
    print(train_data[0])
