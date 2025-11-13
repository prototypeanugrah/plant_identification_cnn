import os

import mlflow
from dotenv import load_dotenv
from rich import print

from plant_classifier.config import TRAIN_CONFIG

load_dotenv()

DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_USER = os.environ.get("MLFLOW_USER")
MLFLOW_REGISTRY_URI = os.environ.get("MLFLOW_REGISTRY_URI")


def delete_experiment(experiment_name: str | None, experiment_id: str | None):
    if experiment_name:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(
                f"[red]Experiment {experiment_name} does not exist; nothing to delete[/red]"
            )
            return
        experiment_id = experiment.experiment_id
    elif experiment_id:
        experiment = mlflow.get_experiment(experiment_id)
        if experiment is None:
            print(
                f"[red]Experiment with ID {experiment_id} does not exist; nothing to delete[/red]"
            )
            return
        experiment_name = experiment.name
    else:
        raise ValueError("Either experiment_name or experiment_id must be provided")

    if experiment_name:
        if mlflow.get_experiment_by_name(experiment_name):
            # delete the experiment
            mlflow.delete_experiment(experiment_id)
            print(
                f"[green]Experiment {experiment_name} with ID {experiment_id} deleted[/green]"
            )
    elif experiment_id:
        if mlflow.get_experiment(experiment_id):
            # delete the experiment
            mlflow.delete_experiment(experiment_id)
            print(
                f"[green]Experiment {experiment_name} with ID {experiment_id} deleted[/green]"
            )


def setup_mlflow_databricks():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if MLFLOW_REGISTRY_URI is not None:
        mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)
    else:
        print("[red]MLflow registry URI not found![/red]")
        mlflow.set_registry_uri("databricks-uc")

    experiment_name = f"/Users/{MLFLOW_USER}/{TRAIN_CONFIG.mlflow_experiment_name}"

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(
            f"[green]Experiment {experiment_name} created with ID: {experiment_id}[/green]"
        )
    else:
        experiment_id = experiment.experiment_id
        print(
            f"[yellow]Experiment {experiment_name} already exists with ID: {experiment_id}[/yellow]"
        )

    mlflow.set_experiment(experiment_name)
    print("=" * 100)


def setup_mlflow_local():
    mlflow.set_tracking_uri(TRAIN_CONFIG.mlflow_tracking_uri)
    mlflow.set_experiment(TRAIN_CONFIG.mlflow_experiment_name)
    print("=" * 100)


def setup_mlflow():
    if DATABRICKS_TOKEN or DATABRICKS_HOST:
        if DATABRICKS_TOKEN is None:
            print("[blue]Databricks token not found! Please enter the token: [/blue]")
            databricks_token = input("Please enter the token: ")
            os.environ["DATABRICKS_TOKEN"] = databricks_token
        if DATABRICKS_HOST is None:
            print("[blue]Databricks host not found! Please enter the host: [/blue]")
            databricks_host = input("Please enter the host: ")
            os.environ["DATABRICKS_HOST"] = databricks_host
        setup_mlflow_databricks()
    else:
        print("[red]Databricks credentials not found! Using local MLflow...[/red]")
        setup_mlflow_local()


if __name__ == "__main__":
    setup_mlflow()
