import mlflow
from rich import print

from plant_classifier.config import TRAIN_CONFIG
from plant_classifier.entities.deploy import deployment_trigger
from plant_classifier.mlflow_databricks_integration import setup_mlflow


def deployment_pipeline(run_name: str | None = None):
    """
    Deployment pipeline.

    Args:
        run_name (str | None): The name of the run to deploy. If None, the run name
            from the training configuration will be used.
    """

    print("[purple]=" * 80)
    print(f"[purple]Deployment pipeline started with run name: {run_name}[/purple]")
    print("[purple]=" * 80)

    # Initialize MLFlow
    setup_mlflow()

    # 1. Find the training run
    run_name = run_name or TRAIN_CONFIG.run_name

    conditions = f"""
        tags.mlflow.runName LIKE '%{run_name}%'
        AND status = 'FINISHED'
    """

    runs = mlflow.search_runs(
        filter_string=conditions,
        search_all_experiments=True,
        order_by=["start_time DESC"],
        max_results=1,
    )

    if runs.empty:
        raise ValueError(f"No training run found with name: {run_name}")

    run_id = runs.iloc[0].run_id
    run = mlflow.get_run(run_id)

    # 2. Check deployment criteria
    should_deploy = deployment_trigger(run)

    # 3. Deploy if criteria met
    if should_deploy:
        model_uri = f"runs:/{run_id}/{TRAIN_CONFIG.name}"
        mlflow.register_model(
            model_uri=model_uri, name=TRAIN_CONFIG.mlflow_registered_name
        )
        print(f"[green]Model from run {run_id} registered for deployment[/green]")
    else:
        print(f"[red]Model from run {run_id} did not meet deployment criteria[/red]")

    print("[purple]=" * 80)
    print(f"[purple]Deployment pipeline completed with run name: {run_name}[/purple]")
    print("[purple]=" * 80)
