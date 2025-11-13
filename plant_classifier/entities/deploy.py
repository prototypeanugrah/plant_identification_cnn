from typing import Dict

from mlflow.entities import Run
from rich import print

from plant_classifier.config import DEPLOY_CONFIG


def deployment_trigger(run: Run) -> bool:
    """
    Check if the logged model meets deployment criteria and register it if so.

    This function retrieves the test metrics already logged during training
    and compares them against the deployment thresholds defined in config.

    Args:
        run (Run): The run to check.

    Returns:
        bool: True if model meets deployment criteria, False otherwise
    """
    print("\n" + "-" * 80)
    print("Starting deployment trigger evaluation...")
    print("-" * 80)

    # Get the full run object with all metrics
    metrics: Dict[str, float] = run.data.metrics

    # Check if test metrics are available
    required_metrics = [
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
    ]
    if not all(metric in metrics for metric in required_metrics):
        print("[red]Missing required test metrics. Skipping deployment.[/red]")
        print(f"[yellow]Available metrics: {list(metrics.keys())}[/yellow]")
        return False

    # Get test metrics
    test_accuracy = metrics["test_accuracy"]
    test_precision = metrics["test_precision"]
    test_recall = metrics["test_recall"]
    test_f1 = metrics["test_f1"]

    # Check if all metrics meet thresholds
    passes_accuracy = test_accuracy >= DEPLOY_CONFIG.min_accuracy
    passes_precision = test_precision >= DEPLOY_CONFIG.min_precision
    passes_recall = test_recall >= DEPLOY_CONFIG.min_recall
    passes_f1 = test_f1 >= DEPLOY_CONFIG.min_f1

    all_pass = passes_accuracy and passes_precision and passes_recall and passes_f1

    print("\nDeployment Criteria Check:")
    print(
        f"[blue]Accuracy:[/blue] {'[green]✓ PASS[/green]' if passes_accuracy else '[red]✗ FAIL[/red]'}"
    )
    print(
        f"[blue]Precision:[/blue] {'[green]✓ PASS[/green]' if passes_precision else '[red]✗ FAIL[/red]'}"
    )
    print(
        f"[blue]Recall:[/blue] {'[green]✓ PASS[/green]' if passes_recall else '[red]✗ FAIL[/red]'}"
    )
    print(
        f"[blue]F1 Score:[/blue] {'[green]✓ PASS[/green]' if passes_f1 else '[red]✗ FAIL[/red]'}"
    )
    print("-" * 80)

    if all_pass:
        print("\n[green]✓ All metrics meet deployment thresholds![/green]")
        return True
    else:
        print("\n[red]✗ Model does not meet deployment thresholds.[/red]")
        print("-" * 80)
        return False
