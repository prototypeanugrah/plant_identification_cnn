import argparse

from plant_classifier.pipelines import (
    deployment_pipeline,
    inference_pipeline,
    training_pipeline,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        required=True,
        choices=[
            "train",
            "inference",
            "deploy",
        ],
        help="The mode to run the pipeline",
    )
    return parser.parse_args()


def main(args):
    if args.mode == "train":
        # Run the training pipeline
        training_pipeline()
    elif args.mode == "inference":
        inference_pipeline()
    elif args.mode == "deploy":
        deployment_pipeline()
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
