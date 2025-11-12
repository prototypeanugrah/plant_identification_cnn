import argparse

from plant_classifier.pipelines import inference_pipeline, training_pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    return parser.parse_args()


def main(args):
    if args.mode == "train":
        # Run the training pipeline
        training_pipeline()
    elif args.mode == "inference":
        inference_pipeline()
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
