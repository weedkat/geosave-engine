import argparse


def train(config: str):
    """Train a model using the specified configuration."""
    # Placeholder for training logic
    print(f"Training model with config: {config}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training with the specified configuration.")
    parser.add_argument("--config", required=True, help="Path to the configuration file.")
    args = parser.parse_args()
    train(args.config)