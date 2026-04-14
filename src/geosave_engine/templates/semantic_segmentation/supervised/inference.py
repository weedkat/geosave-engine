import argparse

def infer(config: str):
    """Run inference with the specified configuration."""
    # Placeholder for inference logic
    print(f"Running inference with config: {config}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with the specified configuration.")
    parser.add_argument("--model", required=True, help="Path to the configuration file.")
    args = parser.parse_args()
    infer(args.model)