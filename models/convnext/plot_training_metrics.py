import argparse
from pathlib import Path
from .visualization import plot_training_curves

# Generate training plots from metrics.jsonl
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    metrics_path = args.metrics.resolve()
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    output_dir = args.output_dir.resolve() if args.output_dir else metrics_path.parent
    for path in plot_training_curves(metrics_path, output_dir):
        print(path)

if __name__ == "__main__":
    main()
