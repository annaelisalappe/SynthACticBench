from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml


def generate_instances(
    input_path: str,
    output_dir: str | None = None,
    seed: int = 100,
    num_instances: int = 100,
    std: float = 1.0,
):
    # Load the original YAML configuration
    with open(input_path) as file:
        config = yaml.safe_load(file)

    # Ensure the output directory exists
    output_dir = Path(input_path).with_suffix("") if not output_dir else Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample values for the instance_parameter
    mean = 0
    instance_generator = np.random.default_rng(seed=seed)
    sampled_values = instance_generator.normal(loc=mean, scale=std, size=num_instances)

    # Generate new YAML files
    for i, value in enumerate(sampled_values, start=1):
        config["instance_parameter"] = float(value)
        output_file = output_dir / f"instance_{i}.yaml"

        # Save the new configuration
        with open(output_file, "w") as file:
            yaml.dump(config, file, default_flow_style=False)

        print(f"Generated: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate YAML files with sampled instance parameters."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the original YAML file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save generated YAML files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed with which the instance parameters are sampled.",
    )
    parser.add_argument(
        "--instances",
        type=int,
        help="Number of instances to generate.",
    )
    parser.add_argument(
        "--std",
        type=float,
        default=1.0,
        help="Standard deviation of the normal distribution (default: 1.0).",
    )

    args = parser.parse_args()
    generate_instances(
        input_path=args.input_path,
        output_dir=args.output_dir,
        seed=args.seed,
        num_instances=args.instances,
        std=args.std,
    )
