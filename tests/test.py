#!/usr/bin/env python

"""Tests for `optbench` package."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from carps.utils.trials import TrialInfo
from hydra.utils import instantiate
from omegaconf import OmegaConf
from synthacticbench import base_functions
from synthacticbench.abstract_function import AbstractFunction
from synthacticbench.generate_instances import generate_instances

function_classes = [
    c[1]
    for c in inspect.getmembers(base_functions, inspect.isclass)
    if issubclass(c[1], AbstractFunction) and c[1] != AbstractFunction
]

CONFIG_PATHS = [
    "synthacticbench/configs/problem/SynthACticBench/RelevantParameters.yaml",
    "synthacticbench/configs/problem/SynthACticBench/InvalidParameterization.yaml",
    "synthacticbench/configs/problem/SynthACticBench/NoisyEvaluation.yaml",
    "synthacticbench/configs/problem/SynthACticBench/ParameterInteractions-rosenbrock.yaml",
    "synthacticbench/configs/problem/SynthACticBench/ParameterInteractions-ackley.yaml",
    "synthacticbench/configs/problem/SynthACticBench/ActivationStructures.yaml",
    "synthacticbench/configs/problem/SynthACticBench/SinglePeak.yaml",
    "synthacticbench/configs/problem/SynthACticBench/MixedDomains.yaml",
    "synthacticbench/configs/problem/SynthACticBench/ShiftingDomains.yaml",
    "synthacticbench/configs/problem/SynthACticBench/TimeDependentOP.yaml",
    "synthacticbench/configs/problem/SynthACticBench/TimeDependentNOP.yaml",
]


@pytest.fixture(scope="module", params=CONFIG_PATHS)
def generated_instances(tmp_path_factory, request):
    config_path = Path(request.param)
    output_dir = tmp_path_factory.mktemp(config_path.stem)

    generate_instances(input_path=config_path, output_dir=output_dir, num_instances=3)
    return list(output_dir.glob("*.yaml"))


def test_instantiate_and_evaluate(generated_instances):
    print(f"Generated instances in test: {generated_instances}")
    for path in generated_instances:
        cfg = OmegaConf.load(path)
        problem = instantiate(cfg.problem)
        problem._evaluate(TrialInfo(config=problem.configspace.sample_configuration()))


def test_reproducibility(generated_instances):
    for path in generated_instances:
        cfg = OmegaConf.load(path)
        problem = instantiate(cfg.problem)
        problem_alt = instantiate(cfg.problem)
        config = problem.configspace.sample_configuration()

        assert problem._evaluate(TrialInfo(config=config)) == problem_alt._evaluate(
            TrialInfo(config=config)
        )
