#!/usr/bin/env python

"""Tests for `optbench` package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from carps.utils.trials import TrialInfo
from hydra.utils import instantiate
from omegaconf import OmegaConf
from synthacticbench.generate_instances import generate_instances

CONFIG_PATHS = [
    "synthacticbench/configs/problem/SynthACticBench/C1-RelevantParameters.yaml",
    "synthacticbench/configs/problem/SynthACticBench/C2-ParameterInteractions-rosenbrock.yaml",
    "synthacticbench/configs/problem/SynthACticBench/C2-ParameterInteractions-ackley.yaml",
    "synthacticbench/configs/problem/SynthACticBench/C3-MixedTypes.yaml",
    "synthacticbench/configs/problem/SynthACticBench/C4-ActivationStructures.yaml",
    "synthacticbench/configs/problem/SynthACticBench/C5-ShiftingDomains.yaml",
    "synthacticbench/configs/problem/SynthACticBench/C6-HierarchicalStructures.yaml",
    "synthacticbench/configs/problem/SynthACticBench/C7-InvalidParameterization.yaml",
    "synthacticbench/configs/problem/SynthACticBench/C8-MixedDomains.yaml",
    #
    "synthacticbench/configs/problem/SynthACticBench/O1-DeterministicObjective.yaml",
    "synthacticbench/configs/problem/SynthACticBench/O2-NoisyEvaluation.yaml",
    "synthacticbench/configs/problem/SynthACticBench/O3-MultipleObjectives.yaml",
    "synthacticbench/configs/problem/SynthACticBench/O4-TimeDependentOP.yaml",
    "synthacticbench/configs/problem/SynthACticBench/O4-TimeDependentNOP.yaml",
    "synthacticbench/configs/problem/SynthACticBench/O5-CensoredObjective.yaml",
    "synthacticbench/configs/problem/SynthACticBench/O6-Multimodal.yaml",
    "synthacticbench/configs/problem/SynthACticBench/O7-SinglePeak.yaml",
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

        result = problem._evaluate(TrialInfo(config=config))
        result_alt = problem_alt._evaluate(TrialInfo(config=config))

        # Convert results to lists of their attributes
        result_list = [
            result.cost.tolist() if isinstance(result.cost, np.ndarray) else result.cost,
            result.time,
            result.virtual_time,
            result.status,
            result.starttime,
            result.endtime,
            result.additional_info,
        ]
        result_alt_list = [
            result_alt.cost.tolist()
            if isinstance(result_alt.cost, np.ndarray)
            else result_alt.cost,
            result_alt.time,
            result_alt.virtual_time,
            result_alt.status,
            result_alt.starttime,
            result_alt.endtime,
            result_alt.additional_info,
        ]

        # Compare each attribute
        for i, (attr, attr_alt) in enumerate(zip(result_list, result_alt_list, strict=True)):
            assert attr == attr_alt, f"Mismatch at index {i}: {attr} != {attr_alt}"
