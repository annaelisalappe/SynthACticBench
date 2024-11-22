#!/usr/bin/env python

"""Tests for `optbench` package."""

import inspect

import pytest
from carps.utils.trials import TrialInfo
from hydra.utils import instantiate
from omegaconf import OmegaConf

from synthacticbench import base_functions
from synthacticbench.abstract_function import AbstractFunction

function_classes = [
    c[1]
    for c in inspect.getmembers(base_functions, inspect.isclass)
    if issubclass(c[1], AbstractFunction) and c[1] != AbstractFunction
]

# @pytest.mark.parametrize("funcclass", function_classes)
# def test_opt(funcclass):
#     func = funcclass("Rosenbrock", 10, 0.2, 52)    # Calling x_min on Noisy Function will return None, so setting noisy_params to zero
#     y = func._function(func.x_min)
#     if funcclass == RelevantParameters:
#         assert np.isclose(func.f_min, y)


@pytest.mark.parametrize(
    "path",
    [
        "synthacticbench/configs/problem/SynthACticBench/RelevantParameters.yaml",
        "synthacticbench/configs/problem/SynthACticBench/InvalidParametrisation.yaml",
        "synthacticbench/configs/problem/SynthACticBench/NoisyEvaluation.yaml",
        "synthacticbench/configs/problem/SynthACticBench/ParameterInteractions-rosenbrock.yaml",
        "synthacticbench/configs/problem/SynthACticBench/ParameterInteractions-ackley.yaml",
        "synthacticbench/configs/problem/SynthACticBench/ActivationStructures.yaml",
    ],
)
def test_instantiate_and_evaluate(path):
    cfg = OmegaConf.load(path)
    problem = instantiate(cfg.problem)
    problem._evaluate(TrialInfo(config=problem.configspace.sample_configuration()))


@pytest.mark.parametrize(
    "path",
    [
        "synthacticbench/configs/problem/SynthACticBench/RelevantParameters.yaml",
        "synthacticbench/configs/problem/SynthACticBench/InvalidParametrisation.yaml",
        "synthacticbench/configs/problem/SynthACticBench/NoisyEvaluation.yaml",
        "synthacticbench/configs/problem/SynthACticBench/ParameterInteractions-rosenbrock.yaml",
        "synthacticbench/configs/problem/SynthACticBench/ParameterInteractions-ackley.yaml",
        "synthacticbench/configs/problem/SynthACticBench/ActivationStructures.yaml",
    ],
)
def test_reproducibility(path):
    cfg = OmegaConf.load(path)
    problem = instantiate(cfg.problem)
    problem_alt = instantiate(cfg.problem)
    config = problem.configspace.sample_configuration()

    assert problem._evaluate(TrialInfo(config=config)) == problem_alt._evaluate(
        TrialInfo(config=config)
    )
