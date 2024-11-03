#!/usr/bin/env python

"""Tests for `optbench` package."""

import inspect

import pytest
import numpy as np

from synthacticbench import functions
from synthacticbench.abstract_function import AbstractFunction
from synthacticbench.functions import RelevantParameters

from omegaconf import OmegaConf
from hydra.utils import instantiate
from carps.utils.trials import TrialInfo

function_classes = [c[1] for c in inspect.getmembers(functions, inspect.isclass) if issubclass(c[1], AbstractFunction) and c[1] != AbstractFunction]

@pytest.mark.parametrize("funcclass", function_classes)
def test_opt(funcclass):
    func = funcclass(relevant_params=2, noisy_params=0, seed=52)    # Calling x_min on Noisy Function will return None, so setting noisy_params to zero
    y = func._function(func.x_min)
    if funcclass == RelevantParameters:
        assert np.isclose(func.f_min, y)

@pytest.mark.parametrize("path", ["synthacticbench/configs/problem/SynthACticBench/RelevantParameters_1.yaml"])
def test_instantiate_and_evaluate(path):
    cfg = OmegaConf.load(path)
    problem = instantiate(cfg.problem)
    problem._evaluate(TrialInfo(config=problem.configspace.sample_configuration()))
