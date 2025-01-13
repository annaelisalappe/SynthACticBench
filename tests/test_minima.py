#!/usr/bin/env python

"""Tests for optbench package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf
from synthacticbench.generate_instances import generate_instances
from synthacticbench.objective_functions import (
    CensoredObjective,
    DeterministicObjective,
    Multimodal,
    MultipleObjectives,
    NoisyEvaluation,
    SinglePeak,
    TimeDependentOP,
)
from synthacticbench.search_space_functions import (
    ActivationStructures,
    HierarchicalStructures,
    InvalidParameterization,
    MixedDomains,
    MixedTypes,
    ParameterInteractions,
    RelevantParameters,
    ShiftingDomains,
)

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


def test_search_space(generated_instances):
    for path in generated_instances:
        cfg = OmegaConf.load(path)
        problem = instantiate(cfg.problem)

        funcclass_instance = problem.function

        # Check if it's an instance of RelevantParameters
        if isinstance(funcclass_instance, RelevantParameters):
            print("Seed:", funcclass_instance.seed)

            func = RelevantParameters(
                num_quadratic=5, dim=funcclass_instance.dim, seed=funcclass_instance.seed
            )
            # Manually set noisy parameters to zero since x_min includes None
            x_min = func.x_min
            y = func._function(x_min)
            noise = 100
            assert np.abs(y - func.f_min) <= noise, (
                f"Failed for {type(funcclass_instance).__name__}: "
                f"f_min={func.f_min}, y={y}, "
                f"error={y - func.f_min} (allowed deviation: {noise})"
            )
        elif isinstance(funcclass_instance, ParameterInteractions):
            # Special case: ParameterInteractions requires a name
            func = ParameterInteractions(
                name=funcclass_instance.name,
                dim=funcclass_instance.dim,
                seed=funcclass_instance.seed,
            )
            x_min = func.x_min
            y = func._function(x_min)
            assert np.isclose(
                func.f_min, y
            ), f"Failed for {type(funcclass_instance).__name__}: f_min={func.f_min}, y={y}"

        elif isinstance(funcclass_instance, MixedTypes):
            # Special case: MixedTypes requires share parameters
            share_cat = 0.3
            share_bool = 0.2
            share_int = 0.3
            share_float = 0.2
            func = MixedTypes(
                dim=funcclass_instance.dim,
                share_cat=share_cat,
                share_bool=share_bool,
                share_int=share_int,
                share_float=share_float,
                seed=funcclass_instance.seed,
            )
            # Since MixedTypes doesn't provide x_min or f_min directly,
            # we test that it runs without errors for a random input
            rng = np.random.default_rng()
            # Generate random values
            x = rng.uniform(low=-100, high=100, size=funcclass_instance.dim)
            y = func._function(x)
            assert isinstance(y, float), f"Failed for {funcclass_instance.__name__}: y={y}"

        elif isinstance(funcclass_instance, ActivationStructures):
            # Special case: ActivationStructures requires groups
            func = ActivationStructures(
                dim=funcclass_instance.dim,
                groups=funcclass_instance.groups,
                seed=funcclass_instance.seed,
            )
            x_min = func.x_min
            y = func._function(x_min)
            assert np.isclose(
                func.f_min, y
            ), f"Failed for {funcclass_instance.__name__}: f_min={func.f_min}, y={y}"

        elif isinstance(funcclass_instance, ShiftingDomains):
            # Special case: ShiftingDomains has shifted domains for x[0] < 0
            func = ShiftingDomains(dim=funcclass_instance.dim, seed=funcclass_instance.seed)
            x_min = func.x_min
            y = func._function(x_min)

            # Ensure that f_min matches the function value at x_min
            assert np.isclose(
                func.f_min, y
            ), f"Failed for {funcclass_instance.__name__}: f_min={func.f_min}, y={y}"

        elif isinstance(funcclass_instance, HierarchicalStructures):
            # Special case: HierarchicalStructures requires groups and subgroups
            func = HierarchicalStructures(
                dim=funcclass_instance.dim,
                groups=funcclass_instance.groups,
                subgroups_per_group=funcclass_instance.subgroups_per_group,
                seed=funcclass_instance.seed,
            )

            # Test that the function calculates a valid minimum
            x_min = func.x_min
            y = func._function(x_min)
            assert np.isclose(
                func.f_min, y
            ), f"Failed for {funcclass_instance.__name__}: f_min={func.f_min}, y={y}"

        elif isinstance(funcclass_instance, InvalidParameterization):
            # Test for InvalidParameterization class
            func = InvalidParameterization(
                dim=funcclass_instance.dim, cube_size=0.05, seed=funcclass_instance.seed
            )

            # Ensure that f(x_min) == f_min
            x_min = func.x_min
            y_min = func._function(x_min)

            # Check that f_min is equal to f(x_min)
            assert np.isclose(
                func.f_min, y_min
            ), f"Failed for {funcclass_instance.__name__}: f_min={func.f_min}, y_min={y_min}"

        elif isinstance(funcclass_instance, MixedDomains):
            # Test for MixedDomains class
            func = MixedDomains(dim=funcclass_instance.dim, seed=funcclass_instance.seed)

            # Ensure that f(x_min) == f_min
            x_min = func.x_min
            y_min = func._function(x_min)

            # Ensure f_min is equal to f(x_min) and handle mixed domains
            assert np.isclose(
                func.f_min, y_min
            ), f"Failed for {funcclass_instance.__name__}: f_min={func.f_min}, y_min={y_min}"


def test_obj_func(generated_instances):
    for path in generated_instances:
        cfg = OmegaConf.load(path)
        problem = instantiate(cfg.problem)

        funcclass_instance = problem.function

        if isinstance(funcclass_instance, DeterministicObjective):
            # Test for DeterministicObjective class
            wrapped_bench = MixedDomains(
                dim=funcclass_instance.dim, seed=funcclass_instance.seed
            )  # Example, adjust as needed
            func = DeterministicObjective(wrapped_bench)

            # Ensure that f(x_min) == f_min
            x_min = func.x_min

            y_min = func._function(x_min)

            # Ensure f_min is equal to f(x_min)
            assert np.isclose(
                func.f_min, y_min
            ), f"Failed for {funcclass_instance.__name__}: f_min={func.f_min}, y_min={y_min}"

        elif isinstance(funcclass_instance, NoisyEvaluation):
            # Test for NoisyEvaluation class
            func = NoisyEvaluation(
                dim=funcclass_instance.dim,
                seed=funcclass_instance.seed,
                distribution="no_noise",
                mean=0,
                stddev=1,
            )

            # Ensure that f(x_min) == f_min
            x_min = func.x_min
            y_min = func._function(x_min)

            # Ensure f_min is equal to f(x_min)
            assert np.isclose(
                func.f_min, y_min
            ), f"Failed for {funcclass_instance.__name__}: f_min={func.f_min}, y_min={y_min}"

        elif isinstance(funcclass_instance, MultipleObjectives):
            for name in ["zdt1", "zdt3"]:  # Example of two well-known benchmark problems
                func = MultipleObjectives(
                    name=name, dim=funcclass_instance.dim, seed=funcclass_instance.seed
                )

                # Ensure that f_min corresponds to the Pareto front
                x_min = func.x_min
                y_min = func._function(x_min)

                assert np.allclose(func.f_min, y_min), (
                    f"Failed for {funcclass_instance.__name__} ({name}): "
                    f"f_min={func.f_min}, y_min={y_min}"
                )

        elif isinstance(funcclass_instance, Multimodal):
            for name in ["ackley", "griewank"]:  # Test multiple multimodal functions
                func = Multimodal(
                    name=name, dim=funcclass_instance.dim, seed=funcclass_instance.seed
                )

                # Ensure that f(x_min) == f_min
                x_min = func.x_min
                y_min = func._function(x_min)

                # Ensure f_min is equal to f(x_min)
                assert np.isclose(func.f_min, y_min), (
                    f"Failed for {funcclass_instance.__name__} ({name}): "
                    f"f_min={func.f_min}, y_min={y_min}"
                )

        elif isinstance(funcclass_instance, CensoredObjective):
            # Special case: CensoredObjective requires a wrapped benchmark
            wrapped_bench = MixedDomains(
                dim=funcclass_instance.dim, seed=funcclass_instance.seed
            )  # Example, adjust as needed
            func = CensoredObjective(
                cutoff=-1e+8, wrapped_bench=wrapped_bench
            )

            # Test the behavior when the function value is below the cutoff
            x_min = func.x_min
            f_min = func.f_min
            y_min = func._function(x_min)

            assert np.isclose(func.f_min, y_min), (
                f"Failed for {type(funcclass_instance).__name__} "
                f"with {type(wrapped_bench).__name__}: "
                f"f_min={func.f_min}, y_min={y_min}"
            )

        elif isinstance(funcclass_instance, SinglePeak):
            # Test for SinglePeak class
            func = SinglePeak(
                dim=funcclass_instance.dim,
                peak_width=funcclass_instance.peak_width,
                seed=funcclass_instance.seed,
                instance_parameter=funcclass_instance.instance_parameter,
            )
            x_min = func.x_min
            f_min = func.f_min
            y_min = func._function(x_min)

            assert np.isclose(
                func.f_min, y_min
            ), f"Failed for {SinglePeak.__name__}: f_min={func.f_min}, y_min={y_min}"

        elif isinstance(funcclass_instance, TimeDependentOP):
            # Test for TimeDependentOP class
            func = TimeDependentOP(
                dim=funcclass_instance.dim,
                name=funcclass_instance.name,
                a=funcclass_instance.a,
                b=funcclass_instance.b,
                seed=funcclass_instance.seed,
                instance_parameter=funcclass_instance.instance_parameter,
            )

            # Test x_min and f_min
            x_min = func.x_min  # Should match the instance's x_min
            f_min = func.f_min  # Should match the instance's f_min
            y_min = func._function(x_min)

            assert np.isclose(
                func.f_min, y_min
            ), f"Failed for {TimeDependentOP.__name__}: f_min={f_min}, y_min={y_min}"
