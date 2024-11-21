from __future__ import annotations

from math import floor

import numpy as np
from carps.loggers.abstract_logger import AbstractLogger
from ConfigSpace import ConfigurationSpace, Float
from numpy import ndarray

from synthacticbench.abstract_function import AbstractFunction
from synthacticbench.base_functions import Ackley, Rosenbrock, SumOfQ


class RelevantParameters(AbstractFunction):
    def __init__(
        self,
        num_quadratic: int,
        dim: int,
        seed: int | None = None,
        loggers: list | None = None,
    ) -> None:
        super().__init__(seed, dim, loggers)

        self.num_quadratic = num_quadratic
        self.num_noisy = self.dim - self.num_quadratic

        self.rng = np.random.default_rng(seed=seed)

        self.instance = SumOfQ(seed=seed, dim=self.num_quadratic)
        self.noisy_functions = self.rng.spawn(n_children=1)[0]

        self._configspace = self._create_config_space()

        self.benchmark_name = "c1"

    def _create_config_space(self):
        quadr_space = self.instance._create_config_space()
        noisy_space = ConfigurationSpace(
            {
                f"n_{i}": Float(
                    bounds=(self.instance.lower_bound, self.instance.upper_bound),
                    default=0,
                    name=f"n_{i}",
                )
                for i in range(self.num_noisy)
            },
            seed=self.seed,
        )
        return quadr_space.add_configuration_space(
            prefix="",
            configuration_space=noisy_space,
            delimiter="",
        )

    # def _create_config_space(self):
    #     domain_sizes = [1, 100, 10000] * ceil(self.total_params / 3)
    #     lower_bounds = self.rng.integers(low=-10000, high=9000, size=self.total_params)

    #     parameters = []
    #     for i in range(self.total_params):
    #         bounds = (lower_bounds[i], lower_bounds[i] + domain_sizes[i])

    #         if i < self.relevant_params:
    #             name = f"quadratic_parameter_{i}"
    #         else:
    #             name = f"noisy_parameter_{i}"

    #         param = Float(
    #             name,
    #             bounds,  # TODO: Does there have to be a default value?
    #         )
    #         parameters.append(param)

    #     return ConfigurationSpace(name="RPspace", space=parameters)

    def _create_noisy_functions(self):
        self.noise_generators = self.rng.spawn(self.num_noisy)

    def _function(self, x: ndarray) -> ndarray:
        quadr_sum = self.instance._function(x[: self.num_quadratic])
        noisy_sum = sum(self.noisy_functions.uniform(size=self.num_noisy))

        return np.array(quadr_sum + noisy_sum)

    @property
    def x_min(self):
        x_mins_quadr = self.instance.x_min
        x_mins_noisy = np.array((None) * self.num_noisy)

        return np.concatenate((x_mins_quadr, x_mins_noisy), axis=0)

    @property
    def f_min(self):
        # Noisy params have their min at 0.0
        return self.instance.x_min


class ParameterInteractions(AbstractFunction):
    def __init__(
        self, name: str, dim: int, seed: int | None = None, loggers: list | None = None
    ) -> None:
        super().__init__(seed, dim, loggers)
        self.name = name
        self.dim = dim
        self.rng = np.random.default_rng(seed=seed)

        if self.name.lower() == "rosenbrock":
            self.instance = Rosenbrock(dim, seed)
        elif self.name.lower() == "ackley":
            self.instance = Ackley(dim, seed)

        self._configspace = self.instance._create_config_space()
        self.benchmark_name = "c2"

    def _function(self, x: np.ndarray) -> np.ndarray:
        return self.instance._function(x)

    @property
    def x_min(self) -> np.ndarray | None:
        return self.instance.x_min

    @property
    def f_min(self):
        return self.instance.f_min


class InvalidParametrisation(AbstractFunction):
    def __init__(
        self,
        dim: int,
        cube_size: float,
        seed: int | None = None,
        loggers: list | None = None,
    ) -> None:
        super().__init__(seed, dim, loggers)
        self.benchmark_name = "c7"

        self.rng = np.random.default_rng(seed=seed)

        self.instance = SumOfQ(seed=seed, dim=dim)

        self._configspace = self.instance._create_config_space()

        self.cube, self.cube_dim = self._make_hypercube(cube_size)

    def _make_hypercube(self, cube_size):
        lower_bound = self.instance.lower_bound
        upper_bound = self.instance.upper_bound

        cube_dim = (abs(lower_bound) + abs(upper_bound)) * cube_size
        cube = np.array(
            self.rng.integers(
                low=lower_bound, high=floor(upper_bound - cube_dim), size=self.dim
            )
        )
        return cube, cube_dim

    def _function(self, x: np.ndarray) -> np.ndarray:
        invalid = self._check_hypercube(x=x)
        if invalid:
            err_message = f"Received invalid parameter value for"
            for i in invalid:
                err_message.append(f"x_{i}: {x[i]} ")
            raise Exception(err_message)
        else:
            return self.instance._function(x)

    def _check_hypercube(self, x: np.ndarray):
        invalid = []
        for i in range(len(x)):
            if self.cube[i] <= x[i] < self.cube[i] + self.cube_dim:
                invalid.append(i)

    # TODO: Is this okay?
    @property
    def x_min(self) -> np.ndarray | None:
        x_min = self.instance.x_min
        invalid = self._check_hypercube(x=x_min)
        for i in invalid:
            x_min[i] = Exception
        return x_min

    # TODO: Is this okay?
    @property
    def f_min(self):
        return self.instance.f_min


class NoisyEvaluation(AbstractFunction):
    def __init__(
        self,
        dim: int,
        loggers: list | None = None,
        seed: int | None = None,
        distribution: str = "uniform",
        **kwargs,
    ) -> None:
        super().__init__(seed, dim, loggers)
        self.dim = dim
        self.rng = np.random.default_rng(seed=seed)

        self.instance = SumOfQ(seed=seed, dim=dim)

        self._configspace = self.instance._create_config_space()
        self.benchmark_name = "o2"

        self.noise_generator = self._create_noise_generator(distribution, **kwargs)

    def _create_noise_generator(self, distribution: str, **kwargs):
        """Initializes the noise generator function based on the specified distribution."""
        if distribution == "normal":
            mean = kwargs.get("mean", 0)
            stddev = kwargs.get("stddev", 1)
            return lambda: self.rng.normal(mean, stddev)

        elif distribution == "uniform":
            low = kwargs.get("low", -1)
            high = kwargs.get("high", 1)
            return lambda: self.rng.uniform(low, high)

        elif distribution == "exponential":
            lambd = kwargs.get("lambd", 1)
            return lambda: self.rng.exponential(1 / lambd)
        else:
            raise ValueError("Unsupported distribution type")

    def _function(self, x: np.ndarray) -> np.ndarray:
        base_value = self.instance._function(x)
        return base_value + self.noise_generator()

    @property
    def x_min(self) -> np.ndarray | None:
        return self.instance.x_min

    @property
    def f_min(self):
        return self.instance.f_min


class MultipleObjectives(AbstractFunction):
    def __init__(
        self, name: str, dim: int, seed: int | None = None, loggers: list | None = None
    ) -> None:
        super().__init__(seed, dim, loggers)
        self.name = name
        self.rng = np.random.default_rng(seed=seed)

        if self.name.lower() == "zdt1":
            self.instance = ZDT1(dim, seed)
        elif self.name.lower() == "zdt3":
            self.instance = ZDT3(dim, seed)

        self._configspace = self.instance._create_config_space()

        self.benchmark_name = "o3"

    def _create_config_space(self):
        """General method to create a configuration space."""
        lower_bounds = [0] * self.dim
        upper_bounds = [1] * self.dim
        return ConfigurationSpace(
            {
                f"x_{i}": Float(
                    bounds=(lower_bounds[i], upper_bounds[i]),
                    default=0.5,
                    name=f"x_{i}",
                )
                for i in range(self.dim)
            },
            seed=self.seed,
        )

    def _function(self, x: np.ndarray) -> np.ndarray:
        return self.instance._function(x)

    @property
    def x_min(self) -> np.ndarray | None:
        return self.instance.x_min

    @property
    def f_min(self):
        return self.instance.f_min
