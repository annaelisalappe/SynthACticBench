from __future__ import annotations

import math

import numpy as np
from ConfigSpace import CategoricalHyperparameter, ConfigurationSpace, Float
from numpy import ndarray

from synthacticbench.abstract_function import AbstractFunction
from synthacticbench.base_functions import ZDT1, ZDT3, Ackley, Rosenbrock, SumOfQ


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

        quadr_space.add_configuration_space(
            configuration_space=ConfigurationSpace(
                {
                    f"n_{i}": Float(
                        bounds=(self.instance.lower_bound, self.instance.upper_bound),
                        default=0,
                        name=f"n_{i}",
                    )
                    for i in range(self.num_noisy)
                },
                seed=self.seed,
            ),
            prefix="",
            delimiter="",
        )
        return quadr_space

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

    def _function(self, x: ndarray) -> float:
        quadr_sum = self.instance._function(x[: self.num_quadratic])
        noisy_sum = sum(self.noisy_functions.uniform(size=self.num_noisy))

        return quadr_sum + noisy_sum

    @property
    def x_min(self):
        x_mins_quadr = self.instance.x_min
        x_mins_noisy = np.array((None) * self.num_noisy)

        return np.concatenate((x_mins_quadr, x_mins_noisy), axis=0)

    @property
    def f_min(self) -> float:
        # Noisy params have their min at 0.0
        return self.instance.f_min


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


class InvalidParameterization(AbstractFunction):
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

        self.cube, self.cube_side = self._make_hypercube(cube_size)

    def _make_hypercube(self, cube_size):
        lower_bound = self.instance.lower_bound
        upper_bound = self.instance.upper_bound

        cube_side = (abs(lower_bound) + abs(upper_bound)) * cube_size
        cube = np.array(
            self.rng.uniform(low=lower_bound, high=(upper_bound - cube_side), size=self.dim)
        )
        return cube, cube_side

    def _function(self, x: np.ndarray) -> float:
        invalid = self._check_hypercube(x=x)
        if not invalid:
            return self.instance._function(x)

        err_message = "Received invalid parameter value for "
        for i in invalid:
            err_message += f"x_{i}: {x[i]}, "

        raise Exception(err_message[:-1])

    def _check_hypercube(self, x: np.ndarray):
        invalid = []
        for i in range(len(x)):
            if self.cube[i] <= x[i] < self.cube[i] + self.cube_side:
                invalid.append(i)
        print(invalid)
        return invalid

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
    def f_min(self) -> float:
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

        if distribution == "uniform":
            low = kwargs.get("low", -1)
            high = kwargs.get("high", 1)
            return lambda: self.rng.uniform(low, high)

        if distribution == "exponential":
            lambd = kwargs.get("lambd", 1)
            return lambda: self.rng.exponential(1 / lambd)

        raise ValueError("Unsupported distribution type")

    def _function(self, x: np.ndarray) -> float:
        base_value = self.instance._function(x)
        return base_value + self.noise_generator()

    @property
    def x_min(self) -> np.ndarray | None:
        return self.instance.x_min

    @property
    def f_min(self) -> float:
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


class ActivationStructures(AbstractFunction):
    def __init__(
        self,
        dim: int,
        groups: int,
        loggers: list | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed, dim, loggers)

        self._x_min = None
        self.groups = groups
        self.rng = np.random.default_rng(seed=seed)

        self.instances = self._make_groups()

        self._configspace = self._create_config_space()
        self.benchmark_name = "c4"

    def _create_config_space(self):
        configuration_space = ConfigurationSpace()

        for cat, instance in self.instances.items():
            function = instance["function"]

            configuration_space.add_configuration_space(
                configuration_space=function._create_config_space(),
                prefix=cat,
                delimiter=":",
            )
        configuration_space.add(
            CategoricalHyperparameter(name="c", choices=range(self.groups), default_value=0)
        )
        return configuration_space

    def _make_groups(self):
        instances = {}
        x_i = 0

        min_group_size = self.dim // self.groups
        remainder = self.dim % self.groups
        group_dims = [min_group_size + 1] * remainder + [min_group_size] * (
            self.groups - remainder
        )
        group_seeds = self.rng.integers(low=0, high=1000, size=self.groups)

        for i in range(self.groups):
            instances[i] = {
                "function": SumOfQ(seed=group_seeds[i], dim=group_dims[i]),
                "starts_at": x_i,
            }
            x_i += group_dims[i]

        return instances

    def _function(self, x: np.ndarray) -> float:
        cat = x[-1]
        instance = self.instances[cat]["function"]
        starts_at = self.instances[cat]["starts_at"]
        x = x[starts_at : starts_at + instance.dim]

        return instance._function(x)

    def _calculate_x_min(self):
        current_best_f = math.inf

        for cat, func in self.instances.items():
            if func["function"].f_min < current_best_f:
                x_min = (func["function"].x_min).append(cat)
        self._x_min = x_min

    @property
    def x_min(self) -> np.ndarray | None:
        if self._x_min is not None:
            self._calculate_x_min()

        return self._x_min

    @property
    def f_min(self) -> float:
        return self._function(self.x_min)


class SinglePeak(AbstractFunction):
    def __init__(
        self,
        dim: int,
        peak_width: float,
        loggers: list | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed, dim, loggers)

        self.peak_width = peak_width
        self.rng = np.random.default_rng(seed=seed)
        self.lower_ends, self.abs_peak_width = self._make_peak()

        self._configspace = self._create_config_space()
        self.benchmark_name = "o7"

    def _create_config_space(self):
        return ConfigurationSpace(
            {
                f"x_{i}": Float(
                    bounds=(self.lower_bound, self.upper_bound),
                    default=0,
                    name=f"x_{i}",
                )
                for i in range(self.dim)
            },
            seed=self.seed,
        )

    def _make_peak(self):
        abs_peak_width = (abs(self.lower_bound) + abs(self.upper_bound)) * self.peak_width
        lower_ends = self.rng.uniform(
            low=self.lower_bound,
            high=(self.upper_bound - abs_peak_width),
            size=self.dim,
        )
        return lower_ends, abs_peak_width

    @property
    def lower_bound(self) -> int | float:
        return -100

    @property
    def upper_bound(self) -> int | float:
        return 100

    def _function(self, x: np.ndarray) -> float:
        if np.all((self.lower_ends <= x) and (x < self.lower_ends + self.abs_peak_width)):
            return 0.0

        return 1.0

    @property
    def x_min(self) -> np.ndarray | None:
        # TODO
        pass

    @property
    def f_min(self) -> float:
        return 0.0
