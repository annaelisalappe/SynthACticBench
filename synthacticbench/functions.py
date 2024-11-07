from __future__ import annotations

import numpy as np
from ConfigSpace import ConfigurationSpace, Float
from numpy import ndarray
from math import ceil
from synthacticbench.utils.math import QuadraticFunction, NoisyFunction


from carps.loggers.abstract_logger import AbstractLogger

from synthacticbench.abstract_function import AbstractFunction

class RelevantParameters(AbstractFunction):
    def __init__(
            self,
            relevant_params: int,
            noisy_params: int,
            dim: int,
            seed: int | None = None,
            loggers: list | None = None
        ) -> None:
        super().__init__(
            seed,
            dim,
            loggers
        )
        self.relevant_params = relevant_params
        self.noisy_params = noisy_params
        self.dim = dim
        assert self.dim == self.relevant_params + self.noisy_params, "Dimension should be equal to num relevant params + num noisy params."
        self.total_params = relevant_params + noisy_params
        self.rng = np.random.default_rng(seed=seed)

        self._configspace = self._create_config_space()
        self.functions = self._create_functions()

        self.benchmark_name = "c1"

    def _create_config_space(self):
        domain_sizes = [1, 100, 10000] * ceil(self.total_params / 3)
        lower_bounds = self.rng.integers(low=-10000, high=9000, size=self.total_params)

        parameters = []
        for i in range(self.total_params):
            bounds = (lower_bounds[i], lower_bounds[i] + domain_sizes[i])

            if i < self.relevant_params:
                name = f"quadratic_parameter_{i}"
            else:
                name = f"noisy_parameter_{i}"

            param = Float(
                name,
                bounds          # TODO: Does there have to be a default value?
            )
            parameters.append(param)

        return ConfigurationSpace(
            name="RPspace",
            space=parameters
        )

    def _create_functions(self):
        total_params = self.relevant_params + self.noisy_params
        child_generators = self.rng.spawn(total_params)
        functions = []
        i = 0
        for name, param in self._configspace.items():
            gen = child_generators[i]
            if name.startswith("quadratic"):
                functions.append((QuadraticFunction(param, gen), gen))
            elif name.startswith("noisy"):
                functions.append((NoisyFunction(param, gen), gen))
            i+=1

        return functions

    def _function(self, x: ndarray) -> ndarray:
        sum = 0
        i = 0
        for func, _ in self.functions:
            sum += func.evaluate(x[i])
            i+=1

        return np.array(([sum]))

    @property
    def x_min(self):
        x_mins = []
        for func, _ in self.functions:
            x_mins.append(func.x_min)
        return np.array(x_mins)

    @property
    def f_min(self):
        f_min = 0
        for func, _ in self.functions:
            f_min += func.f_min
        return f_min


class ParameterInteractions(AbstractFunction):
    def __init__(
        self,
        name: str,
        dim: int,
        seed: int | None = None,
        loggers: list | None = None
    ) -> None:
        super().__init__(
            seed,
            dim,
            loggers
        )
        self.name = name
        self.dim = dim
        self.rng = np.random.default_rng(seed=seed)

        if self.name == "rosenbrock":
            self.instance = Rosenbrock(dim,seed)
        elif self.name == "ackley":
            self.instance = Ackley(dim,seed)

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


class Rosenbrock(AbstractFunction):

    def __init__(self, dim: int, seed: int | None = None, loggers: list[AbstractLogger] | None = None) -> None:
        super().__init__(
            seed,
            dim,
            loggers
        )

    def _create_config_space(self):
        lower_bounds = [-5] * self.dim
        upper_bounds = [10] * self.dim
        return ConfigurationSpace(
            {f"x_{i}": Float(bounds=(lower_bounds[i], upper_bounds[i]), default=0, name=f"x_{i}") for i in
             range(self.dim)},
            seed=self.seed
        )

    @property
    def x_min(self) -> np.ndarray | None:
        return np.array([1.0] * self.dim)

    @property
    def f_min(self) -> float | None:
        return 0.0

    def _function(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x) if self.dim > 1 else x
        totals = np.zeros(x.shape[0])
        if self.dim == 1:
            totals = (1 - x) ** 2  # Simplified 1D Rosenbrock function
        else:
            for i in range(self.dim - 1):
                totals += 100 * (x[:, i + 1] - x[:, i] ** 2) ** 2 + (1 - x[:, i]) ** 2
        return totals

class Ackley(AbstractFunction):

    def __init__(self, dim: int, seed: int | None = None, loggers: list[AbstractLogger] | None = None) -> None:
        super().__init__(
            seed,
            dim,
            loggers
        )

    def _create_config_space(self):
        lower_bounds = [-32.768] * self.dim
        upper_bounds = [32.768] * self.dim
        return ConfigurationSpace(
            {f"x_{i}": Float(bounds=(lower_bounds[i], upper_bounds[i]), default=0, name=f"x_{i}") for i in
             range(self.dim)},
            seed=self.seed
        )

    @property
    def x_min(self) -> np.ndarray | None:
        return np.zeros(self.dim)

    @property
    def f_min(self) -> float | None:
        return 0.0

    def _function(self, x: np.ndarray) -> np.ndarray:
        if self.dim == 1:
            x = np.expand_dims(x, axis=1)
        part1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2, axis=1) / self.dim))
        part2 = -np.exp(np.sum(np.cos(2 * np.pi * x), axis=1) / self.dim)

        return part1 + part2 + 20 + np.e



class MultipleObjectives(AbstractFunction):
    def __init__(
        self,
        name: str,
        dim: int,
        seed: int | None = None,
        loggers: list | None = None
    ) -> None:
        super().__init__(
            seed,
            dim,
            loggers
        )
        self.name = name
        self.rng = np.random.default_rng(seed=seed)

        if self.name == "zdt1":
            self.instance = ZDT1(dim,seed)
        elif self.name == "zdt3":
            self.instance = ZDT3(dim,seed)

        self._configspace = self.instance._create_config_space()

        self.benchmark_name = "o3"

    def _create_config_space(self):
        """General method to create a configuration space."""
        lower_bounds = [0] * self.dim
        upper_bounds = [1] * self.dim
        return ConfigurationSpace(
            {f"x_{i}": Float(bounds=(lower_bounds[i], upper_bounds[i]), default=0.5, name=f"x_{i}") for i in range(self.dim)},
            seed=self.seed
        )

    def _function(self, x: np.ndarray) -> np.ndarray:
        return self.instance._function(x)

    @property
    def x_min(self) -> np.ndarray | None:
        return self.instance.x_min

    @property
    def f_min(self):
        return self.instance.f_min


class ZDT1(AbstractFunction):
    def __init__(self, dim: int, seed: int | None = None, loggers: list[AbstractLogger] | None = None) -> None:
        super().__init__(
            seed,
            dim,
            loggers
        )

    def _create_config_space(self):
        lower_bounds = [0] * self.dim
        upper_bounds = [1] * self.dim
        return ConfigurationSpace(
            {f"x_{i}": Float(bounds=(lower_bounds[i], upper_bounds[i]), default=0.5, name=f"x_{i}") for i in range(self.dim)},
            seed=self.seed
        )

    def _function(self, x: np.ndarray) -> np.ndarray:
        if self.dim == 1:
            x = np.expand_dims(x, axis=1)
        f1 = x[0]
        if self.dim > 1:
            g = 1 + 9 * np.sum(x[1:]) / (self.dim - 1)
            f2 = g * (1 - np.sqrt(f1 / g))
        else:
            f2 = 1 - np.sqrt(f1)

        return np.array([f1, f2])

    @property
    def x_min(self) -> np.ndarray | None:
        # TODO paretofront sth
        return None

    @property
    def f_min(self):
        # TODO
        return None

class ZDT3(AbstractFunction):
    def __init__(self, dim: int, seed: int | None = None, loggers: list[AbstractLogger] | None = None) -> None:
        super().__init__(
            seed,
            dim,
            loggers
        )

    def _create_config_space(self):
        lower_bounds = [0] * self.dim
        upper_bounds = [1] * self.dim
        return ConfigurationSpace(
            {f"x_{i}": Float(bounds=(lower_bounds[i], upper_bounds[i]), default=0.5, name=f"x_{i}") for i in
             range(self.dim)},
            seed=self.seed
        )

    def _function(self, x: np.ndarray) -> np.ndarray:
        if self.dim == 1:
            x = np.expand_dims(x, axis=1)
        f1 = x[0]
        g = 1 + 9 * np.sum(x[1:]) / (self.dim - 1)
        f2 = g * (1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1))

        return np.array([f1, f2])

    @property
    def x_min(self) -> np.ndarray | None:
        # TODO
        return None

    @property
    def f_min(self):
        # TODO
        return None
