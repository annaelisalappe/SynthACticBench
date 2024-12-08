from __future__ import annotations

import numpy as np
from carps.loggers.abstract_logger import AbstractLogger
from ConfigSpace import ConfigurationSpace, Float

from synthacticbench.abstract_function import AbstractFunction


class Rosenbrock(AbstractFunction):
    def __init__(
        self,
        dim: int,
        seed: int | None = None,
        loggers: list[AbstractLogger] | None = None,
    ) -> None:
        super().__init__(seed, dim, loggers)

        self._make_coefficients()

    def _create_config_space(self):
        lower_bounds = [self.lower_bound] * self.dim
        upper_bounds = [self.upper_bound] * self.dim
        return ConfigurationSpace(
            {
                f"x_{i}": Float(
                    bounds=(lower_bounds[i], upper_bounds[i]), default=0, name=f"x_{i}"
                )
                for i in range(self.dim)
            },
            seed=self.seed,
        )

    @property
    def x_min(self) -> np.ndarray | None:
        return np.array([1.0] * self.dim)

    @property
    def f_min(self) -> float | None:
        return 0.0

    @property
    def lower_bound(self) -> int | float:
        return -5

    @property
    def upper_bound(self) -> int | float:
        return 10

    def _make_coefficients(self) -> np.ndarray:
        generator = np.random.default_rng(seed=self.seed)

        self.coefficients = generator.uniform(low=0, high=10, size=(1, 2))

    def _function(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x) if self.dim > 1 else x
        totals = np.zeros(x.shape[0])
        if self.dim == 1:
            totals = (1 - x) ** 2  # Simplified 1D Rosenbrock function
        else:
            for i in range(self.dim - 1):
                totals += self.coefficients[0] * (
                    100 * (x[:, i + 1] - x[:, i] ** 2) ** 2 + (1 - x[:, i]) ** 2
                )

        return self.coefficients[1] * totals


class Ackley(AbstractFunction):
    def __init__(
        self,
        dim: int,
        seed: int | None = None,
        loggers: list[AbstractLogger] | None = None,
    ) -> None:
        super().__init__(seed, dim, loggers)
        self._make_coefficients()

    def _create_config_space(self):
        lower_bounds = [self.lower_bound] * self.dim
        upper_bounds = [self.upper_bound] * self.dim
        return ConfigurationSpace(
            {
                f"x_{i}": Float(
                    bounds=(lower_bounds[i], upper_bounds[i]), default=0, name=f"x_{i}"
                )
                for i in range(self.dim)
            },
            seed=self.seed,
        )

    def _make_coefficients(self) -> np.ndarray:
        generator = np.random.default_rng(seed=self.seed)

        self.coefficients = generator.uniform(low=0, high=10, size=(1, 1))

    @property
    def x_min(self) -> np.ndarray | None:
        return np.zeros(self.dim)

    @property
    def f_min(self) -> float | None:
        return 0.0

    @property
    def lower_bound(self) -> int | float:
        return -32.768

    @property
    def upper_bound(self) -> int | float:
        return 32.768

    def _function(self, x: np.ndarray) -> np.ndarray:
        if self.dim == 1:
            x = np.expand_dims(x, axis=1)
        elif x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        part1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2, axis=1) / self.dim))
        part2 = -np.exp(np.sum(np.cos(2 * np.pi * x), axis=1) / self.dim)

        return self.coefficients[0] * (part1 + part2 + 20 + np.e)


class ZDT1(AbstractFunction):
    def __init__(
        self,
        dim: int,
        seed: int | None = None,
        loggers: list[AbstractLogger] | None = None,
    ) -> None:
        super().__init__(seed, dim, loggers)

    def _create_config_space(self):
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
    def __init__(
        self,
        dim: int,
        seed: int | None = None,
        loggers: list[AbstractLogger] | None = None,
    ) -> None:
        super().__init__(seed, dim, loggers)

    def _create_config_space(self):
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


class SumOfQ(AbstractFunction):
    def __init__(
        self,
        seed: int,
        dim: int | None = None,
        loggers: list[AbstractLogger] | None = None,
    ) -> None:
        super().__init__(seed, dim, loggers)

        self._make_coefficients()

    def _create_config_space(self):
        lower_bounds = [self.lower_bound] * self.dim
        upper_bounds = [self.upper_bound] * self.dim
        return ConfigurationSpace(
            {
                f"x_{i}": Float(
                    bounds=(lower_bounds[i], upper_bounds[i]), default=0, name=f"x_{i}"
                )
                for i in range(self.dim)
            },
            seed=self.seed,
        )

    def _make_coefficients(self) -> np.ndarray:
        generator = np.random.default_rng(seed=self.seed)

        self.coefficients = generator.uniform(low=-10, high=10, size=(self.dim, 3))

        print(self.coefficients)

    @property
    def x_min(self) -> np.ndarray | None:
        a, b, c = (
            self.coefficients[:, 0],
            self.coefficients[:, 1],
            self.coefficients[:, 2],
        )
        x_min = []
        for i in range(a):
            if a[i] > 0:
                x_min.append(-b[i] / (2 * a[i]))
            else:
                y_1 = a[i] * (-100) ** 2 + b[i] * (-100) + c[i]
                y_2 = a[i] * 100**2 + b[i] * (-100) + c[i]
                if y_1 < y_2:
                    x_min.append(-100)
                else:
                    return x_min.append(100)  # TODO: What if there is two minima?
        return np.array(x_min)

    @property
    def f_min(self) -> float | None:
        x_min_values = self.x_min()
        return self._function(x_min_values)

    @property
    def lower_bound(self) -> int | float:
        return -100

    @property
    def upper_bound(self) -> int | float:
        return 100

    def _function(self, x: np.ndarray) -> float:
        a, b, c = (
            self.coefficients[:, 0],
            self.coefficients[:, 1],
            self.coefficients[:, 2],
        )
        return np.sum(a * x**2 + b * x + c)
