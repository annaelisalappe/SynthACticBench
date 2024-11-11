from __future__ import annotations

import numpy as np
from ConfigSpace import ConfigurationSpace, Float
from carps.loggers.abstract_logger import AbstractLogger

from synthacticbench.abstract_function import AbstractFunction

class Rosenbrock(AbstractFunction):
    def __init__(self, dim: int, seed: int | None = None, loggers: list[AbstractLogger] | None = None) -> None:
        super().__init__(
            seed,
            dim,
            loggers
        )

    def _create_config_space(self):
        lower_bounds = [self.lower_bound] * self.dim
        upper_bounds = [self.upper_bound] * self.dim
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
    
    @property
    def lower_bound(self) -> int | float:
        return -5
    
    @property
    def upper_bound(self) -> int | float:
        return 10

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
        lower_bounds = [self.lower_bound] * self.dim
        upper_bounds = [self.upper_bound] * self.dim
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
    
    @property
    def lower_bound(self) -> int | float:
        return -32.768
    
    @property
    def upper_bound(self) -> int | float:
        return 32.768

    def _function(self, x: np.ndarray) -> np.ndarray:
        if self.dim == 1:
            x = np.expand_dims(x, axis=1)
        part1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2, axis=1) / self.dim))
        part2 = -np.exp(np.sum(np.cos(2 * np.pi * x), axis=1) / self.dim)

        return part1 + part2 + 20 + np.e

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
