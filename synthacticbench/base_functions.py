from __future__ import annotations

import numpy as np
from carps.loggers.abstract_logger import AbstractLogger
from ConfigSpace import ConfigurationSpace, Float

from synthacticbench.abstract_function import AbstractFunction


class Griewank(AbstractFunction):
    def __init__(
        self,
        dim: int,
        seed: int | None = None,
        loggers: list[AbstractLogger] | None = None,
    ) -> None:
        super().__init__(seed, dim, loggers)
        self.lower_bounds = [-600] * self.dim
        self.upper_bounds = [600] * self.dim
        self._make_coefficients()

    def _create_config_space(self):
        return ConfigurationSpace(
            {
                f"x_{i}": Float(
                    bounds=(self.lower_bounds[i], self.upper_bounds[i]),
                    default=0,
                    name=f"x_{i}",
                )
                for i in range(self.dim)
            },
            seed=self.seed,
        )

    @property
    def x_min(self) -> np.ndarray | None:
        """
        The global minimum for the Griewank function occurs at the origin.
        """
        return np.array([0.0] * self.dim)

    @property
    def f_min(self) -> float | None:
        """
        The minimum value of the Griewank function is 0.
        """
        return 0.0

    def _make_coefficients(self) -> np.ndarray:
        generator = np.random.default_rng(seed=self.seed)
        self.coefficients = generator.uniform(low=0.1, high=30, size=(1, 1))

    def _function(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the Griewank function for input x.
        Formula:
            f(x) = 1 + (1/d) * sum(x_i^2) - product(cos(x_i / sqrt(i)))
        """
        x = np.atleast_2d(x)  # Ensure x is at least 2D
        d = x.shape[1]  # Dimensionality of the input
        sum_term = np.sum(x**2, axis=1) / 4000  # Sum of x_i^2 divided by 4000
        prod_term = np.prod(
            np.cos(x / np.sqrt(np.arange(1, d + 1))), axis=1
        )  # Product of cos(x_i / sqrt(i))
        return self.coefficients[0] * (1 + sum_term - prod_term)


class Rosenbrock(AbstractFunction):
    def __init__(
        self,
        dim: int,
        seed: int | None = None,
        loggers: list[AbstractLogger] | None = None,
    ) -> None:
        super().__init__(seed, dim, loggers)
        self.lower_bounds = [-5] * self.dim
        self.upper_bounds = [10] * self.dim
        self._make_coefficients()

    def _create_config_space(self):
        return ConfigurationSpace(
            {
                f"x_{i}": Float(
                    bounds=(self.lower_bounds[i], self.upper_bounds[i]),
                    default=0,
                    name=f"x_{i}",
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

    def _make_coefficients(self) -> np.ndarray:
        generator = np.random.default_rng(seed=self.seed)
        self.coefficients = generator.uniform(low=1, high=1, size=(1, 2))[0]

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
        self.lower_bounds = [-32.768] * self.dim
        self.upper_bounds = [32.768] * self.dim
        self._make_coefficients()

    def _create_config_space(self):
        return ConfigurationSpace(
            {
                f"x_{i}": Float(
                    bounds=(self.lower_bounds[i], self.upper_bounds[i]),
                    default=0,
                    name=f"x_{i}",
                )
                for i in range(self.dim)
            },
            seed=self.seed,
        )

    def _make_coefficients(self) -> np.ndarray:
        generator = np.random.default_rng(seed=self.seed)

        self.coefficients = generator.uniform(low=1, high=1, size=(1, 1))

    @property
    def x_min(self) -> np.ndarray | None:
        return np.zeros(self.dim)

    @property
    def f_min(self) -> float | None:
        return 0.0

    def _function(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x) if self.dim > 1 else x
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
        """
        Returns the set of x configurations that yield the Pareto front.
        For ZDT1, this means x1 varies between [0, 1] and the remaining xi = 0.
        """
        x_min = np.zeros((100, self.dim))  # Discretize Pareto set into 100 points
        x_min[:, 0] = np.linspace(0, 1, 100)
        return x_min

    @property
    def f_min(self) -> np.ndarray:
        """
        Returns the Pareto-optimal function values.
        For ZDT1, f1 = x1 and f2 = 1 - sqrt(f1) when the Pareto front is achieved.
        """
        x = np.linspace(0, 1, 100)
        return np.array([x, 1 - np.sqrt(x)]).T


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
        regions = [
            [0, 0.0830015349],
            [0.182228780, 0.2577623634],
            [0.4093136748, 0.4538821041],
            [0.6183967944, 0.6525117038],
            [0.8233317983, 0.8518328654],
        ]
        num_points_per_region = 20  # Number of points to discretize each region
        x_min = []
        for lower, upper in regions:
            x1_values = np.linspace(lower, upper, num_points_per_region)
            for x1 in x1_values:
                x = np.zeros(self.dim)  # All xi = 0
                x[0] = x1
                x_min.append(x)

        return np.array(x_min)

    @property
    def f_min(self):
        regions = [
            [0, 0.0830015349],
            [0.182228780, 0.2577623634],
            [0.4093136748, 0.4538821041],
            [0.6183967944, 0.6525117038],
            [0.8233317983, 0.8518328654],
        ]

        pf = []
        for r in regions:
            x1 = np.linspace(r[0], r[1], 20)
            x2 = 1 - np.sqrt(x1) - x1 * np.sin(10 * np.pi * x1)
            pf.append(np.array([x1, x2]).T)

        return np.row_stack(pf)


class SumOfQ(AbstractFunction):
    def __init__(
        self,
        seed: int,
        dim: int | None = None,
        lower_bounds: list[float] | None = None,
        upper_bounds: list[float] | None = None,
        loggers: list[AbstractLogger] | None = None,
    ) -> None:
        super().__init__(seed, dim, loggers)
        self.lower_bounds = lower_bounds or [-100] * self.dim
        self.upper_bounds = upper_bounds or [100] * self.dim
        print(f"Lower bound: {self.lower_bounds[0]}, {len(self.lower_bounds)}")
        self._make_coefficients()

    def get_value_in_single_dim(self, x: float, i: int) -> float:
        a, b, c = (
            self.coefficients[i, 0],
            self.coefficients[i, 1],
            self.coefficientsi[i, 2],
        )
        return a * x**2 + b * x + c

    def _create_config_space(self):
        return ConfigurationSpace(
            {
                f"x_{i}": Float(
                    bounds=(self.lower_bounds[i], self.upper_bounds[i]),
                    default=0,
                    name=f"x_{i}",
                )
                for i in range(self.dim)
            },
            seed=self.seed,
        )

    def _make_coefficients(self) -> np.ndarray:
        generator = np.random.default_rng(seed=self.seed)

        self.coefficients = generator.uniform(low=-10, high=10, size=(self.dim, 3))

    @property
    def x_min(self) -> np.ndarray:
        """
        Computes the x_min (the locations of minima) for each quadratic function.

        Returns:
            np.ndarray: Array of minima for each dimension.
        """
        if self._x_min:
            return self._x_min

        a, b, c = (
            self.coefficients[:, 0],
            self.coefficients[:, 1],
            self.coefficients[:, 2],
        )
        x_min = []
        for i in range(len(a)):
            if a[i] > 0:
                # Compute the minimum analytically
                x = -b[i] / (2 * a[i])
                # Clip to bounds
                x = np.clip(x, self.lower_bounds[i], self.upper_bounds[i])
                x_min.append(x)
            else:
                # Quadratic is not convex, evaluate at the boundaries
                x_left = self.lower_bounds[i]
                x_right = self.upper_bounds[i]
                y_left = a[i] * x_left**2 + b[i] * x_left + c[i]
                y_right = a[i] * x_right**2 + b[i] * x_right + c[i]
                x_min.append(x_left if y_left < y_right else x_right)

        self._x_min = np.array(x_min)

        return self._x_min

    @property
    def f_min(self) -> float:
        """
        Computes the function value at the minima (f_min).

        Returns:
            float: The sum of function values at the minima for all dimensions.
        """
        # Use the already computed x_min values

        x_min_values = self.x_min
        return self._function(x_min_values)

    def _function(self, x: np.ndarray) -> float:
        a, b, c = (
            self.coefficients[:, 0],
            self.coefficients[:, 1],
            self.coefficients[:, 2],
        )

        return np.sum(a * x**2 + b * x + c)
