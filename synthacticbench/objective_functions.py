from __future__ import annotations

import math

import numpy as np
from ConfigSpace import (
    ConfigurationSpace,
    Float,
)
from numpy import ndarray

from synthacticbench.abstract_function import AbstractFunction, RightCensoredException
from synthacticbench.base_functions import (
    ZDT1,
    ZDT3,
    Ackley,
    Griewank,
    SumOfQ,
)

class DeterministicObjective(AbstractFunction):
    """
    Deterministic Objective - o1
    This benchmark reflects algorithm configuration scenarios where the output of deterministic
    algorithms is also deterministic.
    An important capability of the optimizer is recognizing when an objective function is
    deterministic, allowing it to avoid wasting resources on multiple evaluations of the same
    solution candidate.
    """

    def __init__(self, wrapped_bench: AbstractFunction):
        self.wrapped_bench = wrapped_bench
        super().__init__(
            seed=wrapped_bench.seed,
            dim=wrapped_bench.dim,
            loggers=wrapped_bench.loggers,
        )
        self.benchmark_name = "o1"
        self._configspace = self._create_config_space()

    def _create_config_space(self):
        return self.wrapped_bench.configspace

    def _function(self, x: ndarray) -> float:
        f_eval = self.wrapped_bench._function(x=x)
        return f_eval.item()

    @property
    def x_min(self) -> np.ndarray | None:
        return self.wrapped_bench.x_min

    @property
    def f_min(self) -> float:
        return self.wrapped_bench.f_min


class NoisyEvaluation(AbstractFunction):
    """
    Noisy Evaluation - o2
    This function that simulates real-world algorithm configuration problems where evaluations
    of candidate solutions are noisy. This noise can stem from non-deterministic algorithms
    or from external factors like system load or environmental conditions.

    The objective function for this benchmark is a composition of the base evaluation of a
    parameterization (via the SumOfQ function) and an independent noise term. The noise term
    can follow different distributions, such as normal, uniform, or exponential, and is added
    to the objective function value.
    """

    def __init__(
        self,
        dim: int,
        distribution: str = "uniform",
        seed: int | None = None,
        loggers: list | None = None,
        **kwargs,
    ) -> None:
        """
        Initializes the benchmark.

        Args:
            dim (int): The number of parameters to be optimized (dimensions).
            distribution (str): The distribution of the noise that is added to the function
                evaluations. Must be one of "uniform", "normal, "exponential" or "no_noise".
            seed (int | None, optional): Random seed for reproducibility. Default is None.
            loggers (list | None, optional): List of logger objects. Default is None.
        """
        super().__init__(
            seed=seed, dim=dim, loggers=loggers
        )
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
            self.expd_value = mean
            return lambda: self.rng.normal(mean, stddev)

        if distribution == "uniform":
            low = kwargs.get("low", -1)
            high = kwargs.get("high", 1)
            self.expd_value = (low + high) * 0.5
            return lambda: self.rng.uniform(low, high)

        if distribution == "exponential":
            lambd = kwargs.get("lambd", 1)
            self.expd_value = 1 / lambd
            return lambda: self.rng.exponential(1 / lambd)

        if distribution == "no_noise":
            mean = kwargs.get("mean", 0)
            stddev = kwargs.get("stddev", 0)
            self.expd_value = 0
            return lambda: self.rng.normal(mean, stddev)

        raise ValueError("Unsupported distribution type")

    def _function(self, x: np.ndarray) -> float:
        base_value = self.instance._function(x)
        return base_value + self.noise_generator()

    @property
    def x_min(self) -> np.ndarray | None:
        """
        Returns the optimal parameter vector `x_min` for the benchmark function,
        adjusted with the expected value of the noise.

        Returns:
            np.ndarray | None: The optimal parameter vector with the expected (mean) noise.
        """
        return self.instance.x_min + self.expd_value

    @property
    def f_min(self) -> float:
        """
        Returns the minimum value of the objective function `f_min`. This is independent of
        the noise.

        Returns:
            float: The minimum value of the objective function (SumOfQ instance).
        """
        return self.instance.f_min


class MultipleObjectives(AbstractFunction):
    """
    Multiple Objectives - o3
    This benchmark resembles algorithm configuration scenarios where multiple objective need
    to be optimized simultaneously. It incorporates mathematical test
    functions that feature a set of Pareto-optimal solutions.
    """

    def __init__(
        self,
        dim: int,
        name: str = "zdt1",
        seed: int | None = None,
        loggers: list | None = None,
    ) -> None:
        """
        Initializes the benchmark.

        Args:
            dim (int): The number of parameters to be optimized (dimensions).
            seed (int | None, optional): Random seed for reproducibility. Default is None.
            loggers (list | None, optional): List of logger objects. Default is None.
        """
        super().__init__(
            seed=seed, dim=dim, loggers=loggers
        )
        self.name = name
        self.rng = np.random.default_rng(seed=seed)

        if self.name.lower() == "zdt1":
            self.instance = ZDT1(dim, seed)
            self.pareto_bound = np.array([1,0])
        elif self.name.lower() == "zdt3":
            self.instance = ZDT3(dim, seed)
            self.pareto_bound = np.array([0.8518, -0.77336856])

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

    def _compute_regret(self, f_evals: np.ndarray) -> float:
        # compute hypervolume of fmins
        sorted_fmins = self.f_min[np.argsort(self.f_min[:, 0])]
        sorted_fmins = np.concatenate((sorted_fmins,np.expand_dims(self.pareto_bound, axis=0)))
        ref_fmins = sorted_fmins.copy()  # Make a copy to avoid modifying the original
        ref_fmins[:, 1] = np.array([1] * ref_fmins.shape[0])

        hv_fmin = 0.0
        for i, point in enumerate(sorted_fmins[:-1]):
            width = sorted_fmins[i+1,0] - sorted_fmins[i,0]
            height = ref_fmins[i,1] - sorted_fmins[i,1]
            hv_fmin += width * height

        f_evals = np.expand_dims(np.array(f_evals),axis=0)
        sorted_fevals = f_evals[np.argsort(f_evals[:, 0])]
        sorted_fevals = np.concatenate((sorted_fevals,np.expand_dims(self.pareto_bound, axis=0)))

        ref_fmins = sorted_fevals.copy()
        ref_fmins[:, 1] = np.array([1] * ref_fmins.shape[0])

        hv_feval = 0.0
        for i, point in enumerate(sorted_fevals[:-1]):
            width = sorted_fevals[i + 1, 0] - sorted_fevals[i, 0]
            height = ref_fmins[i, 1] - sorted_fevals[i, 1]
            hv_feval += width * height

        regret = np.abs(hv_fmin - hv_feval)
        return regret


class TimeDependentOP(AbstractFunction):
    """
    Time-Dependent Order-Preserving - o4
    A benchmark modeling order-preserving time-dependent objective functions.
    This benchmark simulates scenarios where the environment or objective function changes
    over time, but the changes preserve the order of the original function.

    The class supports two types of time-dependent behaviors:
    - Linear Drift: A gradual and consistent change in the objective function over time.
    - Oscillations: Periodic fluctuations in the objective function.
    """

    def __init__(
        self,
        dim: int,
        name: str = "linear_drift",
        a: float = 1.0,
        b: float = 0.005,
        seed: int | None = None,
        loggers: list | None = None,
    ) -> None:
        """
        Initializes the benchmark.

        Args:
            dim (int): The number of parameters to be optimized (dimensions).
            name (str): Defines the type of time-dependent behavior. Must be one of
                "linear_drift" or "oscillations". Defaults to "linear_drift".
            a (float): The linear offset that is added to the time-dependent term.
                Defaults to 1.0.
            b (float): The factor that scales the time measure in the time-dependent term.
                Defaults to 0.005.
            seed (int | None, optional): Random seed for reproducibility. Default is None.
            loggers (list | None, optional): List of logger objects. Default is None.
        """
        super().__init__(
            seed=seed, dim=dim, loggers=loggers
        )
        self.benchmark_name = "o4.1"
        self.name = name
        self.a = a
        self.b = b
        self.rng = np.random.default_rng(seed=seed)
        self.timer = 0

        self.instance = SumOfQ(seed=self.seed, dim=self.dim)
        self._configspace = self.instance._create_config_space()
        self.function = self._set_function()

    def _set_function(self):
        if self.name == "linear_drift":
            return self._function_linear_drift
        if self.name == "oscillations":
            return self._function_oscillations
        raise Exception(
            "Invalid function name provided. "
            "Must be one of 'linear_drift' or 'oscillations'."
        )

    def _function_linear_drift(self, x: ndarray) -> float:
        return self.instance._function(x=x) + (self.a + self.b * self.timer)

    def _function_oscillations(self, x: ndarray) -> float:
        return self.instance._function(x=x) + math.sin(self.a + self.b * self.timer)

    def _function(self, x: ndarray) -> float:
        result = self.function(x=x)
        self.timer += (
            1  # TODO: This is not ideal; gets evaluated also when we calculate the f_min
        )
        return result

    @property
    def x_min(self) -> np.ndarray | None:
        """
        Returns the minimum of the benchmark function. The minimum is independent of the
        time-dependent sum that is added to the function value, because is order-preserving.
        """
        return self.instance.x_min

    @property
    def f_min(self) -> float:
        return self._function(self.x_min)


class TimeDependentNOP(AbstractFunction):
    """
    Time-Dependent Non-Order-Preserving - o4.2
    A benchmark modeling order-disrupting time-dependent objective functions.
    This benchmark simulates scenarios where the environment or objective function changes
    over time, and the changes do not preserve the order of the original function.

    The class supports two types of time-dependent behaviors:
    - Linear Drift: A steady linear change in the optimal parameter values over time.
    - Oscillations: Periodic shifts in the optimal parameter values.
    """

    def __init__(
        self,
        dim: int,
        name: str = "linear_drift",
        a: float = 1.0,
        b: float = 0.005,
        seed: int | None = None,
        loggers: list | None = None,
    ) -> None:
        """
        Initializes the benchmark.

        Args:
            dim (int): The number of parameters to be optimized (dimensions).
            name (str): Defines the type of time-dependent behavior. Must be one of
                "linear_drift" or "oscillations". Defaults to "linear_drift".
            a (float): The linear offset that is added to the time-dependent term.
                Defaults to 1.0.
            b (float): The factor that scales the time measure in the time-dependent term.
                Defaults to 0.005.
            seed (int | None, optional): Random seed for reproducibility. Default is None.
            loggers (list | None, optional): List of logger objects. Default is None.
        """
        super().__init__(
            seed=seed, dim=dim, loggers=loggers
        )
        self.benchmark_name = "o4.2"
        self.name = name
        self.a = a
        self.b = b
        self.rng = np.random.default_rng(seed=seed)
        self.timer = 0

        self._configspace = self._create_config_space()
        self.function = self._set_function()

    def _create_config_space(self):
        return ConfigurationSpace(
            {
                f"x_{i}": Float(
                    bounds=(-100, 100),
                    default=0,
                    name=f"x_{i}",
                )
                for i in range(self.dim)
            },
            seed=self.seed,
        )

    def _set_function(self):
        if self.name == "linear_drift":
            return self._function_linear_drift
        if self.name == "oscillations":
            return self._function_oscillations
        raise Exception(
            "Invalid function name provided. "
            "Must be one of 'linear_drift' or 'oscillations'."
        )

    def _function_oscillations(self, x: ndarray) -> float:
        lmd = self.a + math.sin(self.b * self.timer)
        return np.sum((x - lmd) ** 2)

    def _function_linear_drift(self, x: ndarray) -> float:
        lmd = self.a + self.b * self.timer
        return np.sum((x - lmd) ** 2)

    def _function(self, x: ndarray) -> float:
        result = self.function(x=x)
        self.timer += (
            1  # TODO: This is not ideal; gets evaluated also when we calculate the f_min
        )
        return result

    @property
    def x_min(self) -> np.ndarray | None:
        # TODO min
        pass

    @property
    def f_min(self) -> float:
        return self._function(self.x_min)


class CensoredObjective(AbstractFunction):
    """
    Censored Objective - o5
    This benchmark resembles algorithm configuration settings where certain qualities cannot
    be observed. E.g., when optimizing for runtime there is a cutoff on the evaluation time of
    a single algorithm configuration and only lower bounds can be observed for configurations
    hitting the timeout.

    This benchmark wraps any other benchmark and cuts off objective functions that exceed a
    certain amount.
    """

    def __init__(self, cutoff: float, wrapped_bench: AbstractFunction):
        """
        Initializes the benchmark.

        Args:
            cutoff (float): Percentage of objective function value that is still reported.
                Values above the threshold will be censored.
                The cutoff is determined relative to the wrapped function's optimum.
            wrapped_bench (AbstractFunction): Another benchmark function that is wrapped into
                this one.
        """
        self.cutoff = cutoff
        self.wrapped_bench = wrapped_bench
        super().__init__(
            seed=wrapped_bench.seed,
            dim=wrapped_bench.dim,
            loggers=wrapped_bench.loggers,
        )
        self.benchmark_name = "o5"
        self._configspace = self._create_config_space()

    def _create_config_space(self):
        return self.wrapped_bench.configspace

    def _function(self, x: ndarray) -> float:
        f_eval = self.wrapped_bench._function(x=x)

        # if the function value is more that cutoff percent worse than the minimum, return
        # infinity instead of the true
        # function value to indicate that the evaluation was not successful
        dist = np.abs(self.f_min - f_eval)
        if dist >= np.abs((self.f_min + 1e-8) * self.cutoff):
            raise RightCensoredException("Function value exceeds the censoring limit.")

        return f_eval

    @property
    def x_min(self) -> np.ndarray | None:
        return self.wrapped_bench.x_min

    @property
    def f_min(self) -> float:
        return self.wrapped_bench.f_min


class Multimodal(AbstractFunction):
    """
    Multimodal - o6
    This benchmark simulates algorithm configuration scenarios where the objective function
    landscape is highly multi-modal, featuring multiple local optima, some of which could be
    near the global optimum. It incorporates a variety of mathematical test functions
    specifically designed to exhibit such characteristics. The seed parameter
    determines the scaling factor applied to the function values.
    """

    def __init__(
        self,
        dim: int,
        seed: int | None = None,
        name: str | None = "griewank",
        loggers: list | None = None,
    ) -> None:
        """
        Initializes the benchmark.

        Args:
            dim (int): The number of parameters to be optimized (dimensions).
            seed (int | None, optional): Random seed for reproducibility. Default is None.
            loggers (list | None, optional): List of logger objects. Default is None.
        """
        super().__init__(
            seed=seed, dim=dim, loggers=loggers
        )
        self.dim = dim
        self.rng = np.random.default_rng(seed=seed)
        self.name = name

        if self.name.lower() == "ackley":
            self.instance = Ackley(dim, seed)
        elif self.name.lower() == "griewank":
            self.instance = Griewank(dim, seed)

        self._configspace = self.instance._create_config_space()
        self.benchmark_name = "o6"

    def _function(self, x: np.ndarray) -> np.ndarray:
        return self.instance._function(x)

    @property
    def x_min(self) -> np.ndarray | None:
        return self.instance.x_min

    @property
    def f_min(self):
        return self.instance.f_min


class SinglePeak(AbstractFunction):
    """
    Single Peak - o7
    A benchmark representing a single "peak" in the search space where a specific subspace
    (a hyper-cube) yields a value of 0.0, while all other regions yield a value of 1.0.

    The "peak" is defined by a width parameter (`peak_width`) and spans a hyper-cube region
    within the search space.
    """

    def __init__(
        self,
        dim: int,
        peak_width: float = 0.25,
        seed: int | None = None,
        loggers: list | None = None,
    ) -> None:
        """
        Initializes the benchmark.

        Args:
            dim (int): The number of parameters to be optimized (dimensions).
            peak_width (float): The percentage of each parameters' domain that returns the
                minimum. Must be between 0.0 and 1.0. Defaults to 0.01.
            seed (int | None, optional): Random seed for reproducibility. Default is None.
            loggers (list | None, optional): List of logger objects. Default is None.
        """
        super().__init__(
            seed=seed, dim=dim, loggers=loggers
        )
        self.rng = np.random.default_rng(seed=seed)
        self.lower_bound = -100
        self.upper_bound = 100
        self.peak_width = peak_width
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

    def _function(self, x: np.ndarray) -> float:
        if np.all((self.lower_ends <= x) & (x < self.lower_ends + self.abs_peak_width)):
            return 0.0

        return 1.0

    @property
    def x_min(self) -> np.ndarray | None:
        return [self.lower_ends, self.lower_ends + self.abs_peak_width - 1e-8]

    @property
    def f_min(self) -> float:
        return 0.0
