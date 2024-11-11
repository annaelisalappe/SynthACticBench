from __future__ import annotations

import numpy as np
from ConfigSpace import ConfigurationSpace, Float
from numpy import ndarray
from math import ceil, floor
from synthacticbench.utils.math import QuadraticFunction, NoisyFunction
from synthacticbench.base_functions import Rosenbrock, Ackley
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

        if self.name.lower() == "rosenbrock":
            self.instance = Rosenbrock(dim,seed)
        elif self.name.lower() == "ackley":
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


class InvalidParametrisation(AbstractFunction):
    def __init__(
        self,
        name: str,
        dim: int,
        cube_size: float,
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
        self.benchmark_name = "c7"
        self.rng = np.random.default_rng(seed=seed)

        if self.name.lower() == "rosenbrock":
            self.instance = Rosenbrock(dim,seed)
        elif self.name.lower() == "ackley":
            self.instance = Ackley(dim,seed)

        self._configspace = self.instance._create_config_space()
        
        self.cube, self.cube_dim = self._make_hypercube(cube_size)

    def _make_hypercube(self, cube_size):
            lower_bound = self.instance.lower_bound
            upper_bound = self.instance.upper_bound

            cube_dim = (abs(lower_bound) + abs(upper_bound)) * cube_size
            cube = np.array(
                self.rng.integers(
                    low=lower_bound, 
                    high=floor(upper_bound-cube_dim), 
                    size=self.dim
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
    
from __future__ import annotations

import numpy as np
from carps.loggers.abstract_logger import AbstractLogger

from synthacticbench.base_functions import Rosenbrock, Ackley, ZDT1, ZDT3
from synthacticbench.abstract_function import AbstractFunction

class NoisyEvaluation(AbstractFunction):
    def __init__(
        self,
        name: str,
        dim: int,
        loggers: list | None = None,
        seed: int | None = None,
        distribution: str = "uniform",
        **kwargs
    ) -> None:
        super().__init__(seed, dim, loggers)
        self.dim = dim
        self.rng = np.random.default_rng(seed=seed) 
        
        if name.lower() == "rosenbrock":
            self.instance = Rosenbrock(dim, seed)
        elif name.lower() == "ackley":
            self.instance = Ackley(dim, seed)
        
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

        if self.name.lower() == "zdt1":
            self.instance = ZDT1(dim,seed)
        elif self.name.lower() == "zdt3":
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

