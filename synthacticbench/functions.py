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
            seed: int | None = None, 
            loggers: list | None = None
        ) -> None:
        super().__init__(
            seed, 
            loggers
        )
        self.relevant_params = relevant_params
        self.noisy_params = noisy_params
        self.total_params = relevant_params + noisy_params
        self.rng = np.random.default_rng(seed=seed)

        self._configspace = self._create_config_space()
        self.functions = self._create_functions()
        
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
        
    def _function(self, x: ndarray) -> float:
        sum = 0
        i = 0
        for func, _ in self.functions:
            sum += func.evaluate(x[i])
            i+=1
        return sum
    
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