from __future__ import annotations

from typing import Any

import numpy as np
from carps.benchmarks.problem import Problem
from carps.loggers.abstract_logger import AbstractLogger
from carps.utils.trials import StatusType, TrialInfo, TrialValue
from ConfigSpace import ConfigurationSpace


class RightCensoredException(Exception):
    pass

class AbstractFunction(Problem):
    def __init__(
        self,
        seed: int,
        dim: int | None = None,
        loggers: list[AbstractLogger] | None = None,
    ) -> None:
        super().__init__(loggers=loggers)
        self.seed = seed
        self.dim = dim
        self._instances = None

    @property
    def configspace(self) -> ConfigurationSpace:
        return self._configspace

    def set_instances(self, instances):
        self._instances = instances

    def _instance_offset(self, instance: str) -> float:
        if instance is None:
            if self._instances is None:
                self._init_instances_default()
            return self._instances[list(self._instances.keys())[0]]
        return self._instances[instance]

    @property
    def instances(self) -> list[Any]:
        return self._instances

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        config = trial_info.config
        x = np.array(list(config.values()))
        try:
            cost = self._function(x=x)
            status = StatusType.SUCCESS
        except (ValueError, TypeError):  # Replace with relevant exceptions
            cost = np.inf
            status = StatusType.CRASHED
        except RightCensoredException:
            cost = np.inf
            status = StatusType.TIMEOUT
        instance = trial_info.instance
        inst_cost = cost + self._instance_offset(instance)

        return TrialValue(cost=inst_cost, status=status)

    def _function(self, x: np.ndarray) -> np.ndarray:
        ...

    @property
    def x_min(self) -> np.ndarray | None:
        """Return the configuration with the minimum function value.

        Returns:
        -------
        np.ndarray | None
            Point with minimum function value (if exists).
            Else, return None.
        """
        return None

    @property
    def f_min(self) -> np.ndarray | None:
        """Return the function value of the configuration with the minimum function value.

        Returns:
        -------
        np.ndarray | None
            Point with minimum function value (if exists).
            Else, return None.
        """
        return None

    def _compute_regret(self, f_eval):
        return float(np.abs(self.f_min - f_eval))

    def _init_instances_default(self):
        num_instances = 500
        mean = 0
        std = 2
        instance_generator = np.random.default_rng(seed=self.seed)
        sampled_values = instance_generator.normal(loc=mean, scale=std, size=num_instances)
        instance_map = {}
        instances = []
        for i in range(num_instances):
            name = "i" + str(i)
            instance_map[name] = sampled_values[i]
            instances.append(name)
        self.set_instances(instance_map)
