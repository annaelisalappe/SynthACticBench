from __future__ import annotations

import numpy as np
from carps.benchmarks.problem import Problem
from carps.loggers.abstract_logger import AbstractLogger
from carps.utils.trials import TrialInfo, TrialValue, StatusType
from ConfigSpace import ConfigurationSpace


class AbstractFunction(Problem):
    def __init__(
        self,
        seed: int,
        instance_parameter: float,
        dim: int | None = None,
        loggers: list[AbstractLogger] | None = None,
    ) -> None:
        super().__init__(loggers=loggers)
        self.instance_parameter = instance_parameter
        self.seed = seed
        self.dim = dim

    @property
    def configspace(self) -> ConfigurationSpace:
        return self._configspace

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        config = trial_info.config
        x = np.array(list(config.values()))
        try:
            cost = self._function(x=x)
            status = StatusType.SUCCESS
        except Exception:
            cost = np.inf
            status = StatusType.CRASHED
        inst_cost = cost + self.instance_parameter
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
