from __future__ import annotations

from abc import abstractmethod

import numpy as np
from ConfigSpace import ConfigurationSpace
from carps.benchmarks.problem import Problem
from carps.utils.trials import TrialInfo, TrialValue
from carps.loggers.abstract_logger import AbstractLogger

class AbstractFunction(Problem):
    def __init__(self, seed: int | None = None, loggers: list[AbstractLogger] | None = None) -> None:
        super().__init__()

    @property
    def configspace(self) -> ConfigurationSpace:
        return self._configspace

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        config = trial_info.config
        x = np.array(list(config.values()))
        cost = self._function(x=x)
        return TrialValue(cost=cost)

    @abstractmethod
    def _function(x: np.ndarray) -> float:
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
    
    