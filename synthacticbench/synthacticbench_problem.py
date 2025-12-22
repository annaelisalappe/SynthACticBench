from __future__ import annotations

from carps.objective_functions.objective_function import ObjectiveFunction
from carps.loggers.abstract_logger import AbstractLogger
from carps.utils.trials import TrialInfo, TrialValue
from ConfigSpace import ConfigurationSpace

from synthacticbench.abstract_function import AbstractFunction


class SynthACticBenchProblem(ObjectiveFunction):
    def __init__(
        self, function: AbstractFunction, loggers: list[AbstractLogger] | None = None
    ) -> None:
        super().__init__(loggers=loggers)

        self.function = function

    @property
    def configspace(self) -> ConfigurationSpace:
        return self.function.configspace

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        return self.function._evaluate(trial_info)

    @property
    def f_min(self) -> float | None:
        return self.function.f_min

    def set_instances(self, instances):
        self.function.set_instances(instances)
