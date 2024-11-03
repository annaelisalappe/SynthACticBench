from __future__ import annotations

from typing import TYPE_CHECKING

from carps.benchmarks.problem import Problem

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace
    from carps.utils.trials import TrialInfo, TrialValue

    from synthacticbench.abstract_function import AbstractFunction
    from carps.loggers.abstract_logger import AbstractLogger



class SynthACticBenchProblem(Problem):
    def __init__(self, function: AbstractFunction, loggers: list[AbstractLogger] | None = None) -> None:
        super().__init__(loggers=loggers)

        self.function = function

    @property
    def configspace(self) -> ConfigurationSpace:
        return self.function.configspace

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        return self.function._evaluate(trial_info)

    @property
    def f_min(self) -> float | None:
        return self.function.f_min()