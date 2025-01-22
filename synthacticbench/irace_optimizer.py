from typing import Any

from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, Configuration
from ConfigSpace.hyperparameters import FloatHyperparameter, IntegerHyperparameter
from carps.benchmarks.problem import Problem
from carps.optimizers.optimizer import Optimizer
from carps.utils.task import Task
from carps.utils.trials import TrialInfo, TrialValue
from carps.utils.types import Incumbent, SearchSpace

from irace import ParameterSpace, Categorical, Real, Integer, Bool, Scenario, Experiment, irace
import numpy as np

class Instance:

    def __init__(self, instance_id:int):
        self.instance_id = instance_id

    def __repr__(self):
        return self.instance_id


class IRaceOptimizer(Optimizer):

    def __init__(self, problem: Problem, task: Task, instances: list[str] | None = None, loggers=None):
        super().__init__(problem, task, loggers)
        # ToDo: Any additional instantiations
        self.target_runner = IRaceTargetRunner(self)
        self.parameter_space = None
        self.scenario = None
        self.seed = None
        self._instances = instances

        self.convert_configspace(problem.configspace)

    def _setup_optimizer(self) -> Any:
        pass

    def set_instances(self, instances):
        self._instances = instances

    def convert_configspace(self, configspace: ConfigurationSpace) -> SearchSpace:
        param_list = []
        for param in configspace.values():
            if isinstance(param, CategoricalHyperparameter):
                param_list.append(Categorical(param.name, variants=param.choices))
            elif isinstance(param, FloatHyperparameter):
                param_list.append(Real(param.name, param.lower, param.upper, not isinstance(param, UniformFloatHyperparameter)))
            elif isinstance(param, IntegerHyperparameter):
                param_list.append(Integer(param.name, param.lower, param.upper, not isinstance(param, UniformIntegerHyperparameter)))

        self.parameter_space = ParameterSpace(param_list)

        return self.parameter_space

    def convert_to_trial(self, experiment, scenario) -> TrialInfo:
        config = Configuration(self.problem.configspace, values=experiment.configuration)
        trial_info = TrialInfo(config=config, instance=experiment.instance, name=experiment.configuration_id)
        return trial_info

    def get_current_incumbent(self) -> Incumbent:
        return self.target_runner.get_incumbent()

    def run(self):
        self.scenario = Scenario(max_experiments=self.task.n_trials, instances=self._instances, seed=42)
        irace(target_runner=self.target_runner.target_run, parameter_space=self.parameter_space, scenario=self.scenario)
        return self.target_runner.incumbent_trial_info, self.target_runner.incumbent_trial_value

    def ask(self) -> TrialInfo:
        pass

    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        pass


class IRaceTargetRunner:

    def __init__(self, optimizer: IRaceOptimizer):
        self.optimizer = optimizer

        self.incumbent_trial_info = None
        self.incumbent_trial_value = None

    def target_run(self, experiment: Experiment, scenario: Scenario) -> float:
        # convert given experiment to trial info
        trial_info = self.optimizer.convert_to_trial(experiment, scenario)
        # evaluate experiment and obtain trial value
        trial_value = self.optimizer.problem.evaluate(trial_info)

        # update incumbent if there is none so far or the performance of the new evaluation is better
        if self.incumbent_trial_value is None or trial_value.cost < self.incumbent_trial_value.cost:
            self.incumbent_trial_info = trial_info
            self.incumbent_trial_value = trial_value

        return trial_value.cost
