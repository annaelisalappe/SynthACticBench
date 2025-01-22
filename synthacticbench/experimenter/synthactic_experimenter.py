from __future__ import annotations

import argparse
import json

from carps.loggers.abstract_logger import AbstractLogger
from carps.utils.running import make_optimizer, make_problem
from carps.utils.trials import TrialInfo, TrialValue
from carps.utils.types import Incumbent
from omegaconf import OmegaConf
from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor
from rich import inspect
import numpy as np


EXP_CONFIG_FILE_PATH = "config/experiment_config.yml"
DB_CRED_FILE_PATH = "config/database_cred.yml"

scenario_paths = {
    "c1": "synthacticbench/configs/problem/SynthACticBench/C1-RelevantParameters.yaml",
    "c2-ackley": "synthacticbench/configs/problem/SynthACticBench/"
    "C2-ParameterInteractions-ackley.yaml",
    "c2-rosenbrock": "synthacticbench/configs/problem/SynthACticBench/"
    "C2-ParameterInteractions-rosenbrock.yaml",
    "c3": "synthacticbench/configs/problem/SynthACticBench/C3-MixedTypes.yaml",
    "c4": "synthacticbench/configs/problem/SynthACticBench/C4-ActivationStructures.yaml",
    "c5": "synthacticbench/configs/problem/SynthACticBench/C5-ShiftingDomains.yaml",
    "c6": "synthacticbench/configs/problem/SynthACticBench/C6-HierarchicalStructures.yaml",
    "c7": "synthacticbench/configs/problem/SynthACticBench/C7-InvalidParameterization.yaml",
    "o1": "synthacticbench/configs/problem/SynthACticBench/O1-DeterministicObjective.yaml",
    "o2": "synthacticbench/configs/problem/SynthACticBench/O2-NoisyEvaluation.yaml",
    "o3": "synthacticbench/configs/problem/SynthACticBench/O3-MultipleObjectives.yaml",
    "o4-OP": "synthacticbench/configs/problem/SynthACticBench/O4-TimeDependentOP.yaml",
    "o4-NOP": "synthacticbench/configs/problem/SynthACticBench/O4-TimeDependentNOP.yaml",
    "o5": "synthacticbench/configs/problem/SynthACticBench/O5-CensoredObjective.yaml",
    "o6": "synthacticbench/configs/problem/SynthACticBench/O6-Multimodal.yaml",
    "o7": "synthacticbench/configs/problem/SynthACticBench/O7-SinglePeak.yaml",
}


class PyExperimenterLogger(AbstractLogger):

    def __init__(self, result_processor: ResultProcessor):
        super().__init__()
        self.result_processor: ResultProcessor = result_processor
        self.trial_buffer = []

    def log_trial(self, n_trials: float, trial_info: TrialInfo, trial_value: TrialValue,
                  n_function_calls: int | None = None) -> None:
        try:
            config_dict = dict(trial_info.config)
            for key in config_dict.keys():
                if isinstance(config_dict[key], np.int64):
                    config_dict[key] = int(config_dict[key])

            self.result_processor.process_logs({
                "trial_log": {
                    "n_trials": str(n_trials),
                    "trial_config": json.dumps(config_dict),
                    "instance": trial_info.instance,
                    "trial_cost": str(trial_value.cost),
                    "n_function_calls": str(n_function_calls)
                }
            })
        except Exception as e:
            print(e)
            print(trial_info)
            print(trial_info.config)
            exit()

    def log_incumbent(self, n_trials: int, incumbent: Incumbent) -> None:
        config_dict = dict(incumbent[0].config)
        for key in config_dict.keys():
            if isinstance(config_dict[key], np.int64):
                config_dict[key] = int(config_dict[key])
        self.result_processor.process_logs({
            "incumbent_log": {
                "n_trials": str(n_trials),
                "incumbent": json.dumps(config_dict),
                "incumbent_cost": str(incumbent[1].cost)
            }
        })
    def log_arbitrary(self, data: dict, entity: str) -> None:
        pass

def run_config(config: dict, result_processor: ResultProcessor, custom_config: dict):
    print("PyExperimenter fetched this experiment config", config)
    algorithm_configurator_name = config["algorithm_configurator"]
    scenario = config["scenario"]
    seed: int = int(config["seed"])
    n_trials: int = int(config["n_trials"])
    num_instances: int = int(config["num_instances"])

    try:
        problem_task_cfg = OmegaConf.load(scenario_paths[scenario])
    except KeyError as err:
        raise Exception(f"Unknown SynthACticBench scenario: {scenario}") from err

    synthactic_problem = make_problem(problem_task_cfg)
    #inspect(synthactic_problem)

    # generate instances
    mean = 0
    std = 2
    instance_generator = np.random.default_rng(seed=seed)
    sampled_values = instance_generator.normal(loc=mean, scale=std, size=num_instances)

    instance_map = {}
    instances = []
    for i in range(num_instances):
        name = "i" + str(i)
        instance_map[name] = sampled_values[i]
        instances.append(name)
    synthactic_problem.set_instances(instance_map)
    synthactic_problem.loggers.append(PyExperimenterLogger(result_processor))

    if algorithm_configurator_name in ["smac", "random", "irace"]:
        algorithm_configurator_cfg = None
        if algorithm_configurator_name == "smac":
            algorithm_configurator_cfg = OmegaConf.load("config/smac20-ac.yml")
            algorithm_configurator_cfg.outdir = "smac_out"
            algorithm_configurator_cfg.optimizer.smac_cfg.scenario.instances = instances
        elif algorithm_configurator_name == "random":
            algorithm_configurator_cfg = OmegaConf.load("config/randomsearch.yml")
        elif algorithm_configurator_name == "irace":
            algorithm_configurator_cfg = OmegaConf.load("config/irace.yml")
            algorithm_configurator_cfg.instances = instances

    algorithm_configurator_cfg.merge_with(problem_task_cfg)
    algorithm_configurator_cfg.seed = seed
    algorithm_configurator_cfg.task.n_trials = n_trials

    algorithm_configurator = make_optimizer(algorithm_configurator_cfg, synthactic_problem)

    if algorithm_configurator_name == "irace":
        algorithm_configurator.set_instances(instances)

    # obtain incumbent through running the optimizer
    # ToDo: How to access the optimization trace?
    inc_tuple = algorithm_configurator.run()

    f_min = synthactic_problem.f_min
    trial_info: TrialInfo = inc_tuple[0]
    x_hat = np.array(list(trial_info.config.values()))
    cost_hat = synthactic_problem.function._function(x_hat)
    trial_value: TrialValue = inc_tuple[1]

    config_dict = dict(trial_info.config)
    for key in config_dict.keys():
        if isinstance(config_dict[key], np.int64):
            config_dict[key] = int(config_dict[key])

    res = {
        "incumbent": json.dumps(config_dict),
        "incumbent_cost": str(cost_hat),
        "incumbent_found_at": str(trial_value.virtual_time),
        "done": "true",
    }

    if f_min is not None:
        res["f_min"] = f_min
        res["regret"] = str(cost_hat - f_min)
    print(res)

    result_processor.process_results(res)


def setup_table(experimenter: PyExperimenter):
    experimenter.fill_table_from_config()


def run_experiments(experimenter: PyExperimenter, num_experiments: int = 1):
    experimenter.execute(
        experiment_function=run_config,
        max_experiments=num_experiments,
    )


if __name__ == "__main__":
    experimenter = PyExperimenter(
        experiment_configuration_file_path=EXP_CONFIG_FILE_PATH,
        database_credential_file_path=DB_CRED_FILE_PATH,
        use_codecarbon=False,
    )
    parser = argparse.ArgumentParser(
        prog="SynthACtic Bench Experimenter",
        description="This is the benchmark executor of SynthACtic Bench.",
    )

    parser.add_argument("-s", "--setup", action="store_true", required=False)
    parser.add_argument(
        "-e",
        "--exec",
        action="store",
        help="Run the benchmark executor for a certain number of experiments.",
        required=False,
        default=0,
    )

    args = parser.parse_args()
    if args.setup:
        setup_table(experimenter)

    if args.exec != 0:
        run_experiments(experimenter, int(args.exec))
