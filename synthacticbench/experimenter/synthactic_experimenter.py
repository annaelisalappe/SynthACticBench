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

from synthacticbench.abstract_function import RightCensoredException

EXP_CONFIG_FILE_PATH = "config/experiment_config.yml"
DB_CRED_FILE_PATH = "config/database_cred.yml"

base_path = "synthacticbench/configs/problem/SynthACticBench/"

scenario_paths = {
    "c1": base_path + "C1-RelevantParameters.yaml",
    "c2-ackley": base_path + "C2-ParameterInteractions-ackley.yaml",
    "c2-rosenbrock": base_path + "C2-ParameterInteractions-rosenbrock.yaml",
    "c3": base_path + "C3-MixedTypes.yaml",
    "c4": base_path + "C4-ActivationStructures.yaml",
    "c5": base_path + "C5-ShiftingDomains.yaml",
    "c6": base_path + "C6-HierarchicalStructures.yaml",
    "c7": base_path + "C7-InvalidParameterization.yaml",
    "o1": base_path + "O1-DeterministicObjective.yaml",
    "o2": base_path + "O2-NoisyEvaluation.yaml",
    "o3": base_path + "O3-MultipleObjectives.yaml",
    "o4-OP": base_path + "O4-TimeDependentOP.yaml",
    "o4-NOP": base_path + "O4-TimeDependentNOP.yaml",
    "o5": base_path + "O5-CensoredObjective.yaml",
    "o6": base_path + "O6-Multimodal.yaml",
    "o7": base_path + "O7-SinglePeak.yaml",
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
        if isinstance(incumbent[0], (list, tuple)):
            config_dict = dict(incumbent[0][0].config)
        else:
            config_dict = dict(incumbent[0].config)
        for key in config_dict.keys():
            if isinstance(config_dict[key], np.int64):
                config_dict[key] = int(config_dict[key])
        incumbent_cost = str(incumbent[0][1].cost) if isinstance(incumbent[0], (list, tuple)) \
        else str(incumbent[1].cost)
        self.result_processor.process_logs({
            "incumbent_log": {
                "n_trials": str(n_trials),
                "incumbent": json.dumps(config_dict),
                "incumbent_cost": incumbent_cost
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
            if scenario == "o3":
                algorithm_configurator_cfg = OmegaConf.load("config/smac20-ac-moo.yml")
            else:
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
    algorithm_configurator_cfg.task.n_trials = 50 if algorithm_configurator_name == "random" else n_trials
    if scenario == "o3":
        algorithm_configurator_cfg.task.objectives= ['quality_0', 'quality_1']

    algorithm_configurator = make_optimizer(algorithm_configurator_cfg, synthactic_problem)

    #for attr in dir(algorithm_configurator_cfg):
    #    print("obj.%s = %r" % (attr, getattr(algorithm_configurator_cfg, attr)))

    #inspect(algorithm_configurator_cfg)
    #print("INSPEECTION FINSIHED")

    #for attr in dir(algorithm_configurator):
    #    print("obj.%s = %r" % (attr, getattr(algorithm_configurator, attr)))

    if algorithm_configurator_name == "irace":
        algorithm_configurator.set_instances(instances)

    # obtain incumbent through running the optimizer
    # ToDo: How to access the optimization trace?
    inc_tuple = algorithm_configurator.run()

    f_min = synthactic_problem.f_min
    print("FMIN ", f_min)
    trial_info: TrialInfo = inc_tuple[0]
    x_hat = np.array(list(trial_info[0].config.values())) if isinstance(trial_info, (list, tuple)) \
        else np.array(list(trial_info.config.values()))

    try:
        cost_hat = synthactic_problem.function._function(x_hat)
    except (ValueError, TypeError, RightCensoredException):
        cost_hat = np.inf

    trial_value: TrialValue = inc_tuple[1]

    config_dict = dict(trial_info[0].config) if isinstance(trial_info, (list, tuple)) \
        else dict(trial_info.config)


    for key in config_dict.keys():
        if isinstance(config_dict[key], np.int64):
            config_dict[key] = int(config_dict[key])

    incumbent_found_at = str(trial_value[1].virtual_time) if isinstance(trial_value, (list, tuple)) \
        else str(trial_value.virtual_time)

    res = {
        "incumbent": json.dumps(config_dict),
        "incumbent_cost": str(cost_hat),
        "incumbent_found_at": incumbent_found_at,
        "done": "true",
    }

    if f_min is not None:
        res["f_min"] = float(f_min) if scenario != 'o3' else f_min
        res["regret"] = str(synthactic_problem.function._compute_regret(cost_hat))
    print(80*"==")
    print("RES")
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
