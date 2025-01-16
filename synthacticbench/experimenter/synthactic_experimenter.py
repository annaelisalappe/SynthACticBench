from __future__ import annotations

import argparse

from carps.utils.running import make_optimizer, make_problem
from carps.utils.trials import TrialInfo, TrialValue
from omegaconf import OmegaConf
from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor
from rich import inspect

EXP_CONFIG_FILE_PATH = "config/experiment_config.yml"
DB_CRED_FILE_PATH = "config/database_cred.yml"


def run_config(config: dict, result_processor: ResultProcessor, custom_config: dict):
    algorithm_configurator_name = config["algorithm_configurator"]
    scenario = config["scenario"]
    seed: int = int(config["seed"])

    if scenario == "c1":
        problem_task_cfg = OmegaConf.load(
            "synthacticbench/configs/problem/SynthACticBench/S1-RelevantParameters.yaml"
        )
    elif scenario == "c2-ackley":
        problem_task_cfg = OmegaConf.load(
            "synthacticbench/configs/problem/SynthACticBench/S2-ParameterInteractions-ackley.yaml"
        )
    elif scenario == "c2-rosenbrock":
        problem_task_cfg = OmegaConf.load(
            "synthacticbench/configs/problem/SynthACticBench/S2-ParameterInteractions-rosenbrock.yaml"
        )
    elif scenario == "c3":
        problem_task_cfg = OmegaConf.load(
            "synthacticbench/configs/problem/SynthACticBench/S3-MixedTypes.yaml"
        )
    elif scenario == "c4":
        problem_task_cfg = OmegaConf.load(
            "synthacticbench/configs/problem/SynthACticBench/S4-ActivationStructures.yaml"
        )
    elif scenario == "c5":
        problem_task_cfg = OmegaConf.load(
            "synthacticbench/configs/problem/SynthACticBench/S5-ShiftingDomains.yaml"
        )
    elif scenario == "c6":
        problem_task_cfg = OmegaConf.load(
            "synthacticbench/configs/problem/SynthACticBench/S6-HierarchicalStructures.yaml"
        )
    elif scenario == "c7":
        problem_task_cfg = OmegaConf.load(
            "synthacticbench/configs/problem/SynthACticBench/S7-InvalidParameterization.yaml"
        )
    elif scenario == "o1":
        problem_task_cfg = OmegaConf.load(
            "synthacticbench/configs/problem/SynthACticBench/O1-DeterministicObjective.yaml"
        )
    elif scenario == "o2":
        problem_task_cfg = OmegaConf.load(
            "synthacticbench/configs/problem/SynthACticBench/O2-NoisyEvaluation.yaml"
        )
    elif scenario == "o3":
        problem_task_cfg = OmegaConf.load(
            "synthacticbench/configs/problem/SynthACticBench/O3-MultipleObjectives.yaml"
        )
    elif scenario == "o4-OP":
        problem_task_cfg = OmegaConf.load(
            "synthacticbench/configs/problem/SynthACticBench/O4-TimeDependentOP.yaml"
        )
    elif scenario == "o4-NOP":
        problem_task_cfg = OmegaConf.load(
            "synthacticbench/configs/problem/SynthACticBench/O4-TimeDependentNOP.yaml"
        )
    elif scenario == "o5":
        problem_task_cfg = OmegaConf.load(
            "synthacticbench/configs/problem/SynthACticBench/O5-CensoredObjective.yaml"
        )
    elif scenario == "o6":
        problem_task_cfg = OmegaConf.load(
            "synthacticbench/configs/problem/SynthACticBench/O6-Multimodal.yaml"
        )
    elif scenario == "o7":
        problem_task_cfg = OmegaConf.load(
            "synthacticbench/configs/problem/SynthACticBench/O7-SinglePeak.yaml"
        )
    else:
        raise Exception("SynthACticBench scenario unknown")
    synthactic_problem = make_problem(problem_task_cfg)
    inspect(synthactic_problem)

    algorithm_configurator_cfg = None
    if algorithm_configurator_name == "smac":
        algorithm_configurator_cfg = OmegaConf.load("config/smac20-ac.yml")
        algorithm_configurator_cfg.outdir = "smac_out"
    elif algorithm_configurator_name == "random":
        algorithm_configurator_cfg = OmegaConf.load("config/randomsearch.yml")

    algorithm_configurator_cfg.merge_with(problem_task_cfg)
    algorithm_configurator_cfg.seed = seed
    algorithm_configurator_cfg.task.n_trials = 10

    algorithm_configurator = make_optimizer(algorithm_configurator_cfg, synthactic_problem)
    inspect(algorithm_configurator)

    # obtain incumbent through running the optimizer
    # ToDo: How to access the optimization trace?
    inc_tuple = algorithm_configurator.run()

    f_min = synthactic_problem.f_min
    trial_info: TrialInfo = inc_tuple[0]
    trial_value: TrialValue = inc_tuple[1]

    res = {
        "f_min": f_min,
        "regret": trial_value.cost - f_min,
        "incumbent": str(trial_info.config),
        "incumbent_cost": str(trial_value.cost),
        "incumbent_found_at": str(trial_value.virtual_time),
        "done": "true",
    }
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
