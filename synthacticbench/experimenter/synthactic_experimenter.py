import argparse

from carps.utils.running import make_optimizer
from carps.utils.trials import TrialInfo, TrialValue
from omegaconf import DictConfig
from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor
from rich import inspect

from synthacticbench.functions import RelevantParameters
from synthacticbench.synthacticbench_problem import SynthACticBenchProblem

EXP_CONFIG_FILE_PATH = "config/experiment_config.yml"
DB_CRED_FILE_PATH = "config/database_cred.yml"

def run_config(config: dict, result_processor: ResultProcessor, custom_config: dict):
    # parse config to instantiate benchmark problem and optimizer
    synthactic = SynthACticBenchProblem(RelevantParameters(num_quadratic=4, dim=10, seed=42))
    inspect(synthactic)

    # ToDo: What optimizers to consider and how to configure them?
    opt_cfg = DictConfig()
    optimizer = make_optimizer(opt_cfg, synthactic)
    inspect(optimizer)

    # obtain incumbent through running the optimizer
    # ToDo: How to access the optimization trace?
    inc_tuple = optimizer.run()

    trial_info: TrialInfo = inc_tuple[0]
    trial_value: TrialValue = inc_tuple[1]

    result_processor.process_results({
        "incumbent": str(trial_info.config),
        "incumbent_cost": str(trial_value.cost),
        "incumbent_found_at": str(trial_value.virtual_time)
    })
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
        use_codecarbon=False
    )
    parser = argparse.ArgumentParser(
        prog='SynthACtic Bench Experimenter',
        description='This is the benchmark executor of SynthACtic Bench.')

    parser.add_argument('-s', '--setup', action='store_true', required=False)
    parser.add_argument('-e', '--exec', action='store', help="Run the benchmark executor for a certain number of experiments.", required=False, default=0)

    args = parser.parse_args()
    if args.setup:
        setup_table(experimenter)

    if args.exec != 0:
        run_experiments(experimenter, args.exec)


