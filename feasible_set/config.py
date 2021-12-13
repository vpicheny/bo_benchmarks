from problems import get_problem
from dataclasses import dataclass
import itertools
import numpy as np


@dataclass
class CONFIG:
    problem_name:str  # "branin_large_volume" or ....
    rule: str  # "nobatch-ranjan", "evr", "lp-ranjan"
    problem = None
    seed:int
    budget:int = None
    initial_budget_per_dimension:int = 5
    budget_per_dimension:int = 20
    batch_size:int = 5
    dirName:str = None
    num_initial_points: int = None
    results_dir:str = "results"


def make_config(args):
    config = CONFIG(**args)
    config.problem = get_problem(config.problem_name)

    config.exp_name = f"problem_{config.problem_name}" \
                      f"rule_{config.rule}" \
                      f"_batch_{config.batch_size}" \
                      f"_seed_{config.seed}"

    subdir_name = f"{config.problem_name}/" \
                  f"rule_{config.rule}" \
                  f"_init_{config.initial_budget_per_dimension}" \
                      f"_budget_{config.budget_per_dimension}" \
                      f"_batch_{config.batch_size}"

    config.dirName = f"{config.results_dir}/{subdir_name}"

    config.num_initial_points = config.initial_budget_per_dimension * config.problem.dim
    config.budget = config.budget_per_dimension * config.problem.dim

    return config


def dict_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def make_all_configs():
    seeds = np.arange(3)
    problems = ["branin_large_volume"]
    batch_sizes = [3]
    rules = ["nobatch-ranjan", "evr", "lp-ranjan"]

    return list(dict_product(dict(
                                  seed=seeds,
                                  problem_name=problems,
                                  batch_size=batch_sizes,
                                  rule=rules
                                  )))
