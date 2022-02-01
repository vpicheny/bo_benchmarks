from problems import get_problem
from dataclasses import dataclass
import itertools
import numpy as np

@dataclass
class CONFIG:
    model:str  # "quantile", "hetgp" or "GPR"
    problem_name:str  # "quantile_branin" or ....
    problem = None
    seed:int
    budget:int = None
    initial_budget_per_dimension:int = 10
    budget_per_dimension:int = 100
    batch_size:int = 50
    num_inducing_points:int = 50
    num_features:int = 1000
    dirName:str = None
    num_initial_points: int = None
    results_dir:str = "results_prior"


def make_config(args):
    config = CONFIG(**args)
    config.problem = get_problem(config.problem_name)

    config.exp_name = f"problem_{config.problem_name}" \
                      f"_model_{config.model}" \
                      f"_init_{config.initial_budget_per_dimension}" \
                      f"_budget_{config.budget_per_dimension}" \
                      f"_batch_{config.batch_size}" \
                      f"_seed_{config.seed}"

    subdir_name = f"{config.problem_name}/" \
                      f"{config.model}" \
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
    initial_budget_per_dimension = [50]
    seeds = np.arange(10)
    problems = ["flat_branin_noise", "gauss_noise_branin", "exp_noise_branin", 'hartmann_3']  # ["gauss_noise_branin", "exp_noise_branin"]  # ['hartmann_3']  #["gauss_noise_branin", "exp_noise_branin"]
    batch_sizes = [25]
    budgets_per_dimension = [225]
    num_features = [1000]
    models = ["quantile"]  #, "quantile", "GPR"]  # ["quantile"]  #["hetgp", 'GPR']  #

    return list(dict_product(dict(initial_budget_per_dimension=initial_budget_per_dimension,
                                  seed=seeds,
                                  problem_name=problems,
                                  batch_size=batch_sizes,
                                  budget_per_dimension=budgets_per_dimension,
                                  num_features=num_features,
                                  model=models)))
