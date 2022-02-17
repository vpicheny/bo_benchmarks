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
    level:float = None
    budget:int = None
    initial_budget_per_dimension:int = 10
    budget_per_dimension:int = 100
    batch_size:int = 50
    num_inducing_points:int = 50
    num_features:int = 1000
    dirName:str = None
    num_initial_points: int = None
    results_dir:str = "results_quad"
    dimension: int = None
    rule:float = "TS"


def make_config(args):
    config = CONFIG(**args)
    config.problem = get_problem([config.problem_name, config.seed, config.dimension, config.level])

    config.exp_name = f"problem_{config.problem_name}" \
                      f"_model_{config.model}" \
                      f"_init_{config.initial_budget_per_dimension}" \
                      f"_budget_{config.budget_per_dimension}" \
                      f"_batch_{config.batch_size}" \
                      f"_seed_{config.seed}"

    subdir_name = f"{config.problem_name}_dim_{config.dimension}_q_{config.problem.quantile_level}/" \
                    f"{config.model}" \
                  f"_rule_{config.rule}" \
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
    seeds = np.arange(100)
    problems = ["gld"]  #, "flat_branin_noise", "gauss_noise_branin", "exp_noise_branin", 'hartmann_3']  # ["gauss_noise_branin", "exp_noise_branin"]  # ['hartmann_3']  #["gauss_noise_branin", "exp_noise_branin"]

    budgets_per_dimension = [250]
    num_features = [1000]
    models = ["quantile", "homquantile"]  #, "quantile", "GPR"]  # ["quantile"]  #["hetgp", 'GPR']  #
    dimensions = [3, 6]
    levels = [0.75, 0.95]

    batch_sizes = [50]

    rules = ["MES"]

    all_conditions = list(dict_product(dict(initial_budget_per_dimension=initial_budget_per_dimension,
                                  seed=seeds,
                                  problem_name=problems,
                                  batch_size=batch_sizes,
                                  budget_per_dimension=budgets_per_dimension,
                                  num_features=num_features,
                                  dimension=dimensions,
                                  level=levels,
                                  model=models,
                                  rule=rules,
                                  )))

    # filtering out combinations that are not needed
    all_conditions[:] = [
        exp
        for exp in all_conditions
        if not (exp["model"] in ["hetgp", "GPR"] and exp["rule"] != "TS")
    ]

    return all_conditions
