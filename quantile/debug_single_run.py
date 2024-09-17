from config import make_all_configs, make_config
from run_experiments_with_ray import run_single_experiment

configs = make_all_configs()

# config = make_config(configs[0])
#

run_single_experiment(configs[0])

