from run_feasible_set_problem import run_experiment
from config import make_all_configs, make_config

configs = make_all_configs()
config = make_config(configs[0])

ask_tell, accuracy_global, accuracy_boundary = run_experiment(config)
