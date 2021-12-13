import ray
import numpy as np
import os
from run_feasible_set_problem import run_experiment
from config import make_all_configs, make_config
from trieste.observer import OBJECTIVE


def run_single_experiment(config):
    config = make_config(config)

    try:
        # Create target Directory
        os.makedirs(config.dirName)
        print("Directory ", config.dirName, " Created ")
    except FileExistsError:
        print("Directory ", config.dirName, " already exists")

    ask_tell, accuracy_global, accuracy_boundary = run_experiment(config)
    X = ask_tell._datasets[OBJECTIVE].query_points.numpy()
    Y = ask_tell._datasets[OBJECTIVE].observations.numpy()
    experiment_name = config.exp_name
    np.save(f"{config.dirName}/{experiment_name}_X", X)
    np.save(f"{config.dirName}/{experiment_name}_Y", Y)
    np.save(f"{config.dirName}/{experiment_name}_accuracy_global", accuracy_global)
    np.save(f"{config.dirName}/{experiment_name}_accuracy_boundary", accuracy_boundary)
    print(f"finished experiment {experiment_name}")


if __name__ == "__main__":

    num_workers = 5

    ray.init(num_cpus=num_workers)

    @ray.remote
    def run_single_experiment_with_ray(config):
        try:
            run_single_experiment(config)
        except:
            config = make_config(config)
            experiment_name = config.exp_name
            print(f"failed experiment {experiment_name}")


    workers = []

    configs = make_all_configs()

    for config in configs:
        worker = run_single_experiment_with_ray.remote(config)
        workers.append(worker)

    remaining_workers = workers

    while len(remaining_workers) > 0:
        ready_workers, remaining_workers = ray.wait(workers, num_returns=2)
        print(len(remaining_workers))
        workers = remaining_workers

    ray.shutdown()
