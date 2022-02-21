import ray
import numpy as np
import os
from run_quantile_problem import run_quantile_experiment
from config import make_all_configs, make_config
from trieste.observer import OBJECTIVE


def run_single_experiment(config):
    config = make_config(config)
    print("config built")

    try:
        # Create target Directory
        os.makedirs(config.dirName)
        print("Directory ", config.dirName, " Created ")
    except FileExistsError:
        print("Directory ", config.dirName, " already exists")

    experiment_name = config.exp_name
    experiment_done = False
    if os.path.exists(os.path.join(config.dirName, f"{experiment_name}_X.npy")):
        experiment_done = True

    if not experiment_done:
        ask_tell, best_x, best_y = run_quantile_experiment(config)
        X = ask_tell._datasets[OBJECTIVE].query_points.numpy()
        Y = ask_tell._datasets[OBJECTIVE].observations.numpy()
        experiment_name = config.exp_name
        np.save(f"{config.dirName}/{experiment_name}_X", X)
        np.save(f"{config.dirName}/{experiment_name}_Y", Y)
        np.save(f"{config.dirName}/{experiment_name}_best_x", best_x)
        np.save(f"{config.dirName}/{experiment_name}_best_y", best_y)
        np.save(f"{config.dirName}/{experiment_name}_regret", best_y - config.problem.minimum)

        print(f"finished experiment {experiment_name}")
    else:
        print(f"Skipping experiment {experiment_name}, already done!")


if __name__ == "__main__":

    num_workers = 10

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
