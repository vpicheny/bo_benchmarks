######### read arguments ###############
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--igloo-bin-path", type=str, required=True, help="Path to Igloo bin directory. Required.")
parser.add_argument("--init-data-path", type=str, default=None, help="Path to optim.dat file that shall be used for initial data. If not given, new initial data is generated.")
parser.add_argument("--n-init-points", type=int, default=None, help="""Number of initial points to use. Defaults to (2 * number of input parameters).
                                                                       If --init-data-path is specified - max number of points to use from the given path (all points by default).
                                                                       If --init-data-path is not specified - number of points to generate by calling Igloo with random inputs ( by default).""")
parser.add_argument("--n-bo-steps", type=int, default=10, help="Number of Bayesian optimization steps. Defaults to 10.")

args = parser.parse_args()

igloo_bin_path = args.igloo_bin_path
init_data_path = args.init_data_path  # None if not given
n_init_points = args.n_init_points  # None if not given
n_bo_steps = args.n_bo_steps


######## imports, boilerplate code to work with Igloo ###########
import numpy as np

import timeit
import os


import tensorflow as tf
import trieste
import gpflow



from igloo_simulation import IglooSimulationRunner
igloo_cfd = IglooSimulationRunner(igloo_bin_path)

######### Trieste observer and search space ###############

from trieste.data import Dataset

OBJECTIVE = "OBJECTIVE"
CONSTRAINT = "CONSTRAINT"
FAILURE = "FAILURE"

# this follows Pb_test/ego.r
# g(x) <= 0, according to https://rdrr.io/cran/DiceOptim/man/EGO.cst.html
CONSTRAINT_THRESHOLD = 0.0


def observer(query_points):
    if tf.is_tensor(query_points):
        query_points = query_points.numpy()

    objective_values = []
    constraint_values = []
    failure_values = []
    ok_query_points = []

    for point in query_points:
        ov, cv = igloo_cfd(point)
        if ov is None or cv is None:
            # can one be None but not the other? what to do if that happens?
            failure_values.append([0.0])
        else:
            failure_values.append([1.0])
            objective_values.append([ov])
            constraint_values.append([cv])
            ok_query_points.append(point)

    return {
        OBJECTIVE: Dataset(tf.convert_to_tensor(ok_query_points, dtype=tf.float64), tf.convert_to_tensor(objective_values, dtype=tf.float64)),
        CONSTRAINT: Dataset(tf.convert_to_tensor(ok_query_points, dtype=tf.float64), tf.convert_to_tensor(constraint_values, dtype=tf.float64)),
        FAILURE: Dataset(query_points, tf.convert_to_tensor(failure_values, dtype=tf.float64))
    }


search_space = trieste.space.Box(
    [0.02, 0.03, 0.03, 0.03, 0.03, 0.02, 0.01, 0.005, -0.05, -0.06, -0.07, -0.08, -0.08, -0.07, -0.05, -0.03],
    [0.05, 0.08, 0.10, 0.10, 0.10, 0.08, 0.07,  0.05, -0.02, -0.02, -0.03, -0.03, -0.03, -0.01,  0.00,  0.00]
)

n_init_points = 2 * search_space.dimension if n_init_points is None else n_init_points

######### Collecting initial data ###############

def read_run_data(optim_dat_path, max_points=None):
    """Given a path to optim.dat file that contains data for a series of Igloo runs
    returns a dictionary of Trieste datasets built from that data.

    We assume that optim.dat has one or more lines with the following format:
    ID objective constraint input1 ... input16
    ID NaN input1 ... input16

    Above first line is for a successful Igloo run, second is for a failed run
    """
    if not os.path.isfile(optim_dat_path) or not optim_dat_path.endswith("optim.dat"):
        raise ValueError(f"{optim_dat_path} is not a valid path to optim.dat")

    with open(optim_dat_path) as f:
        all_lines = f.readlines()

    if max_points is not None:
        all_lines = all_lines[:max_points]

    objective_values = []
    constraint_values = []
    failure_values = []
    ok_query_points = []
    all_query_points = []

    for line in all_lines:
        if len(line) == 0:
            # skip empty lines
            continue

        tokens = line.split(" ")
        tokens = tokens[1:] # skip first column with run ID
        
        query_point = [float(x) for x in tokens[-search_space.dimension:]]
        all_query_points.append(query_point)
        if tokens[0] == 'NaN':
            # got failure line
            # instead of values for objective and constraint
            # failure lines contains single NaN
            assert len(tokens) == search_space.dimension + 1
            failure_values.append([0.0])
        else:
            # got good line
            assert len(tokens) == search_space.dimension + 2
            objective_value = float(tokens[0])
            constraint_value = float(tokens[1])

            failure_values.append([1.0])
            objective_values.append([objective_value])
            constraint_values.append([constraint_value])
            ok_query_points.append(query_point)

    return {
        OBJECTIVE: Dataset(tf.convert_to_tensor(ok_query_points, dtype=tf.float64), tf.convert_to_tensor(objective_values, dtype=tf.float64)),
        CONSTRAINT: Dataset(tf.convert_to_tensor(ok_query_points, dtype=tf.float64), tf.convert_to_tensor(constraint_values, dtype=tf.float64)),
        FAILURE: Dataset(tf.convert_to_tensor(all_query_points, dtype=tf.float64), tf.convert_to_tensor(failure_values, dtype=tf.float64))
    }

if init_data_path is None:
    # generate new initial data
    num_initial_points = n_init_points
    initial_query_points = search_space.sample(num_initial_points)

    start = timeit.default_timer()
    initial_data = observer(initial_query_points)
    stop = timeit.default_timer()

    print(f"Time for initial points: {stop - start:.0f}s")
else:
    # read existing initial data
    initial_data = read_run_data(init_data_path, max_points=n_init_points)
    num_initial_points = len(initial_data[FAILURE])  # failure dataset contains all points, unlike other two

print("Initial data:")
for tag in initial_data:
    print(f"{tag}: {len(initial_data[tag])} points")

################ Models and acquisitions for BO ##################
# we don't build model to handle failures for now
# see run.sh code for presumably failure-free Igloo setup
initial_data = {
    OBJECTIVE: initial_data[OBJECTIVE],
    CONSTRAINT: initial_data[CONSTRAINT]
}

def observer_fail_free(x):
    datasets = observer(x)

    return {
        OBJECTIVE: datasets[OBJECTIVE],
        CONSTRAINT: datasets[CONSTRAINT]
    }

def create_model(dataset):
    variance = tf.math.reduce_variance(dataset.observations)
    lengthscale = 0.01 * np.ones(search_space.dimension, dtype=np.float64)
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=lengthscale)
    jitter = gpflow.kernels.White(1e-12)
    gpr = gpflow.models.GPR(dataset.astuple(), kernel + jitter, noise_variance=1e-5)
    # gpflow.set_trainable(gpr.likelihood, False)
    return trieste.models.create_model(trieste.models.gpflow.GPflowModelConfig(**{
        "model": gpr,
        "optimizer": gpflow.optimizers.Scipy(),
        "optimizer_args": {
            "minimize_args": {"options": dict(maxiter=100)},
        },
    }))

models = {
    OBJECTIVE: create_model(initial_data[OBJECTIVE]),
    CONSTRAINT: create_model(initial_data[CONSTRAINT])
}

pof = trieste.acquisition.ProbabilityOfFeasibility(threshold=CONSTRAINT_THRESHOLD)
eci = trieste.acquisition.ExpectedConstrainedImprovement(
    OBJECTIVE, pof.using(CONSTRAINT)
)
rule = trieste.acquisition.rule.EfficientGlobalOptimization(eci)


############# BO run ###############
print(f"Running optimization for {n_bo_steps} steps")
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer_fail_free, search_space)

result, history = bo.optimize(n_bo_steps, initial_data, models, rule).astuple()

result_file = igloo_cfd.get_file_path("result.pickle")
history_file = igloo_cfd.get_file_path("history.pickle")

import pickle

with open(result_file, 'wb') as f:
    pickle.dump(result, f)

with open(history_file, 'wb') as f:
    pickle.dump(history, f)
