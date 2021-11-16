import numpy as np
import subprocess

import tensorflow as tf
import trieste
import gpflow
import timeit


def try_get_number(s):
    try:
        float(s)
    except ValueError:
        return None


def single_run(x):
    # we expect `x` to be numpy array (1,D)
    # if it's a tensor, or 2d, convert
    if tf.is_tensor(x):
        x = x.numpy()

    if len(x.shape) == 2:
        x = x[0]

    # first line is the number of elements
    # then one element per line
    np.savetxt("design_vector_0.dat", np.insert(x, 0, len(x), axis=0))

    # invoke Igloo
    subprocess.run(["./Pb_test/run.sh"], cwd="./Pb_test")

    # both files looks like this:
    # 1
    # 0
    # 0.123456789
    #
    # Meaning of first two lines isn't clear, last line is the value we need
    # Last line can sometimes be missing, or NaN, which means what exactly?
    
    with open('simulation_result_0.dat') as output_file:
        all_lines = output_file.readlines()
        if len(all_lines) != 3:
            objective_value = None
        else:
            objective_value = try_get_number(all_lines[-1])

    with open('simulation_result_1.dat') as output_file:
        all_lines = output_file.readlines()
        if len(all_lines) != 3:
            constraint_value = None
        else:
            constraint_value = try_get_number(all_lines[-1])

    return objective_value, constraint_value

from trieste.data import Dataset

OBJECTIVE = "OBJECTIVE"
CONSTRAINT = "CONSTRAINT"
FAILURE = "FAILURE"

def observer(query_points):
    objective_values = []
    constraint_values = []
    failure_values = []
    for point in query_points:
        ov, cv = single_run(point)
        if ov is None or cv is None:
            # can one be None but not the other? what to do if that happens?
            failure_values.append(0.0)
        else:
            failure_values.append(1.0)
            objective_values.append(ov)
            constraint_values.append(cv)

    return {
        OBJECTIVE: Dataset(query_points, tf.convert_to_tensor(objective_values, dtype=tf.float64)),
        CONSTRAINT: Dataset(query_points, tf.convert_to_tensor(constraint_values, dtype=tf.float64)),
        FAILURE: Dataset(query_points, tf.convert_to_tensor(failure_values, dtype=tf.float64))
    }


search_space = trieste.space.Box(
    [0.02, 0.03, 0.03, 0.03, 0.03, 0.02, 0.01, 0.005, -0.05, -0.06, -0.07, -0.08, -0.08, -0.07, -0.05, -0.03],
    [0.05, 0.08, 0.10, 0.10, 0.10, 0.08, 0.07,  0.05, -0.02, -0.02, -0.03, -0.03, -0.03, -0.01,  0.00,  0.00]
)


# num_initial_points = search_space.dimension  # R code uses twice that
num_initial_points = 3
initial_query_points = search_space.sample(num_initial_points)

start = timeit.default_timer()
initial_data = observer(initial_query_points)
stop = timeit.default_timer()

print(f"Time for initial points: {stop - start:.0f}s")
