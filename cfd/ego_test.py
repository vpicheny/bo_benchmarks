import numpy as np
import subprocess

import time
import timeit
import os
import pathlib
import sys

import tensorflow as tf
import trieste
import gpflow


def try_get_number(s):
    try:
        return float(s)
    except ValueError:
        return None


class IglooSimulationRunner():
    def __init__(self, igloo_bin_path):
        self._base_dir = None

        if not os.path.isdir(igloo_bin_path) or not os.path.isfile(os.path.join(igloo_bin_path, "igloo")):
            raise ValueError(f"{igloo_bin_path} does not appear to be an Igloo bin directory")
        self._igloo_bin_path = igloo_bin_path


    @property
    def base_dir(self):
        """ Returns an absolute path to the directory for the current BO run. Creates if it doesn't exist.
        """

        if self._base_dir is None:
            current_date_time = time.strftime("%Y%m%d_%H%M%S")
            dir = f"./bo_run_{current_date_time}"
            abs_path_dir = os.path.abspath(dir)

            # create the directory
            # all parents shall already exist, but the directory itself should not
            pathlib.Path(abs_path_dir).mkdir(parents=False, exist_ok=False)

            self._base_dir = abs_path_dir

        return self._base_dir


    def get_file_path(self, filename):
        """ Creates and returns path to a file within the current base directory.
        """
        return os.path.join(self.base_dir, filename)


    def try_get_value(self, simulation_output_filename):
        # both simlation_result_X.dat files looks like this:
        # 1
        # 0|1
        # 0.123456789
        #
        # Meaning of first two lines isn't clear, last line is the value we need
        # Last line can sometimes be missing, or NaN, which means what exactly?
        with open(self.get_file_path(simulation_output_filename)) as output_file:
            all_lines = output_file.readlines()
            if len(all_lines) != 3:
                value = None
            else:
                value = try_get_number(all_lines[-1])

            return value


    def __call__(self, x):
        """Single run of the Igloo CFD
        """
        # we expect `x` to be numpy array (1,D)
        # if it's a tensor, or 2d, convert
        if tf.is_tensor(x):
            x = x.numpy()

        if len(x.shape) == 2:
            x = x[0]

        # first line is the number of elements
        # then one element per line
        with open(self.get_file_path('design_vector_0.dat'), "w") as input_file:
            input_file.write(f"{len(x)}\n")
            np.savetxt(input_file, x, fmt='%.8f')

        # invoke Igloo
        subprocess.run(["./run.sh", "--igloo-path", self._igloo_bin_path, "--data-dir", self.base_dir], cwd="Pb_test")

        objective_value = self.try_get_value('simulation_result_0.dat')
        constraint_value = self.try_get_value('simulation_result_1.dat')

        return objective_value, constraint_value



######### read igloo bin path from stdin ###############
# we can consider adding more complex options, like name of the run, later
# in which case argsparse shall probably be used here
# or other package for proper args parsing
# for now this simple approach shall do

if len(sys.argv) != 2:
    print("Path to igloo bin is required")
    sys.exit(1)

igloo_bin_path = sys.argv[1]
igloo_cfd = IglooSimulationRunner(igloo_bin_path)


######### Trieste observer and search space ###############

from trieste.data import Dataset

OBJECTIVE = "OBJECTIVE"
CONSTRAINT = "CONSTRAINT"
FAILURE = "FAILURE"

def observer(query_points):
    objective_values = []
    constraint_values = []
    failure_values = []
    for point in query_points:
        ov, cv = igloo_cfd(point)
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


######### Collecting initial data ###############

# num_initial_points = search_space.dimension  # R code uses twice that
num_initial_points = 3
initial_query_points = search_space.sample(num_initial_points)

start = timeit.default_timer()
initial_data = observer(initial_query_points)
stop = timeit.default_timer()

print(f"Time for initial points: {stop - start:.0f}s")
