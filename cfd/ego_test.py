######### read igloo bin path from stdin ###############
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("igloo_bin_path", type=str, help="Path to Igloo bin directory. Required.")
parser.add_argument("--init-data-path", type=str, default=None, help="Path to optim.dat file that shall be used for initial data. If not given, new initial data is generated.")
parser.add_argument("--n-init-points", type=int, default=None, help="Nummber of initial points to generate. Defaults to (2 * number of parameters). Ignored if path to initial data is given.")

args = parser.parse_args()

igloo_bin_path = args.igloo_bin_path
init_data_path = args.init_data_path  # None if not given
n_init_points = args.n_init_points  # None if not given


######## imports, boilerplate code to work with Igloo ###########
import numpy as np
import subprocess

import time
import timeit
import os
import pathlib

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
        # both simulation_result_X.dat files look like this:
        # 1
        # 0|1
        # 0.123456789
        #
        # Meaning of first two lines isn't clear, last line is the value we need

        # Open question: last line can sometimes be missing, or NaN, which means what exactly?
        # we treat this as failure for now
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


######### Collecting initial data ###############

def read_run_data(optim_dat_path):
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
    num_initial_points = 2 * search_space.dimension if n_init_points is None else n_init_points
    initial_query_points = search_space.sample(num_initial_points)

    start = timeit.default_timer()
    initial_data = observer(initial_query_points)
    stop = timeit.default_timer()

    print(f"Time for initial points: {stop - start:.0f}s")
else:
    # read existing initial data
    initial_data = read_run_data(init_data_path)
    num_initial_points = len(initial_data[FAILURE])  # failure dataset contains all points, unlike other two

print("Initial data:")
for tag in initial_data:
    print(f"{tag}: {len(initial_data[tag])} points")
