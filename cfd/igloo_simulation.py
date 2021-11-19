import os
import pathlib
import time
import subprocess

import numpy as np


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
        # we expect `x` to be numpy array, shape (D,)
        # if it is 2d, check that its shape is (1, D) and convert

        assert isinstance(x, np.ndarray)
        assert len(x.shape) == 1 or len(x.shape) == 2
        if len(x.shape) == 2:
            assert x.shape[0] == 1
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
