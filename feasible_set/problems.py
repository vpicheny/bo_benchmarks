from __future__ import annotations

import tensorflow as tf
from trieste.objectives import BRANIN_SEARCH_SPACE, scaled_branin
from trieste.observer import Observer
from trieste.space import Box
from trieste.types import TensorType
from trieste.data import  Dataset
from run_feasible_set_problem import make_observer

class Problem:
    fun = None
    dim:int
    threshold:float
    lower_bounds = [0, 0]
    upper_bounds = [1, 1]
    global_points = None
    boundary_points = None


def get_problem(name):
    problem = Problem
    if name == "branin_large_volume":
        problem.fun = scaled_branin
        problem.lower_bounds = BRANIN_SEARCH_SPACE.lower
        problem.upper_bounds = BRANIN_SEARCH_SPACE.upper
        problem.dim = 2
        problem.threshold = 1.
        global_points, boundary_points = _get_feasible_set_test_data(
            BRANIN_SEARCH_SPACE,
            lambda qp: Dataset(qp, scaled_branin(qp)),
            n_global=10000 * problem.dim,
            n_boundary = 2000 * problem.dim,
            threshold=problem.threshold,
        )
        problem.global_test_points = global_points
        problem.boundary_test_points = boundary_points
        return problem
    else:
        raise NotImplementedError


def _get_feasible_set_test_data(
    search_space: Box,
    observer: Observer,
    n_global: int,
    n_boundary: int,
    threshold: float,
    range_pct: float = 0.01,
) -> tuple[TensorType, TensorType]:

    boundary_done = False
    global_done = False
    boundary_points = tf.constant(0, dtype=tf.float64, shape=(0, search_space.dimension))
    global_points = tf.constant(0, dtype=tf.float64, shape=(0, search_space.dimension))

    while not boundary_done and not global_done:
        test_query_points = search_space.sample(100000)
        test_data = observer(test_query_points)
        threshold_deviation = range_pct * (
            tf.reduce_max(test_data.observations)  # type: ignore
            - tf.reduce_min(test_data.observations)  # type: ignore
        )

        mask = tf.reduce_all(
            tf.concat(
                [
                    test_data.observations > threshold - threshold_deviation,  # type: ignore
                    test_data.observations < threshold + threshold_deviation,  # type: ignore
                ],
                axis=1,
            ),
            axis=1,
        )
        boundary_points = tf.concat(
            [boundary_points, tf.boolean_mask(test_query_points, mask)], axis=0
        )
        global_points = tf.concat(
            [global_points, tf.boolean_mask(test_query_points, tf.logical_not(mask))], axis=0
        )

        if boundary_points.shape[0] > n_boundary:
            boundary_done = True
        if global_points.shape[0] > n_global:
            global_done = True

    return (
        global_points[
            :n_global,
        ],
        boundary_points[
            :n_boundary,
        ],
    )
