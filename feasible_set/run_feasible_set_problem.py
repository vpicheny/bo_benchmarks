import numpy as np
import tensorflow as tf
import trieste
from trieste.ask_tell_optimization import AskTellOptimizer
from model_utils import build_model
from acquisition_utils import create_acquisition_rule
from metrics_utils import compute_metrics
from trieste.data import Dataset


def make_observer(CONFIG):
    return lambda qp: Dataset(qp, CONFIG.problem.fun(qp))


def run_experiment(CONFIG):
    np.random.seed(CONFIG.seed)
    tf.random.set_seed(CONFIG.seed)

    observer = make_observer(CONFIG)
    search_space = trieste.space.Box(CONFIG.problem.lower_bounds, CONFIG.problem.upper_bounds)
    initial_query_points = search_space.sample_halton(CONFIG.num_initial_points)

    data = observer(initial_query_points)
    model = build_model(data)

    acquisition_rule = create_acquisition_rule(CONFIG, search_space)
    ask_tell = AskTellOptimizer(search_space, data, model, acquisition_rule)

    num_iterations = np.int((CONFIG.budget - data.observations.shape[0]) / acquisition_rule._num_query_points)

    accuracy_global, accuracy_boundary = compute_metrics(ask_tell, CONFIG)
    accuracy_global = tf.repeat(accuracy_global, data.observations.shape[0], axis=0)
    accuracy_boundary = tf.repeat(accuracy_boundary, data.observations.shape[0], axis=0)

    for iteration_count in range(num_iterations):
        query_points = ask_tell.ask()
        new_data = observer(query_points)
        ask_tell.tell(new_data)
        metrics = compute_metrics(ask_tell, CONFIG)
        accuracy_global = tf.concat([accuracy_global,
                                     tf.repeat(metrics[0], acquisition_rule._num_query_points, axis=0)],
                                     axis=0)
        accuracy_boundary = tf.concat([accuracy_boundary,
                                       tf.repeat(metrics[1], acquisition_rule._num_query_points, axis=0)],
                                       axis=0)

    return ask_tell, accuracy_global, accuracy_boundary
