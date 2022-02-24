import numpy as np
import tensorflow as tf
import trieste
from trieste.ask_tell_optimization import AskTellOptimizer
from model_utils import build_model
from acquisition_utils import create_initial_query_points, create_acquisition_rule, extract_current_best_quantile
from trieste.data import Dataset
from config import make_all_configs, make_config


def make_observer(CONFIG):
    if CONFIG.model == "GPR":
        def obs(qp):
            qps = tf.repeat(qp, CONFIG.batch_size, axis=0)
            return Dataset(qps, CONFIG.problem.fun(qps))
        return obs
    else:
        return lambda qp: Dataset(qp, CONFIG.problem.fun(qp))


def run_quantile_experiment(CONFIG):
    np.random.seed(CONFIG.seed)
    tf.random.set_seed(CONFIG.seed)

    observer = make_observer(CONFIG)
    search_space = trieste.space.Box(CONFIG.problem.lower_bounds, CONFIG.problem.upper_bounds)
    initial_query_points = create_initial_query_points(search_space, CONFIG)
    data = observer(initial_query_points)
    model = build_model(data, CONFIG, search_space)

    acquisition_rule = create_acquisition_rule(CONFIG)
    ask_tell = AskTellOptimizer(search_space, data, model, acquisition_rule)

    num_iterations = np.int((CONFIG.budget - data.observations.shape[0]) / CONFIG.batch_size)

    all_best_x = extract_current_best_quantile(ask_tell, CONFIG)
    all_best_x = tf.repeat(all_best_x, data.observations.shape[0], axis=0)

    for iteration_count in range(num_iterations):
        query_points = ask_tell.ask()
        new_data = observer(query_points)
        ask_tell.tell(new_data)
        current_best_x = extract_current_best_quantile(ask_tell, CONFIG)

        all_best_x = tf.concat(
            [all_best_x, tf.repeat(current_best_x, new_data.observations.shape[0], axis=0)], 0,
        )

    # all_best_y = CONFIG.problem.quantile_fun(all_best_x)
    all_best_y = None
    return ask_tell, all_best_x, all_best_y




if __name__ == "__main__":

    config = make_all_configs()[0]
    config["model"] = "quantile"
    CONFIG = make_config(config)
    from IPython import embed

    run_quantile_experiment(CONFIG)


