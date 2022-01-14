import numpy as np
import os
import sys
import tensorflow as tf
sys.path.append("/home/victor/Code/trieste")
import matplotlib.pyplot as plt

from plotting import  plot_regret, create_grid, plot_gp_2d, plot_function_2d, plot_bo_points
from run_quantile_problem import run_quantile_experiment, make_observer
from config import make_all_configs, make_config
from trieste.observer import OBJECTIVE
from gpflow.utilities import print_summary


def plot_results(ask_tell, best_x, best_y, estimated_best_y, show_model=True, show_inducings=False, show_regret=False):
    final_dataset = ask_tell._datasets[OBJECTIVE]
    final_model = ask_tell._models[OBJECTIVE].model_gpflux

    query_points = final_dataset.query_points.numpy()
    observations = final_dataset.observations.numpy()
    true_scores = config.problem.quantile_fun(final_dataset.query_points).numpy()
    arg_min_idx = tf.squeeze(tf.argmin(true_scores, axis=0))

    if show_model:
        fig, _ = plot_gp_2d(final_model, ask_tell._search_space.lower, ask_tell._search_space.upper, grid_density=30)
        fig.axes[0].scatter(query_points[:, 0], query_points[:, 1], observations)
        fig.axes[0].scatter(best_x[-1, 0], best_x[-1, 1], best_y[-1], color="red")
        fig.axes[0].scatter(best_x[-1, 0], best_x[-1, 1], estimated_best_y, color="green")
        # xx, yy = np.meshgrid(range(2), range(2))
        # fig.axes[0].plot_surface(xx, yy, yy * 0 + estimated_best_y, alpha=0.2)

    if show_inducings:
        Z = final_model.f_layers[0].inducing_variable.inducing_variable.Z
        plt.figure()
        plt.scatter(Z[:, 0], Z[:, 1], color="black")
        plt.show()

    if show_regret:
        fig, ax = plt.subplots(1, 1)
        plot_regret(true_scores - config.problem.minimum, ax, num_init=config.num_initial_points, idx_best=arg_min_idx)
        ax.plot(np.arange(config.num_initial_points, config.budget + 1, config.batch_size), all_best_y - config.problem.minimum)
        # ax.set_yscale("log")
        fig.show()

    return fig


configs = make_all_configs()
config = make_config(configs[1])

# ask_tell, best_x, best_y = run_quantile_experiment(config)

#############################################
import trieste
from trieste.ask_tell_optimization import AskTellOptimizer
from model_utils import build_model
from acquisition_utils import create_initial_query_points, create_acquisition_rule, extract_current_best_quantile
from trieste.data import Dataset
from tensorflow.keras.callbacks import TensorBoard

# tb_callback = TensorBoard(log_dir="logs/tensorboard/")


CONFIG = config
np.random.seed(CONFIG.seed)
tf.random.set_seed(CONFIG.seed)

observer = make_observer(CONFIG)
search_space = trieste.space.Box(CONFIG.problem.lower_bounds, CONFIG.problem.upper_bounds)
initial_query_points = create_initial_query_points(search_space, CONFIG)
data = observer(initial_query_points)
model = build_model(data, CONFIG, search_space)  #, tb=tb_callback)

summary_writer = tf.summary.create_file_writer("logs/tensorboard/experiment4")
trieste.logging.set_tensorboard_writer(summary_writer)


acquisition_rule = create_acquisition_rule(CONFIG)
ask_tell = AskTellOptimizer(search_space, data, model, acquisition_rule)

num_iterations = np.int((CONFIG.budget - data.observations.shape[0]) / CONFIG.batch_size)

all_best_x = extract_current_best_quantile(ask_tell, CONFIG)
all_best_y = CONFIG.problem.quantile_fun(all_best_x)
mean, var = ask_tell._models[OBJECTIVE].predict(ask_tell._datasets[OBJECTIVE].query_points)
estimated_best_y = tf.reduce_min(mean[:, 0])
plot_results(ask_tell, all_best_x, all_best_y, estimated_best_y)

for step in range(num_iterations):
    trieste.logging.set_step_number(step)
    query_points = ask_tell.ask()
    new_data = observer(query_points)
    ask_tell.tell(new_data)
    current_best_x = extract_current_best_quantile(ask_tell, CONFIG)
    all_best_x = tf.concat([all_best_x, current_best_x], axis=0)
    all_best_y = CONFIG.problem.quantile_fun(all_best_x)
    mean, var = ask_tell._models[OBJECTIVE].predict(ask_tell._datasets[OBJECTIVE].query_points)
    estimated_best_y = tf.reduce_min(mean[:, 0])

    # monitor models after each tell
    if summary_writer:
        with summary_writer.as_default():
            for tag, model in ask_tell._models.items():
                with tf.name_scope(f"{tag}.model"):
                    model.log()
        with summary_writer.as_default(step=trieste.logging.get_step_number()):
            for j in range(query_points.shape[-1]):
                tf.summary.histogram(f"new_qp_x{j}", new_data.query_points[:, j])
            tf.summary.histogram(f"new_data", new_data.observations)

    plot_results(ask_tell, all_best_x, all_best_y, estimated_best_y)

# all_best_y = CONFIG.problem.quantile_fun(all_best_x)
best_x = all_best_x
best_y = all_best_y

plot_results(ask_tell, all_best_x, all_best_y, estimated_best_y, show_model=False, show_regret=True)

#############################################

print(f"finished experiment {config.exp_name}")

# grid_quantile = config.problem.quantile_fun(create_grid(ask_tell._search_space.lower,
#                                                         ask_tell._search_space.upper, grid_density=200)[0])
# actual_minimum = tf.reduce_min(grid_quantile).numpy()
#
# fig = plot_results(ask_tell, best_x, best_y)

# fig, _ = plot_function_2d(config.problem.quantile_fun, ask_tell._search_space.lower, ask_tell._search_space.upper, grid_density=30)

print_summary(ask_tell._models[OBJECTIVE].model_gpflux.f_layers[0])