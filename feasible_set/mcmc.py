
import trieste
import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from trieste.objectives import branin, BRANIN_SEARCH_SPACE
from trieste.space import Box
from trieste.models.gpflow.models import GaussianProcessRegression
from docs.notebooks.util.plotting import plot_bo_points, plot_function_2d
from tensorflow_probability import distributions as tfd
import gpflow
from gpflow.ci_utils import ci_niter
from gpflow import set_trainable
import matplotlib.pyplot as plt
import os

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)
gpflow.config.set_default_summary_fmt("notebook")
import matplotlib.patches as patches

# convert to float64 for tfp to play nicely with gpflow in 64
f64 = gpflow.utilities.to_default_float

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.get_logger().setLevel("ERROR")

import numpy as np


def plot_excursion_probability(title, model, query_points=None, threshold=80.0):
    def objective_function(x):
        p = excursion_probability(x, model, threshold)[0, ...]
        return p * (1. - p)

    _, ax = plot_function_2d(
        objective_function,
        search_space.lower - 0.01,
        search_space.upper + 0.01,
        grid_density=30,
        contour=True,
        colorbar=True,
        figsize=(10, 6),
        title=[title],
        xlabel="$X_1$",
        ylabel="$X_2$",
        fill=True,
    )
    if query_points is not None:
        plot_bo_points(query_points, ax[0, 0], num_initial_points)


np.random.seed(1793)
tf.random.set_seed(1793)

search_space = BRANIN_SEARCH_SPACE

# threshold is arbitrary, but has to be within the range of the function
threshold = 80.0
observer = trieste.objectives.utils.mk_observer(branin)

num_initial_points = 6
initial_query_points = search_space.sample_halton(num_initial_points)
initial_data = observer(initial_query_points)


def build_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=[0.2, 0.2])
    prior_scale = tf.cast(1.0, dtype=tf.float64)
    kernel.variance.prior = tfp.distributions.LogNormal(
        tf.math.log(variance), prior_scale
    )
    kernel.lengthscales.prior = tfp.distributions.LogNormal(
        tf.math.log(kernel.lengthscales), prior_scale
    )
    gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
    gpflow.set_trainable(gpr.likelihood, False)

    return GaussianProcessRegression(gpr)


model = build_model(initial_data)

from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition.function import ExpectedFeasibility

# Bichon criterion
delta = 1

# set up the acquisition rule and initialize the Bayesian optimizer
acq = ExpectedFeasibility(threshold, delta=delta)
rule = EfficientGlobalOptimization(builder=acq)  # type: ignore
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)


#
# num_steps = 2
# result = bo.optimize(num_steps, initial_data, model, rule)


# return normal.cdf(t)


def excursion_probability(x, model, threshold=80.):
    mean, variance = model.model.predict_f(x[None, :])
    normal = tfp.distributions.Normal(tf.cast(0, x.dtype), tf.cast(1, x.dtype))
    threshold = tf.cast(threshold, x.dtype)

    # if tf.size(threshold) == 1:
    t = (mean - threshold) / tf.sqrt(variance)
    return normal.cdf(t)
    # else:
    #     t0 = (mean - threshold[0]) / tf.sqrt(variance)
    #     t1 = (mean - threshold[1]) / tf.sqrt(variance)
    #     return normal.cdf(t1) - normal.cdf(t0)


# dataset = result.try_get_final_dataset()
# query_points = dataset.query_points.numpy()
# observations = dataset.observations.numpy()

# fitting the model only to the initial data
initial_model = build_model(initial_data)
initial_model.optimize(initial_data)

#
# updated_model = result.try_get_final_model()
#
# plot_excursion_probability(
#     "Updated probability of excursion", updated_model, query_points
# )

# def target_log_prob_fn(x):
#     sigmoid = tfp.bijectors.Sigmoid(tf.cast(0, x.dtype), tf.cast(1, x.dtype))
#     p = excursion_probability(sigmoid(x), model, threshold)
#     return tf.math.log(p * (1. - p))


# num_burnin_steps = ci_niter(300)
num_burnin_steps = ci_niter(100)
num_samples = ci_niter(100)

# Note that here we need model.trainable_parameters, not trainable_variables - only parameters can have priors!
x_param = gpflow.Parameter(0.5 * tf.ones([2, ], dtype=initial_query_points.dtype),
                           prior=tfd.Uniform(tf.zeros([2, ], dtype=initial_query_points.dtype),
                                             tf.ones([2, ], dtype=initial_query_points.dtype))
                           )


def target_log_prob_fn():
    sigmoid = tfp.bijectors.Sigmoid(tf.cast(0, x_param.dtype), tf.cast(1, x_param.dtype))
    # p = excursion_probability(x_param, model, threshold)
    # p = excursion_probability(sigmoid(x_param)[None, :], model, threshold)
    p = excursion_probability(x_param, model, threshold)
    # greater_than_zero = tf.reduce_all(x_param >= 0., axis=-1)
    # smaller_than_one = tf.reduce_all(x_param <= 1., axis=-1)
    pen1 = 1. - sigmoid(tf.reduce_sum(tf.maximum(x_param - 1., 0.), axis=-1) * 100. - 5.)
    pen2 = 1. - sigmoid(tf.reduce_sum(tf.maximum(0. - x_param, 0.), axis=-1) * 100. - 5.)
    # mask = tf.logical_and(greater_than_zero, smaller_than_one)
    # p = p * tf.cast(mask, dtype=p.dtype)[:, None]
    return tf.reduce_mean(tf.math.log(p * (1. - p) * pen1 * pen2))


hmc_helper = gpflow.optimizers.SamplingHelper(target_log_prob_fn, [x_param])


# hmc = tfp.mcmc.HamiltonianMonteCarlo(
#     target_log_prob_fn=hmc_helper.target_log_prob_fn, num_leapfrog_steps=10, step_size=1.
# )
# adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
#     hmc, num_adaptation_steps=10, target_accept_prob=f64(0.75), adaptation_rate=0.1
# )
# adaptive_hmc = hmc


def nuts_target_log_prob_fn(x_param):
    # sigmoid = tfp.bijectors.Sigmoid(tf.cast(0, x_param.dtype), tf.cast(1, x_param.dtype))
    # p = excursion_probability(x_param, model, threshold)
    p = excursion_probability(x_param, model, threshold)
    return tf.reduce_mean(tf.math.log(p * (1. - p)))


nuts = tfp.mcmc.NoUTurnSampler(nuts_target_log_prob_fn, step_size=0.1)
adaptive_hmc = nuts


@tf.function
def run_chain_fn():
    return tfp.mcmc.sample_chain(
        num_results=num_samples,
        num_burnin_steps=num_burnin_steps,
        current_state=hmc_helper.current_state,
        kernel=adaptive_hmc,
        # trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
    )


samples, _ = run_chain_fn()
parameter_samples = hmc_helper.convert_to_constrained_values(samples)

# sigmoid = tfp.bijectors.Sigmoid(tf.cast(0, x_param.dtype), tf.cast(1, x_param.dtype))
# samples2 = sigmoid(samples)
plot_excursion_probability(
    "P * (1-P)",
    initial_model, query_points=tf.concat([initial_query_points, samples[0]], axis=0)
)


def plot_target_log_prob_fn(title, model, query_points=None, threshold=80.0):
    def objective_function(x):
        sigmoid = tfp.bijectors.Sigmoid(tf.cast(0, x.dtype), tf.cast(1, x.dtype))
        # greater_than_zero = tf.reduce_all(x >= 0., axis=-1)
        # smaller_than_one = tf.reduce_all(x <= 1., axis=-1)
        pen1 = 1. - sigmoid(tf.reduce_sum(tf.maximum(x - 1., 0.), axis=-1) * 100. - 5.)
        pen2 = 1. - sigmoid(tf.reduce_sum(tf.maximum(0. - x, 0.), axis=-1) * 100. - 5.)
        # mask = tf.logical_and(greater_than_zero, smaller_than_one)
        p = excursion_probability(x, model, threshold)[0, ...]
        # p = p * tf.cast(mask, dtype=p.dtype)[:, None]
        # return p * (1. - p)
        return p * (1. - p) * pen1 * pen2

    _, ax = plot_function_2d(
        objective_function,
        search_space.lower - 1,
        search_space.upper + 1,
        grid_density=30,
        contour=True,
        colorbar=True,
        figsize=(10, 6),
        title=[title],
        xlabel="$X_1$",
        ylabel="$X_2$",
        fill=True,
    )
    rect = patches.Rectangle((0., 0.), 1., 1., linewidth=1, edgecolor='r', facecolor='none')
    ax[0, 0].add_patch(rect)

    if query_points is not None:
        plot_bo_points(query_points, ax[0, 0], num_initial_points)

plot_target_log_prob_fn(
    "target_log_prob_fn",
    initial_model, query_points=tf.concat([initial_query_points, samples[0]], axis=0)
)
