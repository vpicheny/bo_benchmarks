from typing import Callable, Optional
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import gpflow
import gpflux
import numpy as np
import tensorflow as tf
from trieste.space import Box
from gpflow.config import default_float
from gpflux.models.deep_gp import sample_dgp

from gpflow.inducing_variables import (
    InducingPoints,
    SharedIndependentInducingVariables,
)

from quantile.model_utils import create_kernel_with_features
from plotting import  create_grid, plot_function_2d, plot_surface


def softplus(x):
    return tf.math.log(1.0 + tf.exp(x))

class QuantileDistribution(ABC):
    """ Abstract Base Class for QuantileDistributions """

    def __init__(self, num_latent, softplus_function: Optional[Callable] = softplus):
        self.num_latent = num_latent
        self.softplus_function = softplus_function

    @abstractmethod
    def quantile(self, tau, lambda_val):
        pass

    @abstractmethod
    def pdf(self, y, lambda_val):
        pass


class GLD(ABC):

    def __init__(self, softplus_function: Optional[Callable] = softplus):
        self.num_latent = 4
        self.softplus_function = softplus_function

    def quantile(self, tau, lambda_val):
        """
        Quantile function of the GLD distribution

        :param tau: the quantile levels, between 0 and 1 (0.5 is median), [Q] or [Q, S, N]
        :param lambda_val: realizations of the four latent GPs, [S, N, 4]
        :return: Q: quantile values [Q, S, N]
        """
        tau = tf.cast(tau, default_float())

        if tau.get_shape().ndims == 1:
            tau = tau[:, None, None]

        lambda0 = lambda_val[None, :, :, 0] * 2.
        lambda1 = self.softplus_function(lambda_val[None, :, :, 1] * 1.) / 3.
        lambda2 = lambda_val[None, :, :, 2] * .2
        lambda3 = lambda_val[None, :, :, 3] * .2

        # Special cases when lambda2 and lambda3 are zero
        l2_zeros_grid = tf.equal(tf.ones_like(tau) * lambda2, 0.)
        l3_zeros_grid = tf.equal(tf.ones_like(tau) * lambda3, 0.)

        # Trick to avoid NANs
        safe_l2 = tf.where(l2_zeros_grid,
                           tf.ones_like(tau * lambda2),
                           tf.ones_like(tau) * lambda2)

        safe_l3 = tf.where(l3_zeros_grid,
                           tf.ones_like(tau * lambda3),
                           tf.ones_like(tau) * lambda3)

        # Tail distribution terms
        term1 = tf.where(l2_zeros_grid,
                         tf.math.log(tau) * tf.ones_like(safe_l2),
                         (tau ** safe_l2 - 1.) / safe_l2)

        term2 = tf.where(l3_zeros_grid,
                         tf.math.log(tf.ones_like(safe_l3) - tau),
                         ((1. - tau) ** safe_l3 - 1.) / safe_l3)

        # Quantile value
        Q = lambda0 + (term1 - term2) * lambda1
        # Q = tf.minimum(tf.maximum(Q, -1e4), 1e4)
        return Q  # [Q, S, N]

    def sample(self, lambda_val: tf.Tensor):
        """
        Samples from the GLD given lambda
        :param lambda_val: realizations of the four latent GPs [..., L]
        :return: Y: sample values
        """
        tau = tf.random.uniform(lambda_val[..., 0].shape)  # [...]
        Q = self.quantile(tau[None, ...], lambda_val)  # [1, ...]
        return Q[0, 0, ..., None]


def create_gld_trajectory(input_dim, lengthscale, seed, tau):

    np.random.seed(seed)
    tf.random.set_seed(seed)

    lower_bounds = [0.] * input_dim
    upper_bounds = [1.] * input_dim
    box = Box(lower_bounds, upper_bounds)

    num_features = 1000
    kernel_with_features = create_kernel_with_features(1., input_dim, num_features, lengthscale)
    kernel = gpflux.helpers.construct_basic_kernel([kernel_with_features] * 4)

    Z = box.sample_sobol(2)

    shared_ip = InducingPoints(Z)
    inducing_variable = SharedIndependentInducingVariables(shared_ip)

    layer = gpflux.layers.GPLayer(kernel, inducing_variable, num_data=1, whiten=True, num_latent_gps=4,
                                  mean_function=gpflow.mean_functions.Constant(np.zeros([1, 4])))

    model = gpflux.models.DeepGP([layer], gpflow.likelihoods.Gaussian())

    trajectory = sample_dgp(model)

    gld = GLD()

    fun = lambda at: gld.sample(trajectory(at)[None, ...])
    quantile_fun = tf.function(lambda at: gld.quantile(tau=tau, lambda_val=trajectory(at)[None, ...])[0, 0, ..., None])

    return fun, quantile_fun


if __name__ == "__main__":

    quantile_levels = np.linspace(0.1, .9, 2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for quantile_level in quantile_levels:
        fun, quantile_fun = create_gld_trajectory(input_dim=2, lengthscale=0.5, seed=2, tau=quantile_level)

        # Create a regular grid on the parameter space
        Xplot, xx, yy = create_grid(mins=[0., 0.], maxs=[1., 1.], grid_density=30)
        qfun = quantile_fun(Xplot).numpy()
        plot_surface(xx, yy, qfun, ax=ax, contour=False, fill=False, alpha=0.5)

    nrow = 4
    ncol = 4
    fig, ax = plt.subplots(nrow, ncol, squeeze=False, sharex="all", sharey="all")

    bins = np.linspace(-6., 6., 20)

    for i in range(nrow):
        for j in range(ncol):
            x = np.random.uniform(0., 1., 2).reshape(-1, 1)
            xx = np.repeat(x, 10000).reshape(10000, 2)
            f = fun(xx).numpy()
            ax[i, j].hist(f, bins)


