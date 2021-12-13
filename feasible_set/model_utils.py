import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from trieste.models.gpflow.models import GaussianProcessRegression


def set_kernel(var, input_dim):
    kernel = gpflow.kernels.Matern52(variance=var, lengthscales=0.2 * np.ones(input_dim, ))
    prior_scale = tf.cast(1.0, dtype=tf.float64)
    kernel.variance.prior = tfp.distributions.LogNormal(tf.cast(0.0, dtype=tf.float64), prior_scale)
    kernel.lengthscales.prior = tfp.distributions.LogNormal(tf.math.log(kernel.lengthscales), prior_scale)
    return kernel


def build_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = set_kernel(variance, data.query_points.shape[-1])

    gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
    gpflow.set_trainable(gpr.likelihood, False)

    return GaussianProcessRegression(gpr)