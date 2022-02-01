from quantile.problems import Problem
import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from trieste.models.gpflow.models import GaussianProcessRegression
from trieste.space import Box
from trieste.models.gpflux import DeepGaussianProcess

import gpflux
from gpflux.layers import LatentVariableLayer
from gpflux.models import DeepGP
from gpflux.models.deep_gp import sample_dgp
from gpflux.sampling.kernel_with_feature_decomposition import KernelWithFeatureDecomposition
from gpflux.sampling.sample import efficient_sample, Sample
from gpflux.layers.basis_functions.fourier_features import RandomFourierFeaturesCosine
from gpflux.helpers import construct_basic_inducing_variables
from gpflow.inducing_variables import (
    InducingPoints,
    SharedIndependentInducingVariables,
)

from quantile.model_utils import create_kernel_with_features


from typing import Callable, Optional
from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
from gpflow.config import default_float

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

        lambda0 = lambda_val[None, :, :, 0]
        lambda1 = self.softplus_function(lambda_val[None, :, :, 1])
        lambda2 = lambda_val[None, :, :, 2] * .1
        lambda3 = lambda_val[None, :, :, 3] * .1

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
        Q = tf.minimum(tf.maximum(Q, -1e4), 1e4)
        return Q  # [Q, S, N]

    def sample(self, lambda_val: tf.Tensor):
        """
        Samples from the GLD given lambda
        :param lambda_val: realizations of the four latent GPs [..., L]
        :return: Y: sample values
        """
        tau = tf.random.uniform(lambda_val[..., 0].shape)  # [...]
        Q = self.quantile(tau[None, ...], lambda_val)  # [1, ...]
        return Q[0, ...]


lower_bounds = [0.] * 4
upper_bounds = [1.] * 4
box = Box(lower_bounds, upper_bounds)

input_dim = 4
num_features = 1000
kernel_with_features = create_kernel_with_features(1., input_dim, num_features)
kernel = gpflux.helpers.construct_basic_kernel([kernel_with_features] * 4)

Z = box.sample_sobol(10)

shared_ip = InducingPoints(Z)
inducing_variable = SharedIndependentInducingVariables(shared_ip)

layer = gpflux.layers.GPLayer(kernel, inducing_variable, num_data=1, whiten=True, num_latent_gps=4,
                              mean_function=gpflow.mean_functions.Constant(np.zeros([1, 4])))

model = gpflux.models.DeepGP([layer], gpflow.likelihoods.Gaussian())

trajectory = sample_dgp(model)

gp_val = trajectory(box.sample_sobol(3))

gld = GLD()
gld.sample(gp_val[None, ...])
gld.quantile(tau = 0.9, lambda_val=gp_val[None, ...])

