from __future__ import annotations

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
import gpflux
import trieste.models.gpflow

from tensorflow_probability.python.distributions.laplace import Laplace

from gpflow.inducing_variables import InducingPoints
from gpflow.config import default_float
from gpflow.inducing_variables import InducingVariables, SharedIndependentInducingVariables
from gpflow.kernels import SeparateIndependent
from gpflow.models import VGP

from gpflux.layers import LatentVariableLayer
from gpflux.models import DeepGP
from gpflux.models.deep_gp import sample_dgp
from gpflux.sampling.kernel_with_feature_decomposition import KernelWithFeatureDecomposition
from gpflux.sampling.sample import efficient_sample, Sample
from gpflux.layers.basis_functions.fourier_features import RandomFourierFeaturesCosine
from gpflux.helpers import construct_basic_inducing_variables

from trieste.data import Dataset
from trieste.models.gpflux.models import DeepGaussianProcess
from trieste.types import TensorType
from trieste.models.gpflow import VariationalGaussianProcess
from trieste.models.optimizer import Optimizer, BatchOptimizer

from typing import Callable, Dict, Any, Optional
from inducing_point_selector import InducingPointSelector, KMeans

tf.keras.backend.set_floatx("float64")


def build_model(data, CONFIG, search_space):
    if CONFIG.model == "quantile":
        return build_hetgp_rff_model(data=data,
                                     num_features=CONFIG.num_features,
                                     likelihood_distribution=
                                     lambda loc, scale: ASymmetricLaplace(loc, scale, tau=CONFIG.problem.quantile_level),
                                     num_inducing_points=CONFIG.num_inducing_points,
                                     inducing_point_selector=KMeans(search_space))
    elif CONFIG.model == "hetgp":
        return build_hetgp_rff_model(data=data,
                                     num_features=CONFIG.num_features,
                                     likelihood_distribution=tfp.distributions.Normal,
                                     num_inducing_points=CONFIG.num_inducing_points,
                                     inducing_point_selector=KMeans(search_space))
    elif CONFIG.model == "GPR":
        return build_quantile_gpr_model(data,
                                        batch_size=CONFIG.batch_size,
                                        quantile_level=CONFIG.problem.quantile_level)
    else:
        raise NotImplementedError


@efficient_sample.register(
    SharedIndependentInducingVariables,
    SeparateIndependent,
    object
)
def _efficient_sample_matheron_rule(
    inducing_variable: InducingVariables,
    kernel: KernelWithFeatureDecomposition,
    q_mu: tf.Tensor,
    *,
    q_sqrt: Optional[TensorType] = None,
    whiten: bool = False,
) -> Sample:
    samples = []
    for i, k in enumerate(kernel.kernels):
        samples.append(efficient_sample(inducing_variable.inducing_variable, k, q_mu[..., i:(i+1)],
                                        q_sqrt=q_sqrt[i:(i+1), ...], whiten=whiten))

    class MultiOutputSample(Sample):
        def __call__(self, X: TensorType) -> tf.Tensor:
            return tf.concat([s(X) for s in samples], axis=-1)
    return MultiOutputSample()


class ASymmetricLaplace(Laplace):
    def __init__(self,
                 loc,
                 scale,
                 tau,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='Laplace'):

        super().__init__(loc, scale, validate_args, allow_nan_stats, name)

        # with tf.name_scope(name) as name:
        #     dtype = dtype_util.common_dtype([loc, scale], tf.float32)
        #     self.tau = tensor_util.convert_nonref_to_tensor(
        #     tau, name='tau', dtype=dtype)

        self.tau = tau

    # @property
    # def tau(self):
    #     return self._tau

    def _mean(self):
        loc = tf.convert_to_tensor(self.loc)
        return tf.broadcast_to(loc + self.scale * (1. - 2 * self.tau) / (self.tau * (1. - self.tau)),
                               self._batch_shape_tensor(loc=loc))

    def _stddev(self):
        scale = tf.convert_to_tensor(self.scale)
        return tf.broadcast_to(scale * np.sqrt(1. - 2. * self.tau + 2. * self.tau**2) /
                                     (self.tau * (1. - self.tau)),
                               self._batch_shape_tensor(scale=scale))

    def _variance(self):
        scale = tf.convert_to_tensor(self.scale)
        return tf.broadcast_to(scale**2 * (1. - 2. * self.tau + 2. * self.tau**2) /
                                     (self.tau**2 * (1. - self.tau)**2),
                               self._batch_shape_tensor(scale=scale))

    def _log_prob(self, x):
        loc = tf.convert_to_tensor(self.loc)
        scale = tf.convert_to_tensor(self.scale)
        z = (x - loc) / scale
        is_neg = 0.5 - 0.5 * tf.sign(z)
        return tf.math.log(self.tau * (1 - self.tau) / scale) - z * (self.tau - is_neg)

    def _cdf(self, x):
        z = self._z(x)
        is_neg = 0.5 - 0.5 * tf.sign(z)
        negF = self.tau * tf.exp((1. - self.tau) / self.scale * (x - self.loc))
        posF = 1. - (1 - self.tau) + tf.exp(-self.tau/self.scale * (x - self.loc))
        return tf.where(is_neg > 0.5, negF, posF)

    def _log_cdf(self, x):
        return tf.math.log(self._cdf(x))

    def _quantile(self, p):
        loc = tf.convert_to_tensor(self.loc)
        scale = tf.convert_to_tensor(self.scale)
        q1 = loc + scale / (1. - self.tau) * tf.math.log(p / self.tau)
        q2 = loc - scale / self.tau * tf.math.log((1. - p) / (1. - self.tau))
        return tf.where(p > self.tau, q1, q2)

    def _z(self, x):
        return (x - self.loc) / self.scale

    def _sample_n(self, n, seed=None):
        return NotImplementedError

    def _log_survival_function(self, x):
        return NotImplementedError

    def _entropy(self):
        return NotImplementedError

    def _median(self):
        return NotImplementedError

    def _mode(self):
        return NotImplementedError


class FeaturedHetGPFluxModel(DeepGaussianProcess):

    def __init__(self,
                 model: DeepGP,
                 optimizer: tf.optimizers.Optimizer | None = None,
                 fit_args: Dict[str, Any] | None = None,
                 inducing_point_selector: InducingPointSelector = None,
                 ):

        super().__init__(model, optimizer)

        if inducing_point_selector is None:
            inducing_point_selector = KMeans
        self._inducing_point_selector = inducing_point_selector

    def sample_trajectory(self) -> Callable:
        return sample_dgp(self.model_gpflux)

    def update(self, dataset: Dataset) -> None:
        inputs = dataset.query_points
        new_num_data = inputs.shape[0]
        self.model_gpflux.num_data = new_num_data

        # Update num_data for each layer, as well as make sure dataset shapes are ok
        for i, layer in enumerate(self.model_gpflux.f_layers):
            if hasattr(layer, "num_data"):
                layer.num_data = new_num_data

            if isinstance(layer, LatentVariableLayer):
                inputs = layer(inputs)
                continue

            if isinstance(layer.inducing_variable, InducingPoints):
                inducing_variable = layer.inducing_variable
            else:
                inducing_variable = layer.inducing_variable.inducing_variable

            if inputs.shape[-1] != inducing_variable.Z.shape[-1]:
                raise ValueError(
                    f"Shape {inputs.shape} of input to layer {layer} is incompatible with shape"
                    f" {inducing_variable.Z.shape} of that layer. Trailing dimensions must match."
                )
            inputs = layer(inputs)

            if hasattr(layer.kernel, 'feature_functions'): # If using RFF kernel decomp then need to resample for new kernel params
                feature_function = layer.kernel.feature_functions
                input_shape = dataset.query_points.shape
                def renew_rff(feature_f, input_dim):
                    shape_bias = [1, feature_f.output_dim]
                    new_b = feature_f._sample_bias(shape_bias, dtype=feature_f.dtype)
                    feature_f.b = new_b
                    shape_weights = [feature_f.output_dim, input_dim]
                    new_W = feature_f._sample_weights(shape_weights, dtype=feature_f.dtype)
                    feature_f.W = new_W
                renew_rff(feature_function,  input_shape[-1])

            num_inducing = layer.inducing_variable.inducing_variable.Z.shape[0]

            Z = self._inducing_point_selector.get_points(dataset.query_points,
                                                         dataset.observations,
                                                         num_inducing,
                                                         layer.kernel,
                                                         noise=1e-6)

            jitter = 1e-6

            if layer.whiten:
                f_mu, f_cov = self.predict_joint(Z)  # [N, L], [L, N, N]
                Knn = layer.kernel(Z, full_cov=True)  # [N, N]
                jitter_mat = jitter * tf.eye(num_inducing, dtype=Knn.dtype)
                Lnn = tf.linalg.cholesky(Knn + jitter_mat)  # [N, N]
                new_q_mu = tf.linalg.triangular_solve(Lnn, f_mu)  # [N, L]
                tmp = tf.linalg.triangular_solve(Lnn[None], f_cov)  # [L, N, N], L⁻¹ f_cov
                S_v = tf.linalg.triangular_solve(Lnn[None], tf.linalg.matrix_transpose(tmp))  # [L, N, N]
                new_q_sqrt = tf.linalg.cholesky(S_v + jitter_mat)  # [L, N, N]
            else:
                new_q_mu, new_f_cov = layer.predict(Z, full_cov=True)  # [N, L], [L, N, N]
                jitter_mat = jitter * tf.eye(num_inducing, dtype=new_f_cov.dtype)
                new_q_sqrt = tf.linalg.cholesky(new_f_cov + jitter_mat)

            layer.q_mu.assign(new_q_mu)
            layer.q_sqrt.assign(new_q_sqrt)
            layer.inducing_variable.inducing_variable.Z.assign(Z)



def create_kernel_with_features(var, input_dim, num_features):
    kernel = set_kernel(var, input_dim)
    coefficients = np.ones((num_features, 1), dtype=default_float())
    features = RandomFourierFeaturesCosine(kernel, num_features, dtype=default_float())
    return KernelWithFeatureDecomposition(kernel, features, coefficients)

def build_hetgp_rff_model(data, num_features, likelihood_distribution, num_inducing_points,
                          inducing_point_selector):
    num_data, input_dim = data.query_points.shape
    var = tf.math.reduce_variance(data.observations)
    kernel_with_features1 = create_kernel_with_features(var / 2., input_dim, num_features)
    kernel_with_features2 = create_kernel_with_features(var / 2., input_dim, num_features)
    kernel_list = [kernel_with_features1, kernel_with_features2]
    kernel = gpflux.helpers.construct_basic_kernel(kernel_list)

    Z = inducing_point_selector.get_points(X=data.query_points, Y=data.observations,
                                           M=num_inducing_points, kernel=kernel, noise=1e-6)
    inducing_variable = construct_basic_inducing_variables(num_inducing_points, input_dim,
                                                           output_dim=2, share_variables=True, z_init= Z)
    gpflow.utilities.set_trainable(inducing_variable, False)

    layer = gpflux.layers.GPLayer(kernel, inducing_variable, num_data, whiten=False, num_latent_gps=2,
                                  mean_function=gpflow.mean_functions.Constant(np.zeros([1, 2])))

    likelihood = gpflow.likelihoods.HeteroskedasticTFPConditional(
        distribution_class=likelihood_distribution,
        scale_transform=tfp.bijectors.Exp(),
    )

    likelihood_layer = gpflux.layers.LikelihoodLayer(likelihood)
    model = gpflux.models.DeepGP([layer], likelihood_layer)

    epochs = 5000
    batch_size = 200

    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", patience=10, factor=0.5, verbose=1, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor="loss", patience=50, min_delta=0.01, verbose=0, mode="min"),]

    fit_args = {
        "batch_size": batch_size,
        "epochs": epochs,
        "verbose": 0,
        "callbacks": callbacks,
    }
    optimizer = Optimizer(tf.optimizers.Adam(0.01), fit_args)
    return FeaturedHetGPFluxModel(model=model, optimizer=optimizer, fit_args=fit_args,
                                  inducing_point_selector=inducing_point_selector)

from trieste.utils import DEFAULTS, jit
from trieste.models.gpflow.utils import assert_data_is_compatible


class QuantileVGP(VariationalGaussianProcess):
    def __init__(
        self,
        model: VGP,
        optimizer: Optimizer | None = None,
        use_natgrads: bool = False,
        natgrad_gamma: Optional[float] = None,
        quantile_level: Optional[float] = 0.5,
        batch_size: Optional[float] = 1,
    ):
        super().__init__(model, optimizer, use_natgrads, natgrad_gamma)
        self.quantile_level = quantile_level
        self.batch_size = batch_size

    def update(self, dataset: Dataset, *, jitter: float = DEFAULTS.JITTER) -> None:
        """
        Update the model given the specified ``dataset``. Does not train the model.

        :param dataset: The data with which to update the model.
        :param jitter: The size of the jitter to use when stabilizing the Cholesky decomposition of
            the covariance matrix.
        """
        model = self.model

        x, y = self.model.data[0].value(), self.model.data[1].value()
        new_data = aggregate_data(dataset, self.batch_size, self.quantile_level)
        assert_data_is_compatible(new_data, Dataset(x, y))

        f_mu, f_cov = self.model.predict_f(new_data.query_points, full_cov=True)  # [N, L], [L, N, N]

        # GPflow's VGP model is hard-coded to use the whitened representation, i.e.
        # q_mu and q_sqrt parametrise q(v), and u = f(X) = L v, where L = cholesky(K(X, X))
        # Hence we need to back-transform from f_mu and f_cov to obtain the updated
        # new_q_mu and new_q_sqrt:
        Knn = model.kernel(new_data.query_points, full_cov=True)  # [N, N]
        jitter_mat = jitter * tf.eye(len(new_data), dtype=Knn.dtype)
        Lnn = tf.linalg.cholesky(Knn + jitter_mat)  # [N, N]
        new_q_mu = tf.linalg.triangular_solve(Lnn, f_mu)  # [N, L]
        tmp = tf.linalg.triangular_solve(Lnn[None], f_cov)  # [L, N, N], L⁻¹ f_cov
        S_v = tf.linalg.triangular_solve(Lnn[None], tf.linalg.matrix_transpose(tmp))  # [L, N, N]
        new_q_sqrt = tf.linalg.cholesky(S_v + jitter_mat)  # [L, N, N]

        model.data[0].assign(new_data.query_points)
        model.data[1].assign(new_data.observations)
        model.num_data = len(new_data)
        model.q_mu = gpflow.Parameter(new_q_mu)
        model.q_sqrt = gpflow.Parameter(new_q_sqrt, transform=gpflow.utilities.triangular())

    def optimize(self, dataset: Dataset) -> None:
        """
        :class:`VariationalGaussianProcess` has a custom `optimize` method that (optionally) permits
        alternating between standard optimization steps (for kernel parameters) and natural gradient
        steps for the variational parameters (`q_mu` and `q_sqrt`). See :cite:`salimbeni2018natural`
        for details. Using natural gradients can dramatically speed up model fitting, especially for
        ill-conditioned posteriors.

        If using natural gradients, our optimizer inherits the mini-batch behavior and number
        of optimization steps as the base optimizer specified when initializing
        the :class:`VariationalGaussianProcess`.
        """
        model = self.model

        if self._use_natgrads:  # optimize variational params with natgrad optimizer

            natgrad_optimizer = gpflow.optimizers.NaturalGradient(gamma=self._natgrad_gamma)
            base_optimizer = self.optimizer

            gpflow.set_trainable(model.q_mu, False)  # variational params optimized by natgrad
            gpflow.set_trainable(model.q_sqrt, False)
            variational_params = [(model.q_mu, model.q_sqrt)]
            model_params = model.trainable_variables

            loss_fn = base_optimizer.create_loss(model, dataset)

            @jit(apply=self.optimizer.compile)
            def perform_optimization_step() -> None:  # alternate with natgrad optimizations
                natgrad_optimizer.minimize(loss_fn, variational_params)
                base_optimizer.optimizer.minimize(
                    loss_fn, model_params, **base_optimizer.minimize_args
                )

            for _ in range(base_optimizer.max_iter):  # type: ignore
                perform_optimization_step()

            gpflow.set_trainable(model.q_mu, True)  # revert varitional params to trainable
            gpflow.set_trainable(model.q_sqrt, True)

        else:
            self.optimizer.optimize(model, dataset)


def build_quantile_gpr_model(data, batch_size, quantile_level):
    var = tf.math.reduce_variance(data.observations)
    kernel = set_kernel(var, data.query_points.shape[1])
    meanf = gpflow.mean_functions.Constant()
    aggregated_data = aggregate_data(data, batch_size, quantile_level)
    lik = HeteroskedasticGaussian()
    model = gpflow.models.VGP(aggregated_data.astuple(), kernel=kernel,
                              likelihood=lik, num_latent_gps=1, mean_function=meanf)
    return QuantileVGP(model=model, use_natgrads=True, batch_size=batch_size, quantile_level=quantile_level,
                                     optimizer=BatchOptimizer(tf.optimizers.Adam(), batch_size=100))


def set_kernel(var, input_dim):
    kernel = gpflow.kernels.Matern52(variance=var, lengthscales=0.2 * np.ones(input_dim, ))
    prior_scale = tf.cast(1.0, dtype=tf.float64)
    kernel.variance.prior = tfp.distributions.LogNormal(tf.cast(0.0, dtype=tf.float64), prior_scale)
    kernel.lengthscales.prior = tfp.distributions.LogNormal(tf.math.log(kernel.lengthscales), prior_scale)
    return kernel

def aggregate_data(data, batch_size, quantile_level):
    unique_query_points = unique_points_2d(data.query_points)  # [N, D] -> [M, D]
    observations_by_replicates = tf.reshape(data.observations,
                                            [unique_query_points.shape[0], batch_size, data.observations.shape[-1]]) # [N, L] -> [M, B, L]
    quantiles = tfp.stats.percentile(observations_by_replicates, quantile_level * 100, axis=1)  # [M, L]
    variances = get_variance_by_bootstrap(observations_by_replicates, quantile_level)  # [M, L]
    return Dataset(unique_query_points, tf.concat([quantiles, variances], axis=-1))

def get_variance_by_bootstrap(observations, quantile_level, boot_sample_size=100):
    # observations comes in [M, B, L]
    ind = np.random.choice(observations.shape[1], [boot_sample_size, observations.shape[1]])  # [boot, B]
    bootstrapped_data = tf.gather(observations, ind, axis=1)  # [M, B, L] -> [M, boot, B, L]
    bootstrapped_quantiles = tfp.stats.percentile(bootstrapped_data, quantile_level * 100, axis=2)  # [B, M, L]
    var = tf.math.reduce_variance(bootstrapped_quantiles, axis=1)  # [M, L]
    return var


class HeteroskedasticGaussian(gpflow.likelihoods.Likelihood):
    def __init__(self, **kwargs):
        # this likelihood expects a single latent function F, and two columns in the data matrix Y:
        super().__init__(latent_dim=1, observation_dim=2)

    def _log_prob(self, F, Y):
        # log_prob is used by the quadrature fallback of variational_expectations and predict_log_density.
        # Because variational_expectations is implemented analytically below, this is not actually needed,
        # but is included for pedagogical purposes.
        # Note that currently relying on the quadrature would fail due to https://github.com/GPflow/GPflow/issues/966
        Y, NoiseVar = Y[:, 0], Y[:, 1]
        return gpflow.logdensities.gaussian(Y, F, NoiseVar)

    def _variational_expectations(self, Fmu, Fvar, Y):
        Y, NoiseVar = Y[:, 0], Y[:, 1]
        Fmu, Fvar = Fmu[:, 0], Fvar[:, 0]
        return (
            -0.5 * np.log(2 * np.pi)
            - 0.5 * tf.math.log(NoiseVar)
            - 0.5 * (tf.math.square(Y - Fmu) + Fvar) / NoiseVar
        )

    # The following two methods are abstract in the base class.
    # They need to be implemented even if not used.

    def _predict_log_density(self, Fmu, Fvar, Y):
        raise NotImplementedError

    def _predict_mean_and_var(self, Fmu, Fvar):
        raise NotImplementedError

#
# quantile_level = 0.9
# beta = tfp.distributions.Normal(loc=0., scale=1.).quantile(value=0.9).numpy()
#

def unique_points_2d(points):
    new_points = points[0:1, :]
    for i in range(points.shape[0] - 1):
        is_point_present = tf.reduce_all(tf.equal(points[(i+1):(i+2), :], new_points), axis=1)
        if not tf.reduce_any(is_point_present):
            new_points = tf.concat([new_points, points[(i+1):(i+2), :]], axis=0)
    return new_points
