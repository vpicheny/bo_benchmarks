import itertools
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import trieste
from trieste.objectives import branin, BRANIN_SEARCH_SPACE
from trieste.models.gpflow.models import GaussianProcessRegression
from docs.notebooks.util.plotting import plot_bo_points, plot_function_2d
import gpflow
from trieste.acquisition import SingleModelAcquisitionBuilder, AcquisitionFunction
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.models import ProbabilisticModel
from trieste.utils import DEFAULTS
from typing import Optional
from trieste.data import Dataset
from trieste.types import TensorType


class WeightedPredictiveVariance(SingleModelAcquisitionBuilder[ProbabilisticModel]):

  def __init__(self, threshold: float) -> None:
    super().__init__()

    self._threshold = threshold

  def __repr__(self) -> str:
    """"""
    return f"WeightedPredictiveVariance(threshold={self._threshold!r})"

  def prepare_acquisition_function(
          self,
          model: ProbabilisticModel,
          dataset: Optional[Dataset] = None,
  ) -> AcquisitionFunction:

    return weighted_predictive_variance(model, self._threshold)

  def update_acquisition_function(
          self,
          function: AcquisitionFunction,
          model: ProbabilisticModel,
          dataset: Optional[Dataset] = None,
  ) -> AcquisitionFunction:
    return function  # no need to update anything


def weighted_predictive_variance(model: ProbabilisticModel, threshold: float) -> TensorType:
  @tf.function
  def acquisition(x: TensorType) -> TensorType:
    mean, var = model.predict(x)
    val = value_function(mean, var, threshold)
    return tf.squeeze(val**2 * var, -2)
  return acquisition


class WeightedPredictiveVariance2(SingleModelAcquisitionBuilder[ProbabilisticModel]):

  def __init__(self, threshold: float, L11: TensorType, x1: TensorType) -> None:
    super().__init__()

    self._threshold = threshold
    self._L11 = L11
    self._x1 = x1

  def prepare_acquisition_function(
          self,
          model: ProbabilisticModel,
          dataset: Optional[Dataset] = None,
  ) -> AcquisitionFunction:

    return weighted_predictive_variance2(model, self._threshold, self._L11, self._x1)

  def update_acquisition_function(
          self,
          function: AcquisitionFunction,
          model: ProbabilisticModel,
          dataset: Optional[Dataset] = None,
  ) -> AcquisitionFunction:
    return function  # no need to update anything


def weighted_predictive_variance2(model: ProbabilisticModel, threshold: float, L11: TensorType, x1: TensorType) -> TensorType:
  @tf.function
  def acquisition(x: TensorType) -> TensorType:
    mean, var = model.predict(tf.squeeze(x, -2))
    cov = model.covariance_between_points(tf.squeeze(x, -2), x1)[0, ...]

    val = value_function(mean, var, threshold)
    L12 = cov * val  # tf.reduce_prod(val)
    L22 = var * val**2
    d2 = L22 - L12**2 / L11
    return d2
  return acquisition


class WeightedPredictiveVariance3(SingleModelAcquisitionBuilder[ProbabilisticModel]):

  def __init__(self, threshold: float, V12: TensorType, x12: TensorType, val12:TensorType) -> None:
    super().__init__()

    self._threshold = threshold
    self._V12 = V12
    self._x12 = x12
    self._val12 = val12

  def prepare_acquisition_function(
          self,
          model: ProbabilisticModel,
          dataset: Optional[Dataset] = None,
  ) -> AcquisitionFunction:

    return weighted_predictive_variance3(model, self._threshold, self._V12, self._x12, self._val12)

  def update_acquisition_function(
          self,
          function: AcquisitionFunction,
          model: ProbabilisticModel,
          dataset: Optional[Dataset] = None,
  ) -> AcquisitionFunction:
    return function  # no need to update anything


def weighted_predictive_variance3(model: ProbabilisticModel, threshold: float, V12: TensorType,
                                  x12: TensorType, val12: TensorType) -> TensorType:
  # @tf.function
  def acquisition(x: TensorType) -> TensorType:
    # mean, var = model.predict(tf.concat([tf.squeeze(x, -2), x1], axis=0))
    # cov = model.covariance_between_points(tf.squeeze(x, -2), x1)[0, ...]

    mean, var = model.predict(tf.squeeze(x, -2))
    cov = model.covariance_between_points(tf.squeeze(x, -2), x12)[0, ...]

    val = value_function(mean, var, threshold)

    vv = tf.repeat(val12, x.shape[0], axis=-1)
    L123 = tf.math.multiply(tf.math.multiply(cov, tf.transpose(vv)), tf.repeat(val, 2, axis=-1))  # tf.reduce_prod(val)
    L22 = var * val**2
    c3 = tf.linalg.triangular_solve(V12, tf.transpose(L123))
    d2 = L22 - tf.reduce_sum(c3 ** 2, axis=0)[:, None]
    return d2
  return acquisition

def value_function(mean, var, threshold):
  normal = tfp.distributions.Normal(tf.cast(0, mean.dtype), tf.cast(1, mean.dtype))
  threshold = tf.cast(threshold, mean.dtype)
  t = (mean - threshold) / tf.sqrt(var)
  p = normal.cdf(t)
  return p * (1. - p)


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


def excursion_probability(x, model, threshold=80.):
    mean, variance = model.model.predict_f(x)
    normal = tfp.distributions.Normal(tf.cast(0, x.dtype), tf.cast(1, x.dtype))
    threshold = tf.cast(threshold, x.dtype)
    t = (mean - threshold) / tf.sqrt(variance)
    return normal.cdf(t)

def plot_excursion_probability(title, model, threshold=80.0):

    def objective_function(x):
        p = excursion_probability(x, model, threshold)
        return p * (1. - p)

    _, ax = plot_function_2d(
        objective_function,
        search_space.lower - 0.01,
        search_space.upper + 0.01,
        grid_density=70,
        contour=True,
        colorbar=True,
        figsize=(10, 6),
        title=[title],
        xlabel="$X_1$",
        ylabel="$X_2$",
        fill=True,
    )
    return ax


np.random.seed(1793)
tf.random.set_seed(1793)

search_space = BRANIN_SEARCH_SPACE
threshold = 80.0
observer = trieste.objectives.utils.mk_observer(branin)

num_initial_points = 7
initial_query_points = search_space.sample_halton(num_initial_points)
initial_data = observer(initial_query_points)

# fitting the model only to the initial data
model = build_model(initial_data)
model.optimize(initial_data)

acq = WeightedPredictiveVariance(threshold)
rule = EfficientGlobalOptimization(builder=acq)
x1 = rule.acquire_single(search_space, model, dataset=initial_data)


m1, v1 = model.predict(x1)
val1 = value_function(m1, v1, threshold)
L11 = v1 * val1**2

acq2 = WeightedPredictiveVariance2(threshold, L11=L11, x1=x1)
rule2 = EfficientGlobalOptimization(builder=acq2)
x2 = rule2.acquire_single(search_space, model, dataset=initial_data)

x12 = tf.concat([x1, x2], axis=0)
m12, cov12 = model.predict_joint(x12)
val12 = value_function(m12, tf.transpose(tf.linalg.diag_part(cov12)), threshold)

V12 = tf.linalg.cholesky(tf.repeat(val12, 2, axis=1) * cov12[0] * tf.transpose(tf.repeat(val12, 2, axis=1)))

acq3 = WeightedPredictiveVariance3(threshold, V12=V12, x12=x12, val12=val12)
rule3 = EfficientGlobalOptimization(builder=acq3)
x3 = rule3.acquire_single(search_space, model, dataset=initial_data)


ax = plot_excursion_probability(
    "Probability of excursion, initial data",
    model,
)
ax[0,0].scatter(x1[:, 0], x1[:, 1], color = "blue")
ax[0,0].scatter(x2[:, 0], x2[:, 1], color = "green")
ax[0,0].scatter(x3[:, 0], x3[:, 1], color = "purple")

ax[0,0].scatter(initial_query_points[:, 0], initial_query_points[:, 1], color="red")