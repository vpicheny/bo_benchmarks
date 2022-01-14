import itertools
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import trieste
from trieste.objectives import branin, BRANIN_SEARCH_SPACE
from trieste.models.gpflow.models import GaussianProcessRegression
# from ...trieste.notebooks.util.plotting import plot_bo_points, plot_function_2d
from plotting import plot_function_2d
import gpflow
from trieste.acquisition import AcquisitionFunction, SingleModelAcquisitionBuilder, AcquisitionFunctionClass, \
    GIBBON, SingleModelGreedyAcquisitionBuilder, gibbon_repulsion_term, PenalizationFunction
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.models import ProbabilisticModel
from trieste.utils import DEFAULTS
from typing import Optional, cast
from trieste.data import Dataset
from trieste.types import TensorType
from trieste.space import SearchSpace


class feasGIBBON(SingleModelGreedyAcquisitionBuilder[ProbabilisticModel]):

    def __init__(
        self,
        search_space: SearchSpace,
        num_samples: int = 5,
        grid_size: int = 1000,
        rescaled_repulsion: bool = True,
    ):

        tf.debugging.assert_positive(num_samples)
        tf.debugging.assert_positive(grid_size)

        self._search_space = search_space
        self._num_samples = num_samples
        self._grid_size = grid_size
        self._rescaled_repulsion = rescaled_repulsion

        self._min_value_samples: Optional[TensorType] = None
        self._quality_term: Optional[feasibility_quality_term] = None
        self._diversity_term: Optional[gibbon_repulsion_term] = None
        self._gibbon_acquisition: Optional[AcquisitionFunction] = None

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:

        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")

        acq = self._update_quality_term(dataset, model)
        if pending_points is not None and len(pending_points) != 0:
            acq = self._update_repulsion_term(acq, dataset, model, pending_points)

        return acq

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
        pending_points: Optional[TensorType] = None,
        new_optimization_step: bool = True,
    ) -> AcquisitionFunction:

        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(self._quality_term is not None, [])

        if new_optimization_step:
            self._update_quality_term(dataset, model)

        if pending_points is None:
            # no repulsion term required if no pending_points.
            return cast(AcquisitionFunction, self._quality_term)

        return self._update_repulsion_term(function, dataset, model, pending_points)

    def _update_repulsion_term(
        self,
        function: Optional[AcquisitionFunction],
        dataset: Dataset,
        model: ProbabilisticModel,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        tf.debugging.assert_rank(pending_points, 2)

        if self._gibbon_acquisition is not None and isinstance(
            self._diversity_term, gibbon_repulsion_term
        ):
            # if possible, just update the repulsion term
            self._diversity_term.update(pending_points)
            return self._gibbon_acquisition
        else:
            # otherwise construct a new repulsion term and acquisition function
            self._diversity_term = gibbon_repulsion_term(
                model, pending_points, rescaled_repulsion=self._rescaled_repulsion
            )

            @tf.function
            def gibbon_acquisition(x: TensorType) -> TensorType:
                return cast(PenalizationFunction, self._diversity_term)(x) + cast(
                    AcquisitionFunction, self._quality_term
                )(x)

            self._gibbon_acquisition = gibbon_acquisition
            return gibbon_acquisition

    def _update_quality_term(
        self, dataset: Dataset, model: ProbabilisticModel
    ) -> AcquisitionFunction:
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")

        if self._quality_term is not None:  # if possible, just update the quality term
            self._quality_term.update(self._min_value_samples)
        else:  # otherwise build quality term
            self._quality_term = feasibility_quality_term(model)
        return cast(AcquisitionFunction, self._quality_term)


def value_function(mean, var, threshold):
  normal = tfp.distributions.Normal(tf.cast(0, mean.dtype), tf.cast(1, mean.dtype))
  threshold = tf.cast(threshold, mean.dtype)
  t = (mean - threshold) / tf.sqrt(var)
  p = normal.cdf(t)
  return - p * tf.math.log(p) - (1. - p) * tf.math.log(1. - p) # (p * (1. - p))


class feasibility_quality_term(AcquisitionFunctionClass):
    def __init__(self, model: ProbabilisticModel):

        if not hasattr(model, "covariance_between_points"):
            raise AttributeError(
                """
                GIBBON only supports models with a covariance_between_points method.
                """
            )

        self._model = model
        # self._samples = tf.Variable(samples)

    def update(self, samples: TensorType) -> None:
        pass


    @tf.function
    def __call__(self, x: TensorType) -> TensorType:  # [N, D] -> [N, 1]
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        fmean, fvar = self._model.predict(tf.squeeze(x, -2))
        return value_function(fmean, fvar, threshold)  # [N, 1]


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
        # return - p * tf.math.log(p)
        return p
        # return (p * (1. - p) )

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

acq = feasGIBBON(search_space=search_space, rescaled_repulsion=True)
rule = EfficientGlobalOptimization(builder=acq, num_query_points=250)
x1 = rule.acquire_single(search_space, model, dataset=initial_data)

ax = plot_excursion_probability(
    "Excursion probability entropy, initial data (red), query points (blue)",
    model,
)
ax[0,0].scatter(x1[:, 0], x1[:, 1], color = "blue")
ax[0,0].scatter(initial_query_points[:, 0], initial_query_points[:, 1], color="red")