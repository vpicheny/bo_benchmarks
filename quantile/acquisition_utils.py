from __future__ import annotations

import numpy as np
import tensorflow as tf
import trieste
import tensorflow_probability as tfp
from scipy.optimize import bisect
from trieste.data import Dataset
from trieste.acquisition.rule import OBJECTIVE, EfficientGlobalOptimization
from trieste.acquisition.interface import SingleModelGreedyAcquisitionBuilder, AcquisitionFunction
from trieste.acquisition.function import ExpectedImprovement, MinValueEntropySearch, min_value_entropy_search, \
    LocalPenalization
from trieste.acquisition.sampler import GumbelSampler
from trieste.types import TensorType
from typing import Optional, cast
from model_utils import FeaturedHetGPFluxModel

tf.keras.backend.set_floatx("float64")


def create_initial_query_points(search_space, CONFIG):
    if CONFIG.model == "GPR":
        query_points = search_space.sample_halton(np.round(CONFIG.num_initial_points / CONFIG.batch_size))
        return query_points
    else:
        return search_space.sample_halton(CONFIG.num_initial_points)


def create_acquisition_rule(CONFIG):
    if CONFIG.model == "quantile":
        # commented out TS for now
        # acq_function = NegativeGaussianProcessTrajectory()
        # return trieste.acquisition.rule.EfficientGlobalOptimization(acq_function.using(OBJECTIVE),num_query_points=CONFIG.batch_size)
        # instead lets do new MES combined with LP
        search_space = trieste.space.Box(CONFIG.problem.lower_bounds, CONFIG.problem.upper_bounds)
        d = tf.shape(CONFIG.problem.lower_bounds)[0]
        acq_function = LocalPenalization(
            search_space,
            num_samples=1_000 * d,  # perhaps too large, might be slow
            base_acquisition_function_builder=MinValueEntropySearchForQuantile(
                search_space,
                quantile_level=CONFIG.problem.quantile_level,
                num_samples=10,
                grid_size=1_000 * d,  # perhaps too large, might be slow
            )
        )
        return trieste.acquisition.rule.EfficientGlobalOptimization(acq_function.using(OBJECTIVE),
                                                                    num_query_points=CONFIG.batch_size)


    elif CONFIG.model == "hetgp":
        hetgp_traj = NegativeQuantilefromGaussianHetGPTrajectory(quantile_level=CONFIG.problem.quantile_level)
        return trieste.acquisition.rule.EfficientGlobalOptimization(hetgp_traj.using(OBJECTIVE),
                                                                    num_query_points=CONFIG.batch_size)
    elif CONFIG.model == "GPR":
        return EfficientGlobalOptimization(myExpectedImprovement().using(OBJECTIVE))
    else:
        raise NotImplementedError


def extract_current_best_quantile(ask_tell, CONFIG):
    model = ask_tell._models[OBJECTIVE]
    data = ask_tell._datasets[OBJECTIVE]
    if CONFIG.model == "quantile":
        mean, var = model.predict(data.query_points)
        return data.query_points[tf.argmin(mean[:, 0]), :][None, :]

    elif CONFIG.model == "hetgp":
        mean, var = model.predict(data.query_points)
        lik_layer = model.model_gpflux.likelihood_layer
        dist = lik_layer.likelihood.conditional_distribution(mean)
        quantile = dist.quantile(value=CONFIG.problem.quantile_level)
        return data.query_points[tf.argmin(quantile)[0], :][None, :]

    elif CONFIG.model == "GPR":
        mean, var = model.predict(data.query_points)
        return data.query_points[tf.argmin(mean, axis=0)[0], :][None, :]


class NegativeGaussianProcessTrajectory(SingleModelGreedyAcquisitionBuilder):
    def __repr__(self) -> str:
        return f"NegativeGaussianProcessTrajectory"

    def prepare_acquisition_function(
            self, model: FeaturedHetGPFluxModel, dataset: Dataset = None,
            pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        trajectory = model.sample_trajectory()
        return lambda at: -trajectory(tf.squeeze(at, axis=1))[..., 0:1]


class MinValueEntropySearchForQuantile(MinValueEntropySearch):

    def __init__(
            self,
            search_space: SearchSpace,
            quantile_level: float,
            num_samples: int = 5,
            grid_size: int = 1000,
    ):
        """
        :param search_space: The global search space over which the optimisation is defined.
        :param num_samples: Number of samples to draw from the distribution over the minimum of the
            objective function.
        :param grid_size: Size of the grid from which to sample the min-values. We recommend
            scaling this with search space dimension.
        :raise tf.errors.InvalidArgumentError: If
            - ``num_samples`` or ``grid_size`` are negative.
        """

        min_value_sampler = QuantileGumbelSampler(sample_min_value=True)  # MIGHT BE WORTH USING THE TRAJECTORIES

        self._quantile_level = quantile_level
        self._min_value_sampler = min_value_sampler
        self._search_space = search_space
        self._num_samples = num_samples
        self._grid_size = grid_size

    def __repr__(self) -> str:
        return f"MinValueEntropySearchForQuantile"

    def prepare_acquisition_function(self, model: FeaturedHetGPFluxModel,
                                     dataset: Optional[Dataset] = None) -> AcquisitionFunction:
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        query_points = self._search_space.sample(num_samples=self._grid_size)
        tf.debugging.assert_same_float_dtype([dataset.query_points, query_points])
        query_points = tf.concat([dataset.query_points, query_points], 0)
        min_value_samples = self._min_value_sampler.sample(model, self._num_samples, query_points)
        return min_value_entropy_search_for_quantile(model, min_value_samples, self._quantile_level)

    def update_acquisition_function(
            self, function: AcquisitionFunction, model: FeaturedHetGPFluxModel,
            dataset: Optional[Dataset] = None) -> AcquisitionFunction:
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(isinstance(function, min_value_entropy_search_for_quantile), [])

        query_points = self._search_space.sample(num_samples=self._grid_size)
        tf.debugging.assert_same_float_dtype([dataset.query_points, query_points])
        query_points = tf.concat([dataset.query_points, query_points], 0)
        min_value_samples = self._min_value_sampler.sample(model, self._num_samples, query_points)
        function.update(min_value_samples)  # type: ignore
        return function


class QuantileGumbelSampler(GumbelSampler):

    def sample(self, model: ProbabilisticModel, sample_size: int, at: TensorType) -> TensorType:
        """
        Return approximate samples from of the objective function's minimum value.
        :param model: The model to sample from.
        :param sample_size: The desired number of samples.
        :param at: Points at where to fit the Gumbel distribution, with shape `[N, D]`, for points
            of dimension `D`. We recommend scaling `N` with search space dimension.
        :return: The samples, of shape `[S, 1]`, where `S` is the `sample_size`.
        :raise ValueError: If ``at`` has an invalid shape or if ``sample_size`` is not positive.
        """
        tf.debugging.assert_positive(sample_size)
        tf.debugging.assert_shapes([(at, ["N", None])])

        fmean, fvar = model.predict(at)
        fmean = fmean[:, 0:1]  # only need posterior for f to get samples of f*
        fvar = fvar[:, 0:1]
        fsd = tf.math.sqrt(fvar)

        def probf(y: tf.Tensor) -> tf.Tensor:  # Build empirical CDF for Pr(y*^hat<y)
            unit_normal = tfp.distributions.Normal(tf.cast(0, fmean.dtype), tf.cast(1, fmean.dtype))
            log_cdf = unit_normal.log_cdf(-(y - fmean) / fsd)
            return 1 - tf.exp(tf.reduce_sum(log_cdf, axis=0))

        left = tf.reduce_min(fmean - 5 * fsd)
        right = tf.reduce_max(fmean + 5 * fsd)

        def binary_search(val: float) -> float:  # Find empirical interquartile range
            return bisect(lambda y: probf(y) - val, left, right, maxiter=10000)

        q1, q2 = map(binary_search, [0.25, 0.75])

        log = tf.math.log
        l1 = log(log(4.0 / 3.0))
        l2 = log(log(4.0))
        b = (q1 - q2) / (l1 - l2)
        a = (q2 * l1 - q1 * l2) / (l1 - l2)

        uniform_samples = tf.random.uniform([sample_size], dtype=fmean.dtype)
        gumbel_samples = log(-log(1 - uniform_samples)) * tf.cast(b, fmean.dtype) + tf.cast(
            a, fmean.dtype
        )
        gumbel_samples = tf.expand_dims(gumbel_samples, axis=-1)  # [S, 1]

        return gumbel_samples


class min_value_entropy_search_for_quantile(min_value_entropy_search):
    def __init__(self, model: ProbabilisticModel, samples: TensorType, quantile_level: float):
        r"""
        Return the max-value entropy search acquisition function adapted for quantile models.

        This function calculates the information gain (or
        change in entropy) in the distribution over the objective minimum :math:`y^*`, if we were
        to evaluate the objective at a given point.

        :param model: The model of the objective function.
        :param samples: Samples from the distribution over :math:`y^*`.
        :return: The max-value entropy search acquisition function modified for objective
            minimisation. This function will raise :exc:`ValueError` or
            :exc:`~tf.errors.InvalidArgumentError` if used with a batch size greater than one.
        :raise ValueError or tf.errors.InvalidArgumentError: If ``samples`` has rank less than two,
            or is empty.
        """
        tf.debugging.assert_rank(samples, 2)
        tf.debugging.assert_positive(len(samples))

        self._model = model
        self._samples = tf.Variable(samples)
        self._quantile_level = quantile_level

    def update(self, samples: TensorType) -> None:
        """Update the acquisition function with new samples."""
        tf.debugging.assert_rank(samples, 2)
        tf.debugging.assert_positive(len(samples))
        self._samples.assign(samples)

    # @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )

        tau = self._quantile_level

        combined_mean, combined_var = self._model.model_gpflux.predict_f(x)
        f_mean = combined_mean[:, :, 0]  # [N, 1]
        f_var = combined_var[:, :, 0]  # [N, 1]
        g_mean = combined_mean[:, :, 1]  # [N, 1]
        g_var = combined_var[:, :, 1]  # [N, 1]
        f_sd = tf.clip_by_value(
            tf.math.sqrt(f_var), 1e-10, f_mean.dtype.max
        )  # clip below to improve numerical stability

        # entropy reduction is  H(y) - E_{f*}[H(y|f>f*)]
        # so we need H(y) and H(y| f>f*)
        # lets do MM approx H(y)=0.5log(2*pi*e*Var(y)) and H(y|f>f*)=0.5log(2*pi*e*Var(y|f>f*))

        # first calc Var(y) = Var_{q,sigma}(E[y|q,sigma]) + E{q,sigma}[Var(y|q, sigma)]
        # where sigma = e^g and q=f (for gaussian f and g)
        term_1 = tf.math.exp(2 * (g_mean + g_var))  # [N, 1]
        term_1 = (1. - 2. * tau + 2. * tau ** 2) / ((tau ** 2) * ((1. - tau) ** 2)) * term_1  # [N, 1]

        term_2 = (tf.math.exp(g_var) - 1) * tf.math.exp(2 * g_mean + g_var)  # [N, 1]
        term_2 = ((1. - 2. * tau) ** 2) / ((tau ** 2) * ((1. - tau) ** 2)) * term_2  # [N, 1]
        term_2 = term_2 + f_var  # [N, 1]

        variance_of_y = term_1 + term_2  # [N, 1]

        # now calc Var(y | f < f*)

        # first we need Var(f | f<f*) for each of our M f* samples, i.e. truncated normal (see GIBBON or MES)
        normal = tfp.distributions.Normal(tf.cast(0, f_mean.dtype), tf.cast(1, f_mean.dtype))
        gamma = (tf.squeeze(self._samples) - f_mean) / f_sd  # [N, M]
        log_minus_cdf = normal.log_cdf(-gamma)  # [N, M]
        ratio = tf.math.exp(normal.log_prob(gamma) - log_minus_cdf)  # [N, M]
        variance_of_f_given_f_star = f_var * (1 + gamma * ratio - ratio ** 2)  # [N, M]

        # now fiddle Var(y) to get Var(y|f<f*) for each of our M f* samples
        variance_of_y_given_f_star = variance_of_y - f_var  # [N, 1]
        variance_of_y_given_f_star = variance_of_y_given_f_star + variance_of_f_given_f_star  # [N, M]

        # Entropy is H(y) - E_{f*}[H(y|f*)]
        # so approximate using MM approx H(y)=0.5log(2*pi*e*Var(y)) and H(y|f>f*)=0.5log(2*pi*e*Var(y|f>f*)) and use MC over our f* samples

        H_of_y = 0.5 * tf.math.log(variance_of_y)  # [N, 1]
        H_of_y_given_f_star = 0.5 * tf.math.log(variance_of_y_given_f_star)  # [N, M]
        entropy_reduction_for_each_f_star = H_of_y - H_of_y_given_f_star  # [N, M]

        return tf.math.reduce_mean(entropy_reduction_for_each_f_star, axis=1,
                                   keepdims=True)  # [N, 1] average over f* samples


class NegativeQuantilefromGaussianHetGPTrajectory(SingleModelGreedyAcquisitionBuilder):

    def __init__(self, quantile_level: float = 0.9):
        self._quantile_level = quantile_level

    def __repr__(self) -> str:
        return f"NegativeGaussianProcessTrajectory"

    def prepare_acquisition_function(
            self, model: FeaturedHetGPFluxModel, dataset: Dataset = None,
            pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        trajectory = model.sample_trajectory()

        def quantile_traj(at):
            lik_layer = model.model_gpflux.likelihood_layer
            dist = lik_layer.likelihood.conditional_distribution(trajectory(tf.squeeze(at, axis=1)))
            return -dist.quantile(value=self._quantile_level)

        return quantile_traj


class ProbabilityOfValidity(trieste.acquisition.SingleModelAcquisitionBuilder):
    def prepare_acquisition_function(self, model, dataset=None):
        def acquisition(at):
            mean, _ = model.predict(tf.squeeze(at, -2))
            return mean

        return acquisition


from trieste.acquisition.function import expected_improvement
from trieste.models import ProbabilisticModel


class myExpectedImprovement(ExpectedImprovement):

    def update_acquisition_function(
            self, function: AcquisitionFunction, model: ProbabilisticModel, dataset: Dataset = None
    ) -> AcquisitionFunction:
        tf.debugging.assert_positive(len(dataset))
        tf.debugging.Assert(isinstance(function, expected_improvement), [])
        mean, _ = model.predict(dataset.query_points)
        eta = tf.reduce_min(mean, axis=0)
        return expected_improvement(model, eta)
