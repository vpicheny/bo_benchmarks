from __future__ import annotations

import numpy as np
import tensorflow as tf
import trieste
import tensorflow_probability as tfp
from scipy.optimize import bisect
from trieste.data import Dataset
from trieste.acquisition.rule import OBJECTIVE, EfficientGlobalOptimization
from trieste.acquisition.interface import SingleModelGreedyAcquisitionBuilder, AcquisitionFunction,  PenalizationFunction
from trieste.acquisition.function import ExpectedImprovement, MinValueEntropySearch, min_value_entropy_search, LocalPenalization, GIBBON, gibbon_repulsion_term, gibbon_quality_term
from trieste.acquisition.sampler import GumbelSampler
from trieste.types import TensorType
from typing import Optional, cast
from model_utils import FeaturedHetGPFluxModel
tf.keras.backend.set_floatx("float64")


class GIBBONForQuantile(GIBBON):
    def __init__(
        self,
        search_space: SearchSpace,
        quantile_level:float,
        num_samples: int = 5,
        grid_size: int = 1000,
        rescaled_repulsion: bool = True,
    ):   

        tf.debugging.assert_positive(num_samples)
        tf.debugging.assert_positive(grid_size)

        min_value_sampler = QuantileGumbelSampler(sample_min_value=True) # MIGHT BE WORTH USING THE TRAJECTORIES

        self._quantile_level = quantile_level
 
        self._min_value_sampler = min_value_sampler
        self._search_space = search_space
        self._num_samples = num_samples
        self._grid_size = grid_size
        self._rescaled_repulsion = rescaled_repulsion

        self._min_value_samples: Optional[TensorType] = None
        self._quality_term: Optional[gibbon_quality_term] = None
        self._diversity_term: Optional[gibbon_repulsion_term] = None
        self._gibbon_acquisition: Optional[AcquisitionFunction] = None

    def prepare_acquisition_function(
        self,
        model: GIBBONModelType,
        dataset: Optional[Dataset] = None,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: The data from the observer. Must be populated.
        :param pending_points: The points we penalize with respect to.
        :return: The GIBBON acquisition function modified for objective minimisation.
        :raise tf.errors.InvalidArgumentError: If ``dataset`` is empty.
        """

        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")

        acq = self._update_quality_term(dataset, model)
        if pending_points is not None and len(pending_points) != 0:
            acq = self._update_repulsion_term(acq, dataset, model, pending_points)

        return acq

    def _update_repulsion_term(
        self,
        function: Optional[AcquisitionFunction],
        dataset: Dataset,
        model: GIBBONModelType,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        tf.debugging.assert_rank(pending_points, 2)

        if self._gibbon_acquisition is not None and isinstance(
            self._diversity_term, gibbon_repulsion_term_for_quantile
        ):
            # if possible, just update the repulsion term
            self._diversity_term.update(pending_points)
            return self._gibbon_acquisition
        else:
            # otherwise construct a new repulsion term and acquisition function
            self._diversity_term = gibbon_repulsion_term_for_quantile(
                model, pending_points, rescaled_repulsion=self._rescaled_repulsion, quantile_level = self._quantile_level
            )

            @tf.function
            def gibbon_acquisition(x: TensorType) -> TensorType:
                return cast(PenalizationFunction, self._diversity_term)(x) + cast(
                    AcquisitionFunction, self._quality_term
                )(x)

            self._gibbon_acquisition = gibbon_acquisition
            return gibbon_acquisition

    def _update_quality_term(self, dataset: Dataset, model: GIBBONModelType) -> AcquisitionFunction:
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")

        query_points = self._search_space.sample(num_samples=self._grid_size)
        tf.debugging.assert_same_float_dtype([dataset.query_points, query_points])
        query_points = tf.concat([dataset.query_points, query_points], 0)
        self._min_value_samples = self._min_value_sampler.sample(
            model, self._num_samples, query_points
        )

        if self._quality_term is not None:  # if possible, just update the quality term
            self._quality_term.update(self._min_value_samples)
        else:  # otherwise build quality term
            self._quality_term = min_value_entropy_search_for_quantile(model, self._min_value_samples,self._quantile_level)
        return cast(AcquisitionFunction, self._quality_term)




class gibbon_repulsion_term_for_quantile(gibbon_repulsion_term):
    def __init__(
        self,
        model: SupportsCovarianceObservationNoise,
        pending_points: TensorType,
        quantile_level: float,
        rescaled_repulsion: bool = True,
    ):
        tf.debugging.assert_rank(pending_points, 2)
        tf.debugging.assert_positive(len(pending_points))

        self._model = model
        self._pending_points = tf.Variable(pending_points, shape=[None, *pending_points.shape[1:]])
        self._rescaled_repulsion = rescaled_repulsion
        self._quantile_level = quantile_level


    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This penalization function cannot be calculated for batches of points.",
        )


        tau = self._quantile_level
        const = (1 - 2.0 * tau)**2 / ((tau**2)*((1-tau)**2))


        fmean, fvar = self._model.predict(tf.squeeze(x, -2))

        combined_mean, combined_var = self._model.model_gpflux.predict_f(x)
        f_mean = combined_mean[:,:,0] # [N, 1]
        f_var = combined_var[:,:,0] # [N, 1]
        g_mean = combined_mean[:,:,1] # [N, 1]
        g_var = combined_var[:,:,1] # [N, 1]

        # calc variance of y
        term_1 = tf.math.exp(2*(g_mean+g_var)) # [N, 1]
        term_1 = (1. - 2. * tau + 2. * tau**2) /((tau**2) * ((1. - tau)**2)) * term_1 # [N, 1]
        term_2 = (tf.math.exp(g_var) - 1) * tf.math.exp(2*g_mean + g_var) # [N, 1]
        term_2 = const * term_2 # [N, 1]
        term_2 = term_2 + f_var # [N, 1]
        y_var = term_1 + term_2 # [N, 1]

        # calc predictions for pending points
        combined_pending_mean, combined_pending_var = self._model.model_gpflux.f_layers[0].predict(self._pending_points)  # [m, 2]
        _, combined_pending_covar = self._model.model_gpflux.f_layers[0].predict(self._pending_points, full_cov=True)  # [1, m,m]
        f_mean_pending = combined_pending_mean[:,:1] # [m, 1]
        g_mean_pending = combined_pending_mean[:,1:2] # [m, 1]
        f_var_pending = combined_pending_var[:,:1] # [m, 1]
        g_var_pending = combined_pending_var[:,1:2] # [m, 1]
        f_covar_pending = combined_pending_covar[0,:,:] # [m, m]
        g_covar_pending = combined_pending_covar[1,:,:] # [m, m]


        # calc y variance for pending points
        term_1 = tf.math.exp(2*(g_mean_pending+g_var_pending)) # [N, 1]
        term_1 = (1. - 2. * tau + 2. * tau**2) /((tau**2) * ((1. - tau)**2)) * term_1 # [N, 1]
        term_2 = (tf.math.exp(g_var_pending) - 1) * tf.math.exp(2*g_mean_pending + g_var_pending) # [N, 1]
        term_2 = const * term_2 # [N, 1]
        term_2 = term_2 + f_var_pending # [N, 1]
        y_var_pending = term_1 + term_2 # [N, 1]


        # calc covariance between pending points
        sigma_covar_pending = g_mean_pending + tf.transpose(g_mean_pending) # [m,m]
        sigma_covar_pending += (g_var_pending +tf.transpose(g_var_pending))/0.5 # [m,m]
        sigma_covar_pending = tf.math.exp(sigma_covar_pending)*(tf.math.exp(g_covar_pending)-1.0) # [m,m]
        y_covar_pending = f_covar_pending + const * sigma_covar_pending
        y_covar_pending = tf.linalg.set_diag(y_covar_pending, tf.squeeze(y_var_pending,-1))
        B = y_covar_pending # [m, m]
        B = tf.expand_dims(B,0) # [1, m , m]

        # calculate covariance between candidate and pending points (cov(y_1,y_2 | f, g) = 0 unless y_1=y_2 )
        covars = self._model.covariance_between_points(tf.squeeze(x, -2), self._pending_points)
        f_covar = covars[0][0] # [1, N, m]
        g_covar = covars[1][0] # [1, N, m] 
        sigma_covar = g_mean + tf.transpose(g_mean_pending) # [N,m]
        sigma_covar += (g_var + tf.transpose(g_var_pending)) / 0.5
        sigma_covar = tf.math.exp(sigma_covar) * (tf.math.exp(g_covar)-1.0)
        y_covar = f_covar + const * sigma_covar # [N, m]
        A = y_covar # [N, m]
        A = tf.expand_dims(A, -1) # [N, m , 1]

        # efficiently calc det of covariance
        L = tf.linalg.cholesky(B) # [1, m, m]
        L_inv_A = tf.linalg.triangular_solve(L, A)
        V_det = y_var - tf.squeeze(
            tf.matmul(L_inv_A, L_inv_A, transpose_a=True), -1
        )  # equation for determinant of block matrices
        repulsion = 0.5 * (tf.math.log(V_det) - tf.math.log(y_var))

        if self._rescaled_repulsion:
            batch_size = tf.cast(tf.shape(self._pending_points)[0], dtype=fmean.dtype)
            repulsion_weight = (1 / batch_size) ** (2)
        else:
            repulsion_weight = 1.0

        return repulsion_weight * repulsion




class MinValueEntropySearchForQuantile(MinValueEntropySearch):
    def __init__(
        self,
        search_space: SearchSpace,
        quantile_level:float,
        num_samples: int = 5,
        grid_size: int = 1000,
    ):
        """
        """
 
        min_value_sampler = QuantileGumbelSampler(sample_min_value=True) # MIGHT BE WORTH USING THE TRAJECTORIES

        self._quantile_level = quantile_level
        self._min_value_sampler = min_value_sampler
        self._search_space = search_space
        self._num_samples = num_samples
        self._grid_size = grid_size

    def __repr__(self) -> str:
        return f"MinValueEntropySearchForQuantile"

    def prepare_acquisition_function(self, model: FeaturedHetGPFluxModel, dataset: Optional[Dataset] = None) -> AcquisitionFunction:

        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        query_points = self._search_space.sample(num_samples=self._grid_size)
        tf.debugging.assert_same_float_dtype([dataset.query_points, query_points])
        query_points = tf.concat([dataset.query_points, query_points], 0)
        min_value_samples = self._min_value_sampler.sample(model, self._num_samples, query_points)
        return min_value_entropy_search_for_quantile(model, min_value_samples, self._quantile_level)

    def update_acquisition_function(
        self,function: AcquisitionFunction,model: FeaturedHetGPFluxModel,dataset: Optional[Dataset] = None) -> AcquisitionFunction:

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

class min_value_entropy_search_for_quantile(min_value_entropy_search):
    def __init__(self, model: ProbabilisticModel, samples: TensorType, quantile_level: float):
        r"""
        Return the max-value entropy search acquisition function adapted for quantile models
        """
        tf.debugging.assert_rank(samples, 2)
        tf.debugging.assert_positive(len(samples))

        self._model = model
        self._samples = tf.Variable(samples)
        self._quantile_level = quantile_level


    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        
        tau = self._quantile_level


        combined_mean, combined_var = self._model.model_gpflux.predict_f(x)
        f_mean = combined_mean[:,:,0] # [N, 1]
        f_var = combined_var[:,:,0] # [N, 1]
        g_mean = combined_mean[:,:,1] # [N, 1]
        g_var = combined_var[:,:,1] # [N, 1]
        f_sd = tf.clip_by_value(
            tf.math.sqrt(f_var), 1e-10, f_mean.dtype.max
        )  # clip below to improve numerical stability


        # entropy reduction is  H(y) - E_{f*}[H(y|f>f*)]
        # so we need H(y) and H(y| f>f*)
        # lets do MM approx H(y)=0.5log(2*pi*e*Var(y)) and H(y|f>f*)=0.5log(2*pi*e*Var(y|f>f*))
        
        # first calc Var(y) = Var_{q,sigma}(E[y|q,sigma]) + E{q,sigma}[Var(y|q, sigma)] 
        # where sigma = e^g and q=f (for gaussian f and g)
        term_1 = tf.math.exp(2*(g_mean+g_var)) # [N, 1]
        term_1 = (1. - 2. * tau + 2. * tau**2) /((tau**2) * ((1. - tau)**2)) * term_1 # [N, 1]
    
        term_2 = (tf.math.exp(g_var) - 1) * tf.math.exp(2*g_mean + g_var) # [N, 1]
        term_2 = ((1. -2. * tau)**2) /((tau**2) * ((1. - tau)**2)) * term_2 # [N, 1]
        term_2 = term_2 + f_var # [N, 1]

        variance_of_y = term_1 + term_2 # [N, 1]

        # now calc Var(y | f < f*) 
        
        # first we need Var(f | f<f*) for each of our M f* samples, i.e. truncated normal (see GIBBON or MES)
        normal = tfp.distributions.Normal(tf.cast(0, f_mean.dtype), tf.cast(1, f_mean.dtype))
        gamma = (tf.squeeze(self._samples) - f_mean) / f_sd # [N, M]
        log_minus_cdf = normal.log_cdf(-gamma) # [N, M]
        ratio = tf.math.exp(normal.log_prob(gamma) - log_minus_cdf)  # [N, M]
        variance_of_f_given_f_star = f_var * (1 + gamma * ratio - ratio**2)  # [N, M]

        # now fiddle Var(y) to get Var(y|f<f*) for each of our M f* samples
        variance_of_y_given_f_star = variance_of_y - f_var  # [N, 1] 
        variance_of_y_given_f_star = variance_of_y_given_f_star + variance_of_f_given_f_star # [N, M]

        # Entropy is H(y) - E_{f*}[H(y|f*)]
        # so approximate using MM approx H(y)=0.5log(2*pi*e*Var(y)) and H(y|f>f*)=0.5log(2*pi*e*Var(y|f>f*)) and use MC over our f* samples

        H_of_y = 0.5 * tf.math.log(variance_of_y)  # [N, 1]
        H_of_y_given_f_star = 0.5 * tf.math.log(variance_of_y_given_f_star) # [N, M]
        entropy_reduction_for_each_f_star = H_of_y - H_of_y_given_f_star # [N, M]

        return tf.math.reduce_mean(entropy_reduction_for_each_f_star, axis=1, keepdims=True) # [N, 1] average over f* samples
 


class QuantileGumbelSampler(GumbelSampler):

    def sample(self, model: ProbabilisticModel, sample_size: int, at: TensorType) -> TensorType:
        """
        Return approximate samples from of the objective function's minimum value.
        """
        tf.debugging.assert_positive(sample_size)
        tf.debugging.assert_shapes([(at, ["N", None])])

        fmean, fvar = model.predict(at)
        fmean = fmean[:,0:1] # only need posterior for f to get samples of f*
        fvar = fvar[:,0:1]
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
