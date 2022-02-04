from __future__ import annotations

import numpy as np
import tensorflow as tf
import trieste
import tensorflow_probability as tfp
from scipy.optimize import bisect
from trieste.data import Dataset
from trieste.acquisition.rule import OBJECTIVE, EfficientGlobalOptimization
from trieste.acquisition.interface import SingleModelGreedyAcquisitionBuilder, AcquisitionFunction
from trieste.acquisition.function import ExpectedImprovement
from trieste.types import TensorType
from typing import Optional, cast
from model_utils import FeaturedHetGPFluxModel
tf.keras.backend.set_floatx("float64")

from entropy_utils import MinValueEntropySearchForQuantile, GIBBONForQuantile
from lp_utils import LocalPenalizationForQuantile, soft_local_penalizer_for_quantile, hard_local_penalizer_for_quantile



def create_initial_query_points(search_space, CONFIG):
    if CONFIG.model == "GPR":
        query_points = search_space.sample_halton(np.round(CONFIG.num_initial_points / CONFIG.batch_size))
        return query_points
    else:
        return search_space.sample_halton(CONFIG.num_initial_points)

def create_acquisition_rule(CONFIG):
    if CONFIG.model == "quantile":
        
        search_space = trieste.space.Box(CONFIG.problem.lower_bounds, CONFIG.problem.upper_bounds)
        d = tf.shape(CONFIG.problem.lower_bounds)[0]

        # commented out TS for now
        #acq_function = NegativeGaussianProcessTrajectory()
        #return trieste.acquisition.rule.EfficientGlobalOptimization(acq_function.using(OBJECTIVE),num_query_points=CONFIG.batch_size)
        
        #commented out new MES combined with LP for now
       
        acq_function = LocalPenalizationForQuantile(
            search_space,
            num_samples = 10_000 * d, # perhaps too large, might be slow
            penalizer = soft_local_penalizer_for_quantile, # also try hard_local_penalizer_for_quantile
            base_acquisition_function_builder = MinValueEntropySearchForQuantile(
                search_space,
                quantile_level = CONFIG.problem.quantile_level,
                num_samples = 10,
                grid_size = 10_000 * d, # perhaps too large, might be slow
            )
        )

        # lets go full new GIBBON


        acq_function = GIBBONForQuantile(
            search_space,
            quantile_level = CONFIG.problem.quantile_level,
            num_samples=10,
            grid_size = 10_000 * d,
        )


        LocalPenalizationForQuantile(
            search_space,
            num_samples = 10_000 * d, # perhaps too large, might be slow
            penalizer = soft_local_penalizer_for_quantile, # also try hard_local_penalizer_for_quantile
            base_acquisition_function_builder = MinValueEntropySearchForQuantile(
                search_space,
                quantile_level = CONFIG.problem.quantile_level,
                num_samples = 10,
                grid_size = 10_000 * d, # perhaps too large, might be slow
            )
        )




        return trieste.acquisition.rule.EfficientGlobalOptimization(acq_function.using(OBJECTIVE),num_query_points=CONFIG.batch_size)
       


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
        self, function: AcquisitionFunction, model: ProbabilisticModel, dataset: Dataset=None
    ) -> AcquisitionFunction:

        tf.debugging.assert_positive(len(dataset))
        tf.debugging.Assert(isinstance(function, expected_improvement), [])
        mean, _ = model.predict(dataset.query_points)
        eta = tf.reduce_min(mean, axis=0)
        return expected_improvement(model, eta)
