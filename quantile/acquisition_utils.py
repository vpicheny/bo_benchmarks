from __future__ import annotations

import numpy as np
import tensorflow as tf
import trieste
import tensorflow_probability as tfp
from trieste.data import Dataset
from trieste.acquisition.rule import OBJECTIVE, EfficientGlobalOptimization
from trieste.acquisition.interface import SingleModelGreedyAcquisitionBuilder, AcquisitionFunction
from trieste.acquisition.function import ExpectedImprovement
from trieste.types import TensorType
from typing import Optional
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
        quantile_traj = NegativeGaussianProcessTrajectory()
        return trieste.acquisition.rule.EfficientGlobalOptimization(quantile_traj.using(OBJECTIVE),
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
