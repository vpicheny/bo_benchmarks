from __future__ import annotations

import numpy as np
import tensorflow as tf
import trieste
import tensorflow_probability as tfp
from scipy.optimize import bisect
from trieste.data import Dataset
from trieste.acquisition.rule import OBJECTIVE, EfficientGlobalOptimization
from trieste.acquisition.interface import SingleModelGreedyAcquisitionBuilder, AcquisitionFunction
from trieste.acquisition.function import ExpectedImprovement, MinValueEntropySearch, min_value_entropy_search, LocalPenalization, soft_local_penalizer, hard_local_penalizer
from trieste.acquisition.sampler import GumbelSampler
from trieste.types import TensorType
from typing import Optional, cast
from model_utils import FeaturedHetGPFluxModel
tf.keras.backend.set_floatx("float64")



class LocalPenalizationForQuantile(LocalPenalization):
    @tf.function(experimental_relax_shapes=True)
    def _get_lipschitz_estimate(
        self, model: ProbabilisticModel, sampled_points: TensorType
    ) -> tuple[TensorType, TensorType]:
        with tf.GradientTape() as g:
            g.watch(sampled_points)
            mean, _ = model.predict(sampled_points)
        grads = g.gradient(mean, sampled_points)
        grads_norm = tf.norm(grads, axis=1)
        max_grads_norm = tf.reduce_max(grads_norm)
        eta = tf.reduce_min(mean, axis=0)
        return max_grads_norm, eta


class soft_local_penalizer_for_quantile(soft_local_penalizer):
    def __init__(
        self,
        model: ProbabilisticModel,
        pending_points: TensorType,
        lipschitz_constant: TensorType,
        eta: TensorType,
    ):

        self._model = model

        mean_pending, variance_pending = model.predict(pending_points)
        mean_pending = mean_pending[:,0:1] # only need posterior for f to get samples of f*
        variance_pending = variance_pending[:,0:1]
        self._pending_points = tf.Variable(pending_points, shape=[None, *pending_points.shape[1:]])
        self._radius = tf.Variable(
            tf.transpose((mean_pending - eta) / lipschitz_constant),
            shape=[1, None],
        )
        self._scale = tf.Variable(
            tf.transpose(tf.sqrt(variance_pending) / lipschitz_constant),
            shape=[1, None],
        )





class  hard_local_penalizer_for_quantile(hard_local_penalizer):
    def __init__(
        self,
        model: ProbabilisticModel,
        pending_points: TensorType,
        lipschitz_constant: TensorType,
        eta: TensorType,
    ):
        self._model = model

        mean_pending, variance_pending = model.predict(pending_points)
        mean_pending = mean_pending[:,0:1] # only need posterior for f to get samples of f*
        variance_pending = variance_pending[:,0:1]

        self._pending_points = tf.Variable(pending_points, shape=[None, *pending_points.shape[1:]])
        self._radius = tf.Variable(
            tf.transpose((mean_pending - eta) / lipschitz_constant),
            shape=[1, None],
        )
        self._scale = tf.Variable(
            tf.transpose(tf.sqrt(variance_pending) / lipschitz_constant),
            shape=[1, None],
        )


