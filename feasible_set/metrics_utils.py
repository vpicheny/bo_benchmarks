from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp
from trieste.acquisition.rule import OBJECTIVE
from trieste.models import TrainableProbabilisticModel
from trieste.types import TensorType


def compute_metrics(ask_tell, CONFIG):
    model = ask_tell._models[OBJECTIVE]
    accuracy_global = _get_excursion_accuracy(CONFIG.problem.global_test_points, model,
                                              CONFIG.problem.threshold)

    accuracy_boundary = _get_excursion_accuracy(CONFIG.problem.boundary_test_points, model,
                                                CONFIG.problem.threshold)
    return accuracy_global, accuracy_boundary


def _excursion_probability(
    x: TensorType, model: TrainableProbabilisticModel, threshold: int
) -> tfp.distributions.Distribution:
    mean, variance = model.model.predict_f(x)  # type: ignore
    normal = tfp.distributions.Normal(tf.cast(0, x.dtype), tf.cast(1, x.dtype))
    t = (mean - threshold) / tf.sqrt(variance)
    return normal.cdf(t)


def _get_excursion_accuracy(
    x: TensorType, model: TrainableProbabilisticModel, threshold: int
) -> float:
    prob = _excursion_probability(x, model, threshold)
    accuracy = tf.reduce_sum(prob * (1 - prob), axis=0)

    return accuracy
