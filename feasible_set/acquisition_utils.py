from __future__ import annotations

from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition.function import (
    ExpectedFeasibility,
    IntegratedVarianceReduction,
    LocalPenalization,
    Fantasizer,
)


def create_acquisition_rule(CONFIG, search_space):
    if CONFIG.rule == "nobatch-ranjan":
        return EfficientGlobalOptimization(ExpectedFeasibility(threshold=CONFIG.problem.threshold))

    elif CONFIG.rule == "lp-ranjan":
        acq = LocalPenalization(search_space,
                                base_acquisition_function_builder=ExpectedFeasibility(threshold=CONFIG.problem.threshold))
        return EfficientGlobalOptimization(  # type: ignore
            num_query_points=CONFIG.batch_size, builder=acq
        )  # TODO: this doesn't work

    elif CONFIG.rule == "kb-ranjan":
        acq = Fantasizer(ExpectedFeasibility(threshold=CONFIG.problem.threshold))
        return EfficientGlobalOptimization(  # type: ignore
            num_query_points=CONFIG.batch_size, builder=acq
        )

    elif CONFIG.rule == "evr":
        return EfficientGlobalOptimization(
            IntegratedVarianceReduction(search_space.sample_sobol(1000), threshold=CONFIG.problem.threshold),
        num_query_points=CONFIG.batch_size)
    else:
        raise NotImplementedError
