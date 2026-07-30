from __future__ import annotations

import numpy as np

from .._regression import (
    aggregate,
    basic_errors,
    concordance_from,
    explained_variance_from,
    pearson_from,
    r2_from,
    r2_reductions,
    regression_reductions,
)


def mean_squared_error(preds, target, squared=True, num_outputs=1):
    if num_outputs == 1:
        errors, count = basic_errors(preds, target)
        value = errors[0] / count
        return float(value if squared else np.sqrt(value))
    reductions = regression_reductions(preds, target, num_outputs)
    values = reductions[:, 1] / reductions[:, 0]
    if not squared:
        values = np.sqrt(values)
    return float(values[0]) if values.size == 1 else values


def mean_absolute_error(preds, target, num_outputs=1):
    if num_outputs == 1:
        errors, count = basic_errors(preds, target)
        return float(errors[1] / count)
    reductions = regression_reductions(preds, target, num_outputs)
    values = reductions[:, 2] / reductions[:, 0]
    return float(values[0]) if values.size == 1 else values


def r2_score(preds, target, adjusted=0, multioutput="uniform_average"):
    return r2_from(r2_reductions(preds, target), adjusted, multioutput)


def explained_variance(
    preds, target, multioutput="uniform_average",
):
    return explained_variance_from(
        regression_reductions(preds, target), multioutput
    )


def mean_absolute_percentage_error(preds, target):
    reductions = regression_reductions(preds, target, 1)
    return float(reductions[0, 3] / reductions[0, 0])


def symmetric_mean_absolute_percentage_error(preds, target):
    reductions = regression_reductions(preds, target, 1)
    return float(reductions[0, 4] / reductions[0, 0])


def weighted_mean_absolute_percentage_error(preds, target):
    reductions = regression_reductions(preds, target, 1)
    denominator = reductions[0, 5]
    return float(reductions[0, 2] / denominator) if denominator else 0.0


def mean_squared_log_error(preds, target):
    reductions = regression_reductions(preds, target, 1, include_log=True)
    if np.isnan(reductions[0, 12]):
        raise ValueError("preds and target must be greater than -1")
    return float(reductions[0, 12] / reductions[0, 0])


def pearson_corrcoef(preds, target):
    return pearson_from(regression_reductions(preds, target))


def concordance_corrcoef(preds, target):
    return concordance_from(regression_reductions(preds, target))


__all__ = [
    "concordance_corrcoef",
    "explained_variance",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_error",
    "mean_squared_log_error",
    "pearson_corrcoef",
    "r2_score",
    "symmetric_mean_absolute_percentage_error",
    "weighted_mean_absolute_percentage_error",
]
