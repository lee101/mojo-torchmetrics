from __future__ import annotations

import numpy as np

from ._classification import array
from ._lib import addr, lib


WIDTH = 13
R2_PARALLEL_THRESHOLD = 1_000_000


def basic_errors(preds, target):
    prediction = array(preds, np.float64)
    truth = array(target, np.float64)
    if prediction.shape != truth.shape or not prediction.size:
        raise ValueError("preds and target must have the same non-empty shape")
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(truth)):
        raise ValueError("preds and target must contain only finite values")
    prediction = np.ascontiguousarray(prediction.reshape(-1))
    truth = np.ascontiguousarray(truth.reshape(-1))
    result = np.empty(2, dtype=np.float64)
    lib().mt_basic_errors(
        addr(prediction), addr(truth), addr(result), prediction.size
    )
    return result, prediction.size


def regression_reductions(preds, target, num_outputs=None, include_log=False):
    prediction = array(preds, np.float64)
    truth = array(target, np.float64)
    if prediction.shape != truth.shape or prediction.size == 0:
        raise ValueError("preds and target must have the same non-empty shape")
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(truth)):
        raise ValueError("preds and target must contain only finite values")
    if num_outputs is None:
        outputs = prediction.shape[-1] if prediction.ndim > 1 else 1
    else:
        if (
            isinstance(num_outputs, (bool, np.bool_))
            or not isinstance(num_outputs, (int, np.integer))
        ):
            raise ValueError("num_outputs must be a positive integer")
        outputs = int(num_outputs)
    if outputs < 1 or prediction.size % outputs:
        raise ValueError("num_outputs is incompatible with the input shape")
    rows = prediction.size // outputs
    prediction = np.ascontiguousarray(prediction.reshape(rows, outputs))
    truth = np.ascontiguousarray(truth.reshape(rows, outputs))
    reductions = np.empty((outputs, WIDTH), dtype=np.float64)
    log_safe = not include_log or not (
        np.any(prediction <= -1) or np.any(truth <= -1)
    )
    lib().mt_regression_reductions(
        addr(prediction), addr(truth), addr(reductions), rows, outputs,
        include_log,
    )
    if include_log and not log_safe:
        reductions[:, 12] = np.nan
    return reductions


def r2_reductions(preds, target):
    prediction = array(preds, np.float64)
    truth = array(target, np.float64)
    if prediction.shape != truth.shape or prediction.size == 0:
        raise ValueError("preds and target must have the same non-empty shape")
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(truth)):
        raise ValueError("preds and target must contain only finite values")
    outputs = prediction.shape[-1] if prediction.ndim > 1 else 1
    if outputs != 1:
        return regression_reductions(prediction, truth)
    prediction = np.ascontiguousarray(prediction.reshape(-1))
    truth = np.ascontiguousarray(truth.reshape(-1))
    partials = np.empty(24, dtype=np.float64)
    lib().mt_r2_reductions(
        addr(prediction), addr(truth), addr(partials), prediction.size
    )
    reductions = np.zeros((1, WIDTH), dtype=np.float64)
    reductions[0, 0] = prediction.size
    reductions[0, 1] = partials[0]
    reductions[0, 6] = partials[1]
    reductions[0, 8] = partials[2]
    return reductions


def aggregate(values, mode, variance=None):
    values = np.asarray(values, np.float64)
    if mode == "raw_values":
        return values
    if mode == "uniform_average":
        return float(np.mean(values))
    if mode == "variance_weighted":
        total = np.sum(variance)
        return float(np.average(values, weights=variance)) if total else float(np.mean(values))
    raise ValueError(
        "multioutput must be 'raw_values', 'uniform_average', or 'variance_weighted'"
    )


def means(reductions):
    count = reductions[:, 0]
    return reductions[:, 6] / count, reductions[:, 7] / count


def target_variance(reductions):
    count = reductions[:, 0]
    target_mean = reductions[:, 6] / count
    return reductions[:, 8] - count * target_mean * target_mean


def r2_from(reductions, adjusted=0, multioutput="uniform_average"):
    residual = reductions[:, 1]
    total = target_variance(reductions)
    raw = np.ones_like(total)
    normal = total > 1e-12
    raw[normal] = 1.0 - residual[normal] / total[normal]
    raw[~normal & (residual > 1e-12)] = 0.0
    observations = int(reductions[0, 0])
    if adjusted:
        if adjusted > observations - 1:
            raise ValueError("adjusted must be smaller than n_samples - 1")
        raw = 1.0 - (1.0 - raw) * (observations - 1) / (
            observations - adjusted - 1
        )
    return aggregate(raw, multioutput, total)


def explained_variance_from(reductions, multioutput="uniform_average"):
    count = reductions[:, 0]
    error_variance = reductions[:, 1] - reductions[:, 11] ** 2 / count
    total = target_variance(reductions)
    raw = np.ones_like(total)
    normal = total > 1e-12
    raw[normal] = 1.0 - error_variance[normal] / total[normal]
    raw[~normal & (error_variance > 1e-12)] = 0.0
    return aggregate(raw, multioutput, total)


def pearson_from(reductions):
    count = reductions[:, 0]
    covariance = reductions[:, 10] - reductions[:, 6] * reductions[:, 7] / count
    target_ss = target_variance(reductions)
    pred_ss = reductions[:, 9] - reductions[:, 7] ** 2 / count
    denominator = np.sqrt(np.maximum(target_ss * pred_ss, 0.0))
    result = np.divide(
        covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0
    )
    return float(result[0]) if result.size == 1 else result


def concordance_from(reductions):
    count = reductions[:, 0]
    target_mean, pred_mean = means(reductions)
    degrees = np.maximum(count - 1.0, 1.0)
    covariance = (
        reductions[:, 10] - count * target_mean * pred_mean
    ) / degrees
    target_var = (
        reductions[:, 8] - count * target_mean**2
    ) / degrees
    pred_var = (
        reductions[:, 9] - count * pred_mean**2
    ) / degrees
    denominator = target_var + pred_var + (target_mean - pred_mean) ** 2
    result = np.divide(
        2 * covariance, denominator, out=np.zeros_like(covariance),
        where=denominator > 0,
    )
    return float(result[0]) if result.size == 1 else result
