from __future__ import annotations

import math

import numpy as np

from ._lib import addr, lib


def array(value, dtype):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    values = np.asarray(value)
    if np.issubdtype(values.dtype, np.complexfloating):
        raise TypeError("complex inputs are not supported")
    return np.asarray(values, dtype=dtype)


def integer_array(value, name):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    values = np.asarray(value)
    if np.issubdtype(values.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain integers")
    if np.issubdtype(values.dtype, np.floating):
        if not np.all(np.isfinite(values)) or np.any(values != np.floor(values)):
            raise ValueError(f"{name} must contain finite integer values")
    try:
        converted = np.asarray(values, dtype=np.int64)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} cannot be represented as int64") from error
    if values.dtype == object:
        try:
            if np.any(values != converted):
                raise ValueError(f"{name} must contain integer values")
        except TypeError as error:
            raise ValueError(f"{name} must contain integer values") from error
    return converted


def probabilities(preds):
    values = array(preds, np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("preds must contain only finite values")
    if values.size and (values.min() < 0.0 or values.max() > 1.0):
        positive = values >= 0
        transformed = np.empty_like(values)
        transformed[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        exp_values = np.exp(values[~positive])
        transformed[~positive] = exp_values / (1.0 + exp_values)
        values = transformed
    return np.ascontiguousarray(values.reshape(-1))


def binary_matrix(preds, target, threshold=0.5, ignore_index=None):
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and between 0 and 1")
    scores = probabilities(preds)
    truth = np.ascontiguousarray(integer_array(target, "target").reshape(-1))
    if scores.size != truth.size or not scores.size:
        raise ValueError("preds and target must have the same non-zero number of elements")
    if np.any((truth != 0) & (truth != 1) & (truth != ignore_index)):
        raise ValueError("target must contain only 0 and 1")
    matrix = np.empty((2, 2), dtype=np.float64)
    lib().mt_binary_confusion(
        addr(scores), addr(truth), addr(matrix), truth.size, threshold,
        ignore_index or 0, ignore_index is not None,
    )
    return matrix


def multiclass_matrix(preds, target, num_classes, ignore_index=None, top_k=1):
    if (
        isinstance(num_classes, (bool, np.bool_))
        or not isinstance(num_classes, (int, np.integer))
        or num_classes < 2
    ):
        raise ValueError("num_classes must be an integer greater than 1")
    pred_ndim = preds.ndim if hasattr(preds, "ndim") else np.asarray(preds).ndim
    target_ndim = (
        target.ndim if hasattr(target, "ndim") else np.asarray(target).ndim
    )
    values = (
        array(preds, np.float64)
        if pred_ndim > target_ndim
        else integer_array(preds, "preds")
    )
    truth = integer_array(target, "target")
    if values.ndim == truth.ndim + 1:
        if top_k != 1:
            raise NotImplementedError("top_k greater than 1 is not covered")
        if values.shape[1] != num_classes:
            raise ValueError("the class-score dimension must equal num_classes")
        if not np.all(np.isfinite(values)):
            raise ValueError("preds must contain only finite values")
        values = np.argmax(values, axis=1)
    elif top_k != 1:
        raise ValueError("top_k requires floating class scores")
    prediction = np.ascontiguousarray(values.reshape(-1), dtype=np.int64)
    truth = np.ascontiguousarray(truth.reshape(-1))
    if prediction.size != truth.size or not truth.size:
        raise ValueError("preds and target shapes are incompatible")
    valid = truth != ignore_index if ignore_index is not None else np.ones(truth.shape, bool)
    if np.any((truth[valid] < 0) | (truth[valid] >= num_classes)):
        raise ValueError("target contains a class outside num_classes")
    if np.any((prediction[valid] < 0) | (prediction[valid] >= num_classes)):
        raise ValueError("preds contains a class outside num_classes")
    matrix = np.empty((num_classes, num_classes), dtype=np.float64)
    lib().mt_multiclass_confusion(
        addr(prediction), addr(truth), addr(matrix), truth.size, num_classes,
        ignore_index or 0, ignore_index is not None,
    )
    return matrix


def normalize_matrix(matrix, normalize):
    if normalize in (None, "none"):
        return matrix.astype(np.int64)
    if normalize not in ("true", "pred", "all"):
        raise ValueError("normalize must be 'true', 'pred', 'all', 'none', or None")
    axis = 1 if normalize == "true" else 0
    denominator = matrix.sum(axis=axis, keepdims=True) if normalize != "all" else matrix.sum()
    return np.divide(matrix, denominator, out=np.zeros_like(matrix), where=denominator != 0)


def class_stats(matrix):
    tp = np.diag(matrix)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    tn = matrix.sum() - tp - fp - fn
    return tp, fp, fn, tn


def divide(numerator, denominator, zero_division=0.0):
    numerator = np.asarray(numerator, np.float64)
    denominator = np.asarray(denominator, np.float64)
    return np.divide(
        numerator, denominator,
        out=np.full(np.broadcast_shapes(numerator.shape, denominator.shape), zero_division, dtype=np.float64),
        where=denominator != 0,
    )


def reduce_scores(scores, support, average):
    if average in (None, "none"):
        return scores
    if average == "macro":
        return float(np.mean(scores))
    if average == "weighted":
        return float(np.average(scores, weights=support)) if support.sum() else 0.0
    raise ValueError("average must be 'micro', 'macro', 'weighted', 'none', or None")


def score_from_matrix(matrix, metric, average="macro", zero_division=0.0):
    tp, fp, fn, tn = class_stats(matrix)
    if average == "micro":
        tp, fp, fn, tn = (np.array([x.sum()]) for x in (tp, fp, fn, tn))
        average = None
    if metric == "accuracy" or metric == "recall":
        scores = divide(tp, tp + fn, zero_division)
    elif metric == "precision":
        scores = divide(tp, tp + fp, zero_division)
    elif metric == "specificity":
        scores = divide(tn, tn + fp, zero_division)
    elif metric == "f1":
        scores = divide(2 * tp, 2 * tp + fp + fn, zero_division)
    elif metric == "jaccard":
        scores = divide(tp, tp + fp + fn, zero_division)
    else:
        raise ValueError(metric)
    result = reduce_scores(scores, tp + fn, average)
    return float(result[0]) if isinstance(result, np.ndarray) and result.size == 1 else result


def binary_score_from_matrix(matrix, metric, zero_division=0.0):
    tn, fp, fn, tp = matrix.ravel()
    if metric == "accuracy":
        numerator, denominator = tp + tn, matrix.sum()
    elif metric == "precision":
        numerator, denominator = tp, tp + fp
    elif metric == "recall":
        numerator, denominator = tp, tp + fn
    elif metric == "specificity":
        numerator, denominator = tn, tn + fp
    elif metric == "f1":
        numerator, denominator = 2 * tp, 2 * tp + fp + fn
    elif metric == "jaccard":
        numerator, denominator = tp, tp + fp + fn
    else:
        raise ValueError(metric)
    return float(numerator / denominator) if denominator else float(zero_division)


def matthews(matrix):
    total = matrix.sum()
    correct = np.trace(matrix)
    predicted = matrix.sum(axis=0)
    actual = matrix.sum(axis=1)
    numerator = correct * total - np.dot(predicted, actual)
    denominator = math.sqrt(
        (total * total - np.dot(predicted, predicted))
        * (total * total - np.dot(actual, actual))
    )
    return numerator / denominator if denominator else 0.0


def kappa(matrix, weights=None):
    classes = matrix.shape[0]
    if weights in (None, "none"):
        weight = np.ones_like(matrix) - np.eye(classes)
    elif weights == "linear":
        index = np.arange(classes)
        weight = np.abs(index[:, None] - index[None, :]) / max(classes - 1, 1)
    elif weights == "quadratic":
        index = np.arange(classes)
        weight = ((index[:, None] - index[None, :]) / max(classes - 1, 1)) ** 2
    else:
        raise ValueError("weights must be None, 'none', 'linear', or 'quadratic'")
    total = matrix.sum()
    if not total:
        return 0.0
    expected = np.outer(matrix.sum(axis=1), matrix.sum(axis=0)) / total
    denominator = np.sum(weight * expected)
    return 1.0 - np.sum(weight * matrix) / denominator if denominator else 0.0


def binary_ranking(preds, target, ignore_index=None):
    scores = probabilities(preds)
    truth = np.ascontiguousarray(integer_array(target, "target").reshape(-1))
    if scores.size != truth.size or not scores.size:
        raise ValueError("preds and target must have the same non-zero number of elements")
    valid = truth != ignore_index if ignore_index is not None else np.ones(truth.shape, bool)
    if np.any((truth[valid] != 0) & (truth[valid] != 1)):
        raise ValueError("target must contain only 0 and 1")
    pair_work = np.empty(
        scores.size, dtype=[("score", np.float64), ("target", np.int64)]
    )
    result = np.empty(2, dtype=np.float64)
    kept = lib().mt_binary_ranking(
        addr(scores), addr(truth), addr(pair_work), addr(result),
        truth.size, ignore_index or 0, ignore_index is not None,
    )
    if kept < 0 or kept > truth.size:
        raise RuntimeError("Mojo ranking kernel returned an invalid result length")
    return result
