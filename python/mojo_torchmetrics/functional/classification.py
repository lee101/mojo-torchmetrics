from __future__ import annotations

import numpy as np

from .._classification import (
    binary_score_from_matrix,
    binary_matrix,
    binary_ranking,
    kappa,
    matthews,
    multiclass_matrix,
    normalize_matrix,
    score_from_matrix,
)


def binary_accuracy(
    preds, target, threshold=0.5, multidim_average="global",
    ignore_index=None, validate_args=True,
):
    _global_only(multidim_average)
    return score_from_matrix(
        binary_matrix(preds, target, threshold, ignore_index), "accuracy", "micro"
    )


def multiclass_accuracy(
    preds, target, num_classes=None, average="macro", top_k=1,
    multidim_average="global", ignore_index=None, validate_args=True,
):
    _global_only(multidim_average)
    if num_classes is None:
        raise ValueError("num_classes is required")
    return score_from_matrix(
        multiclass_matrix(preds, target, num_classes, ignore_index, top_k),
        "accuracy", average,
    )


def binary_precision(
    preds, target, threshold=0.5, multidim_average="global",
    ignore_index=None, validate_args=True, zero_division=0,
):
    _global_only(multidim_average)
    matrix = binary_matrix(preds, target, threshold, ignore_index)
    return binary_score_from_matrix(matrix, "precision", zero_division)


def multiclass_precision(
    preds, target, num_classes, average="macro", top_k=1,
    multidim_average="global", ignore_index=None, validate_args=True,
    zero_division=0,
):
    _global_only(multidim_average)
    return score_from_matrix(
        multiclass_matrix(preds, target, num_classes, ignore_index, top_k),
        "precision", average, zero_division,
    )


def binary_recall(
    preds, target, threshold=0.5, multidim_average="global",
    ignore_index=None, validate_args=True, zero_division=0,
):
    _global_only(multidim_average)
    matrix = binary_matrix(preds, target, threshold, ignore_index)
    return binary_score_from_matrix(matrix, "recall", zero_division)


def multiclass_recall(
    preds, target, num_classes, average="macro", top_k=1,
    multidim_average="global", ignore_index=None, validate_args=True,
    zero_division=0,
):
    _global_only(multidim_average)
    return score_from_matrix(
        multiclass_matrix(preds, target, num_classes, ignore_index, top_k),
        "recall", average, zero_division,
    )


def binary_specificity(
    preds, target, threshold=0.5, multidim_average="global",
    ignore_index=None, validate_args=True,
):
    _global_only(multidim_average)
    matrix = binary_matrix(preds, target, threshold, ignore_index)
    return binary_score_from_matrix(matrix, "specificity")


def multiclass_specificity(
    preds, target, num_classes, average="macro", top_k=1,
    multidim_average="global", ignore_index=None, validate_args=True,
):
    _global_only(multidim_average)
    return score_from_matrix(
        multiclass_matrix(preds, target, num_classes, ignore_index, top_k),
        "specificity", average,
    )


def binary_f1_score(
    preds, target, threshold=0.5, multidim_average="global",
    ignore_index=None, validate_args=True, zero_division=0,
):
    _global_only(multidim_average)
    matrix = binary_matrix(preds, target, threshold, ignore_index)
    return binary_score_from_matrix(matrix, "f1", zero_division)


def multiclass_f1_score(
    preds, target, num_classes, average="macro", top_k=1,
    multidim_average="global", ignore_index=None, validate_args=True,
    zero_division=0,
):
    _global_only(multidim_average)
    return score_from_matrix(
        multiclass_matrix(preds, target, num_classes, ignore_index, top_k),
        "f1", average, zero_division,
    )


def binary_jaccard_index(
    preds, target, threshold=0.5, ignore_index=None, validate_args=True,
    zero_division=0.0,
):
    matrix = binary_matrix(preds, target, threshold, ignore_index)
    return binary_score_from_matrix(matrix, "jaccard", zero_division)


def multiclass_jaccard_index(
    preds, target, num_classes, average="macro", ignore_index=None,
    validate_args=True, zero_division=0.0,
):
    return score_from_matrix(
        multiclass_matrix(preds, target, num_classes, ignore_index),
        "jaccard", average, zero_division,
    )


def binary_confusion_matrix(
    preds, target, threshold=0.5, normalize=None, ignore_index=None,
    validate_args=True,
):
    return normalize_matrix(
        binary_matrix(preds, target, threshold, ignore_index), normalize
    )


def multiclass_confusion_matrix(
    preds, target, num_classes, normalize=None, ignore_index=None,
    validate_args=True,
):
    return normalize_matrix(
        multiclass_matrix(preds, target, num_classes, ignore_index), normalize
    )


def binary_matthews_corrcoef(
    preds, target, threshold=0.5, ignore_index=None, validate_args=True,
):
    return float(matthews(binary_matrix(preds, target, threshold, ignore_index)))


def multiclass_matthews_corrcoef(
    preds, target, num_classes, ignore_index=None, validate_args=True,
):
    return float(matthews(
        multiclass_matrix(preds, target, num_classes, ignore_index)
    ))


def binary_cohen_kappa(
    preds, target, threshold=0.5, weights=None, ignore_index=None,
    validate_args=True,
):
    return float(kappa(
        binary_matrix(preds, target, threshold, ignore_index), weights
    ))


def multiclass_cohen_kappa(
    preds, target, num_classes, weights=None, ignore_index=None,
    validate_args=True,
):
    return float(kappa(
        multiclass_matrix(preds, target, num_classes, ignore_index), weights
    ))


def binary_auroc(
    preds, target, max_fpr=None, thresholds=None, ignore_index=None,
    validate_args=True,
):
    if thresholds is not None:
        raise NotImplementedError("binned thresholds are not covered")
    value = float(binary_ranking(preds, target, ignore_index)[0])
    if max_fpr is not None and max_fpr != 1.0:
        raise NotImplementedError("partial AUROC is not covered")
    return value


def binary_average_precision(
    preds, target, thresholds=None, ignore_index=None, validate_args=True,
):
    if thresholds is not None:
        raise NotImplementedError("binned thresholds are not covered")
    return float(binary_ranking(preds, target, ignore_index)[1])


def accuracy(preds, target, task, threshold=0.5, num_classes=None, **kwargs):
    if task == "binary":
        return binary_accuracy(preds, target, threshold=threshold, **kwargs)
    if task == "multiclass":
        return multiclass_accuracy(
            preds, target, num_classes=num_classes, **kwargs
        )
    raise ValueError("task must be 'binary' or 'multiclass'")


def _dispatch(name, preds, target, task, threshold=0.5, num_classes=None, **kwargs):
    if task == "binary":
        return globals()[f"binary_{name}"](
            preds, target, threshold=threshold, **kwargs
        )
    if task == "multiclass":
        return globals()[f"multiclass_{name}"](
            preds, target, num_classes=num_classes, **kwargs
        )
    raise ValueError("task must be 'binary' or 'multiclass'")


def precision(preds, target, task, threshold=0.5, num_classes=None, **kwargs):
    return _dispatch("precision", preds, target, task, threshold, num_classes, **kwargs)


def recall(preds, target, task, threshold=0.5, num_classes=None, **kwargs):
    return _dispatch("recall", preds, target, task, threshold, num_classes, **kwargs)


def specificity(preds, target, task, threshold=0.5, num_classes=None, **kwargs):
    return _dispatch("specificity", preds, target, task, threshold, num_classes, **kwargs)


def f1_score(preds, target, task, threshold=0.5, num_classes=None, **kwargs):
    return _dispatch("f1_score", preds, target, task, threshold, num_classes, **kwargs)


def jaccard_index(preds, target, task, threshold=0.5, num_classes=None, **kwargs):
    return _dispatch("jaccard_index", preds, target, task, threshold, num_classes, **kwargs)


def confusion_matrix(preds, target, task, threshold=0.5, num_classes=None, **kwargs):
    return _dispatch("confusion_matrix", preds, target, task, threshold, num_classes, **kwargs)


def matthews_corrcoef(preds, target, task, threshold=0.5, num_classes=None, **kwargs):
    return _dispatch("matthews_corrcoef", preds, target, task, threshold, num_classes, **kwargs)


def cohen_kappa(preds, target, task, threshold=0.5, num_classes=None, **kwargs):
    return _dispatch("cohen_kappa", preds, target, task, threshold, num_classes, **kwargs)


def auroc(preds, target, task, **kwargs):
    if task != "binary":
        raise NotImplementedError("only binary AUROC is covered")
    return binary_auroc(preds, target, **kwargs)


def average_precision(preds, target, task, **kwargs):
    if task != "binary":
        raise NotImplementedError("only binary average precision is covered")
    return binary_average_precision(preds, target, **kwargs)


def _global_only(multidim_average):
    if multidim_average != "global":
        raise NotImplementedError("multidim_average='samplewise' is not covered")


__all__ = [name for name in globals() if not name.startswith("_")]
