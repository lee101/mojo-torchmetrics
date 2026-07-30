from __future__ import annotations

import numpy as np

from ._classification import (
    array,
    integer_array,
    binary_matrix,
    binary_ranking,
    binary_score_from_matrix,
    kappa,
    matthews,
    multiclass_matrix,
    normalize_matrix,
    score_from_matrix,
)
from .metric import Metric


class _BinaryConfusionMetric(Metric):
    metric = None

    def __init__(
        self, threshold=0.5, multidim_average="global", ignore_index=None,
        validate_args=True, zero_division=0, **kwargs,
    ):
        if multidim_average != "global":
            raise NotImplementedError("multidim_average='samplewise' is not covered")
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.zero_division = zero_division
        self.reset()

    def update(self, preds, target):
        self._matrix += binary_matrix(
            preds, target, self.threshold, self.ignore_index
        )
        return self

    def compute(self):
        return binary_score_from_matrix(
            self._matrix, self.metric, self.zero_division
        )

    def reset(self):
        self._matrix = np.zeros((2, 2), dtype=np.float64)
        return self


class BinaryAccuracy(_BinaryConfusionMetric):
    metric = "accuracy"


class BinaryPrecision(_BinaryConfusionMetric):
    metric = "precision"


class BinaryRecall(_BinaryConfusionMetric):
    metric = "recall"


class BinarySpecificity(_BinaryConfusionMetric):
    metric = "specificity"


class BinaryF1Score(_BinaryConfusionMetric):
    metric = "f1"


class BinaryJaccardIndex(_BinaryConfusionMetric):
    metric = "jaccard"


class BinaryConfusionMatrix(_BinaryConfusionMetric):
    def __init__(
        self, threshold=0.5, ignore_index=None, normalize=None,
        validate_args=True, **kwargs,
    ):
        self.normalize = normalize
        super().__init__(
            threshold=threshold, ignore_index=ignore_index,
            validate_args=validate_args, **kwargs,
        )

    def compute(self):
        return normalize_matrix(self._matrix.copy(), self.normalize)


class BinaryMatthewsCorrCoef(_BinaryConfusionMetric):
    def compute(self):
        return float(matthews(self._matrix))


class BinaryCohenKappa(_BinaryConfusionMetric):
    def __init__(
        self, threshold=0.5, ignore_index=None, weights=None,
        validate_args=True, **kwargs,
    ):
        self.weights = weights
        super().__init__(
            threshold=threshold, ignore_index=ignore_index,
            validate_args=validate_args, **kwargs,
        )

    def compute(self):
        return float(kappa(self._matrix, self.weights))


class _MulticlassConfusionMetric(Metric):
    metric = None

    def __init__(
        self, num_classes=None, top_k=1, average="macro",
        multidim_average="global", ignore_index=None, validate_args=True,
        zero_division=0, **kwargs,
    ):
        if num_classes is None:
            raise ValueError("num_classes is required")
        if multidim_average != "global":
            raise NotImplementedError("multidim_average='samplewise' is not covered")
        self.num_classes = int(num_classes)
        self.top_k = top_k
        self.average = average
        self.ignore_index = ignore_index
        self.zero_division = zero_division
        self.reset()

    def update(self, preds, target):
        self._matrix += multiclass_matrix(
            preds, target, self.num_classes, self.ignore_index, self.top_k
        )
        return self

    def compute(self):
        return score_from_matrix(
            self._matrix, self.metric, self.average, self.zero_division
        )

    def reset(self):
        self._matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.float64
        )
        return self


class MulticlassAccuracy(_MulticlassConfusionMetric):
    metric = "accuracy"


class MulticlassPrecision(_MulticlassConfusionMetric):
    metric = "precision"


class MulticlassRecall(_MulticlassConfusionMetric):
    metric = "recall"


class MulticlassSpecificity(_MulticlassConfusionMetric):
    metric = "specificity"


class MulticlassF1Score(_MulticlassConfusionMetric):
    metric = "f1"


class MulticlassJaccardIndex(_MulticlassConfusionMetric):
    metric = "jaccard"


class MulticlassConfusionMatrix(_MulticlassConfusionMetric):
    def __init__(
        self, num_classes, ignore_index=None, normalize=None,
        validate_args=True, **kwargs,
    ):
        self.normalize = normalize
        super().__init__(
            num_classes=num_classes, ignore_index=ignore_index,
            validate_args=validate_args, **kwargs,
        )

    def compute(self):
        return normalize_matrix(self._matrix.copy(), self.normalize)


class MulticlassMatthewsCorrCoef(_MulticlassConfusionMetric):
    def compute(self):
        return float(matthews(self._matrix))


class MulticlassCohenKappa(_MulticlassConfusionMetric):
    def __init__(
        self, num_classes, ignore_index=None, weights=None,
        validate_args=True, **kwargs,
    ):
        self.weights = weights
        super().__init__(
            num_classes=num_classes, ignore_index=ignore_index,
            validate_args=validate_args, **kwargs,
        )

    def compute(self):
        return float(kappa(self._matrix, self.weights))


class _BinaryRankingMetric(Metric):
    index = 0

    def __init__(
        self, thresholds=None, ignore_index=None, validate_args=True, **kwargs,
    ):
        if thresholds is not None:
            raise NotImplementedError("binned thresholds are not covered")
        self.ignore_index = ignore_index
        self.reset()

    def update(self, preds, target):
        self._preds.append(np.array(array(preds, np.float64).reshape(-1), copy=True))
        self._target.append(np.array(integer_array(target, "target").reshape(-1), copy=True))
        return self

    def compute(self):
        if not self._preds:
            raise RuntimeError("compute() called before update()")
        return float(binary_ranking(
            np.concatenate(self._preds), np.concatenate(self._target),
            self.ignore_index,
        )[self.index])

    def reset(self):
        self._preds = []
        self._target = []
        return self


class BinaryAUROC(_BinaryRankingMetric):
    index = 0

    def __init__(
        self, max_fpr=None, thresholds=None, ignore_index=None,
        validate_args=True, **kwargs,
    ):
        if max_fpr not in (None, 1.0):
            raise NotImplementedError("partial AUROC is not covered")
        self.max_fpr = max_fpr
        super().__init__(thresholds, ignore_index, validate_args, **kwargs)


class BinaryAveragePrecision(_BinaryRankingMetric):
    index = 1


def _factory(binary, multiclass):
    class Factory:
        def __new__(cls, task, **kwargs):
            if task == "binary":
                return binary(**kwargs)
            if task == "multiclass":
                return multiclass(**kwargs)
            raise ValueError("task must be 'binary' or 'multiclass'")
    return Factory


Accuracy = _factory(BinaryAccuracy, MulticlassAccuracy)
Precision = _factory(BinaryPrecision, MulticlassPrecision)
Recall = _factory(BinaryRecall, MulticlassRecall)
Specificity = _factory(BinarySpecificity, MulticlassSpecificity)
F1Score = _factory(BinaryF1Score, MulticlassF1Score)
JaccardIndex = _factory(BinaryJaccardIndex, MulticlassJaccardIndex)
ConfusionMatrix = _factory(BinaryConfusionMatrix, MulticlassConfusionMatrix)
MatthewsCorrCoef = _factory(BinaryMatthewsCorrCoef, MulticlassMatthewsCorrCoef)
CohenKappa = _factory(BinaryCohenKappa, MulticlassCohenKappa)


class AUROC:
    def __new__(cls, task, **kwargs):
        if task != "binary":
            raise NotImplementedError("only binary AUROC is covered")
        return BinaryAUROC(**kwargs)


class AveragePrecision:
    def __new__(cls, task, **kwargs):
        if task != "binary":
            raise NotImplementedError("only binary average precision is covered")
        return BinaryAveragePrecision(**kwargs)


__all__ = [
    name for name in globals()
    if name[0].isupper() and not name.startswith("_") and name not in {"Metric"}
]
