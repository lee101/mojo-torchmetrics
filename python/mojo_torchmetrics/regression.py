from __future__ import annotations

import numpy as np

from ._regression import (
    concordance_from,
    explained_variance_from,
    pearson_from,
    r2_from,
    r2_reductions,
    regression_reductions,
)
from .metric import Metric


class _RegressionMetric(Metric):
    num_outputs = None
    include_log = False

    def update(self, preds, target):
        current = regression_reductions(
            preds, target, self.num_outputs, self.include_log
        )
        if getattr(self, "_reductions", None) is None:
            self._reductions = current
        elif self._reductions.shape != current.shape:
            raise ValueError("the number of outputs changed between updates")
        else:
            self._reductions += current
        return self

    def _state(self):
        if getattr(self, "_reductions", None) is None:
            raise RuntimeError("compute() called before update()")
        return self._reductions

    def reset(self):
        self._reductions = None
        return self


class MeanSquaredError(_RegressionMetric):
    def __init__(self, squared=True, num_outputs=1, **kwargs):
        self.squared = squared
        self.num_outputs = num_outputs
        self.reset()

    def compute(self):
        state = self._state()
        result = state[:, 1] / state[:, 0]
        if not self.squared:
            result = np.sqrt(result)
        return float(result[0]) if result.size == 1 else result


class MeanAbsoluteError(_RegressionMetric):
    def __init__(self, num_outputs=1, **kwargs):
        self.num_outputs = num_outputs
        self.reset()

    def compute(self):
        state = self._state()
        result = state[:, 2] / state[:, 0]
        return float(result[0]) if result.size == 1 else result


class R2Score(_RegressionMetric):
    def __init__(self, adjusted=0, multioutput="uniform_average", **kwargs):
        self.adjusted = adjusted
        self.multioutput = multioutput
        self.num_outputs = None
        self.reset()

    def compute(self):
        return r2_from(self._state(), self.adjusted, self.multioutput)

    def update(self, preds, target):
        current = r2_reductions(preds, target)
        if getattr(self, "_reductions", None) is None:
            self._reductions = current
        elif self._reductions.shape != current.shape:
            raise ValueError("the number of outputs changed between updates")
        else:
            self._reductions += current
        return self


class ExplainedVariance(_RegressionMetric):
    def __init__(self, multioutput="uniform_average", **kwargs):
        self.multioutput = multioutput
        self.num_outputs = None
        self.reset()

    def compute(self):
        return explained_variance_from(self._state(), self.multioutput)


class MeanAbsolutePercentageError(_RegressionMetric):
    def __init__(self, **kwargs):
        self.num_outputs = 1
        self.reset()

    def compute(self):
        state = self._state()[0]
        return float(state[3] / state[0])


class SymmetricMeanAbsolutePercentageError(_RegressionMetric):
    def __init__(self, **kwargs):
        self.num_outputs = 1
        self.reset()

    def compute(self):
        state = self._state()[0]
        return float(state[4] / state[0])


class WeightedMeanAbsolutePercentageError(_RegressionMetric):
    def __init__(self, **kwargs):
        self.num_outputs = 1
        self.reset()

    def compute(self):
        state = self._state()[0]
        return float(state[2] / state[5]) if state[5] else 0.0


class MeanSquaredLogError(_RegressionMetric):
    include_log = True

    def __init__(self, **kwargs):
        self.num_outputs = 1
        self.reset()

    def compute(self):
        state = self._state()[0]
        if np.isnan(state[12]):
            raise ValueError("preds and target must be greater than -1")
        return float(state[12] / state[0])


class PearsonCorrCoef(_RegressionMetric):
    def __init__(self, num_outputs=1, **kwargs):
        self.num_outputs = num_outputs
        self.reset()

    def compute(self):
        return pearson_from(self._state())


class ConcordanceCorrCoef(_RegressionMetric):
    def __init__(self, num_outputs=1, **kwargs):
        self.num_outputs = num_outputs
        self.reset()

    def compute(self):
        return concordance_from(self._state())


__all__ = [
    "ConcordanceCorrCoef",
    "ExplainedVariance",
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    "MeanSquaredError",
    "MeanSquaredLogError",
    "PearsonCorrCoef",
    "R2Score",
    "SymmetricMeanAbsolutePercentageError",
    "WeightedMeanAbsolutePercentageError",
]
