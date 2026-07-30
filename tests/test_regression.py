from __future__ import annotations

import numpy as np
import pytest
import torch
import torchmetrics.functional.regression as tf
import torchmetrics.regression as tr

import mojo_torchmetrics.functional.regression as mf
import mojo_torchmetrics.regression as mr
from mojo_torchmetrics._regression import R2_PARALLEL_THRESHOLD


SCALAR_FUNCTIONS = [
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "explained_variance",
    "mean_absolute_percentage_error",
    "symmetric_mean_absolute_percentage_error",
    "weighted_mean_absolute_percentage_error",
    "mean_squared_log_error",
    "pearson_corrcoef",
    "concordance_corrcoef",
]


@pytest.mark.parametrize("function", SCALAR_FUNCTIONS)
def test_scalar_regression_functional(function):
    preds = np.array([1.2, 2.8, 4.1, 8.0, 5.5, 2.0])
    target = np.array([1.0, 3.0, 3.5, 7.0, 6.0, 2.5])
    ours = getattr(mf, function)(preds, target)
    theirs = getattr(tf, function)(
        torch.tensor(preds), torch.tensor(target)
    ).numpy()
    assert np.allclose(ours, theirs, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize(
    "function", ["mean_squared_error", "mean_absolute_error"]
)
def test_multioutput_error(function):
    preds = np.array([[1.0, 2.0], [3.0, 5.0], [8.0, 4.0]])
    target = np.array([[0.0, 2.0], [4.0, 1.0], [7.0, 6.0]])
    ours = getattr(mf, function)(preds, target, num_outputs=2)
    theirs = getattr(tf, function)(
        torch.tensor(preds), torch.tensor(target), num_outputs=2
    ).numpy()
    assert np.allclose(ours, theirs, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("function", ["r2_score", "explained_variance"])
@pytest.mark.parametrize(
    "multioutput", ["raw_values", "uniform_average", "variance_weighted"]
)
def test_multioutput_variance_metrics(function, multioutput):
    preds = np.array([[1.0, 2.0], [3.0, 5.0], [8.0, 4.0], [2.0, 7.0]])
    target = np.array([[0.0, 2.0], [4.0, 1.0], [7.0, 6.0], [3.0, 8.0]])
    ours = getattr(mf, function)(
        preds, target, multioutput=multioutput
    )
    theirs = getattr(tf, function)(
        torch.tensor(preds), torch.tensor(target), multioutput=multioutput
    ).numpy()
    assert np.allclose(ours, theirs, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize(
    "ours_class,upstream_class",
    [
        (mr.MeanSquaredError, tr.MeanSquaredError),
        (mr.MeanAbsoluteError, tr.MeanAbsoluteError),
        (mr.R2Score, tr.R2Score),
        (mr.ExplainedVariance, tr.ExplainedVariance),
        (mr.MeanAbsolutePercentageError, tr.MeanAbsolutePercentageError),
        (
            mr.SymmetricMeanAbsolutePercentageError,
            tr.SymmetricMeanAbsolutePercentageError,
        ),
        (
            mr.WeightedMeanAbsolutePercentageError,
            tr.WeightedMeanAbsolutePercentageError,
        ),
        (mr.MeanSquaredLogError, tr.MeanSquaredLogError),
        (mr.PearsonCorrCoef, tr.PearsonCorrCoef),
        (mr.ConcordanceCorrCoef, tr.ConcordanceCorrCoef),
    ],
)
def test_stateful_regression_batches(ours_class, upstream_class):
    preds = np.array([1.2, 2.8, 4.1, 8.0, 5.5, 2.0])
    target = np.array([1.0, 3.0, 3.5, 7.0, 6.0, 2.5])
    ours, theirs = ours_class(), upstream_class()
    for indices in (slice(0, 2), slice(2, None)):
        ours.update(preds[indices], target[indices])
        theirs.update(
            torch.tensor(preds[indices]), torch.tensor(target[indices])
        )
    assert np.allclose(
        ours.compute(), theirs.compute().numpy(), rtol=2e-6, atol=2e-7
    )


def test_r2_adjusted():
    preds = np.array([1.0, 2.0, 4.0, 8.0, 6.0, 3.0])
    target = np.array([1.5, 2.5, 3.0, 7.0, 5.0, 4.0])
    assert mf.r2_score(preds, target, adjusted=2) == pytest.approx(
        tf.r2_score(
            torch.tensor(preds), torch.tensor(target), adjusted=2
        ).item()
    )


@pytest.mark.parametrize("size", [11, R2_PARALLEL_THRESHOLD + 3])
def test_r2_simd_tail_and_parallel_threshold(size):
    rng = np.random.default_rng(123)
    target = rng.normal(size=size)
    preds = target + rng.normal(scale=0.2, size=size)
    ours = mf.r2_score(preds, target)
    theirs = tf.r2_score(
        torch.from_numpy(preds), torch.from_numpy(target)
    ).item()
    assert ours == pytest.approx(theirs, rel=2e-6, abs=2e-7)


def test_reset_and_empty_compute():
    metric = mr.MeanSquaredError()
    with pytest.raises(RuntimeError):
        metric.compute()
    metric.update([1.0, 2.0], [0.0, 2.0])
    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_regression_input_rejected_before_ffi(bad):
    with pytest.raises(ValueError):
        mf.r2_score([1.0, bad], [1.0, 2.0])
    with pytest.raises(ValueError):
        mf.mean_squared_error([1.0, bad], [1.0, 2.0])


def test_complex_regression_input_is_not_silently_narrowed():
    with pytest.raises(TypeError):
        mf.mean_squared_error([1 + 2j], [1.0])


def test_regression_shapes_must_match_before_flattening():
    with pytest.raises(ValueError):
        mf.mean_squared_error(np.ones((2, 2)), np.ones(4))


def test_num_outputs_rejects_silent_narrowing():
    with pytest.raises(ValueError):
        mf.mean_absolute_error(np.ones(4), np.ones(4), num_outputs=1.5)
