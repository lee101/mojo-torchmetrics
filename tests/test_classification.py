from __future__ import annotations

import numpy as np
import pytest
import torch
import torchmetrics.classification as tc
import torchmetrics.functional.classification as tf

import mojo_torchmetrics.classification as mc
import mojo_torchmetrics.functional.classification as mf


BINARY_FUNCTIONS = [
    "binary_accuracy",
    "binary_precision",
    "binary_recall",
    "binary_specificity",
    "binary_f1_score",
    "binary_jaccard_index",
    "binary_matthews_corrcoef",
    "binary_cohen_kappa",
    "binary_auroc",
    "binary_average_precision",
]

MULTICLASS_FUNCTIONS = [
    "multiclass_accuracy",
    "multiclass_precision",
    "multiclass_recall",
    "multiclass_specificity",
    "multiclass_f1_score",
    "multiclass_jaccard_index",
    "multiclass_matthews_corrcoef",
    "multiclass_cohen_kappa",
]


def upstream(function, preds, target, **kwargs):
    value = getattr(tf, function)(
        torch.as_tensor(preds), torch.as_tensor(target), **kwargs
    )
    return value.detach().cpu().numpy()


@pytest.mark.parametrize("function", BINARY_FUNCTIONS)
def test_binary_functional_parity(function):
    preds = np.array([0.1, 0.9, 0.7, 0.2, 0.8, 0.6, 0.4, 0.5])
    target = np.array([0, 1, 0, 0, 1, 1, 1, 0])
    ours = getattr(mf, function)(preds, target)
    theirs = upstream(function, preds, target)
    assert np.allclose(ours, theirs, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("function", BINARY_FUNCTIONS[:6])
def test_binary_logits_and_ignore_index(function):
    preds = np.array([-3.0, 2.0, 0.8, -1.0, 4.0, 0.2, -0.5])
    target = np.array([0, 1, -1, 0, 1, 1, 0])
    kwargs = {"ignore_index": -1, "threshold": 0.65}
    ours = getattr(mf, function)(preds, target, **kwargs)
    theirs = upstream(function, preds, target, **kwargs)
    assert ours == pytest.approx(theirs, rel=2e-6, abs=2e-7)


@pytest.mark.parametrize("normalize", [None, "true", "pred", "all"])
def test_binary_confusion_matrix(normalize):
    preds = np.array([0.2, 0.7, 0.6, 0.1, 0.9, 0.3])
    target = np.array([0, 1, 0, 0, 1, 1])
    ours = mf.binary_confusion_matrix(preds, target, normalize=normalize)
    theirs = upstream(
        "binary_confusion_matrix", preds, target, normalize=normalize
    )
    assert np.allclose(ours, theirs)


@pytest.mark.parametrize("function", MULTICLASS_FUNCTIONS)
@pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
def test_multiclass_functional_parity(function, average):
    preds = np.array([0, 2, 1, 2, 0, 3, 3, 1, 0, 2, 3])
    target = np.array([0, 1, 1, 2, 3, 3, 2, 1, 0, 2, 0])
    kwargs = {"num_classes": 4}
    if function not in {
        "multiclass_matthews_corrcoef",
        "multiclass_cohen_kappa",
    }:
        kwargs["average"] = average
    ours = getattr(mf, function)(preds, target, **kwargs)
    theirs = upstream(function, preds, target, **kwargs)
    assert np.allclose(ours, theirs, rtol=2e-6, atol=2e-7)


def test_multiclass_logits_parity():
    rng = np.random.default_rng(8)
    preds = rng.normal(size=(1000, 5))
    target = rng.integers(0, 5, size=1000)
    ours = mf.multiclass_accuracy(preds, target, num_classes=5)
    theirs = upstream(
        "multiclass_accuracy", preds, target, num_classes=5
    )
    assert ours == pytest.approx(theirs, rel=2e-6)


@pytest.mark.parametrize("normalize", [None, "true", "pred", "all"])
def test_multiclass_confusion_matrix(normalize):
    preds = np.array([0, 2, 1, 2, 0, 3, 3, 1, 0])
    target = np.array([0, 1, 1, 2, 3, 3, 2, 1, 0])
    ours = mf.multiclass_confusion_matrix(
        preds, target, 4, normalize=normalize
    )
    theirs = upstream(
        "multiclass_confusion_matrix", preds, target,
        num_classes=4, normalize=normalize,
    )
    assert np.allclose(ours, theirs)


@pytest.mark.parametrize("weights", [None, "linear", "quadratic"])
def test_weighted_kappa(weights):
    preds = np.array([0, 2, 1, 2, 0, 3, 3, 1, 0])
    target = np.array([0, 1, 1, 2, 3, 3, 2, 1, 0])
    ours = mf.multiclass_cohen_kappa(preds, target, 4, weights=weights)
    theirs = upstream(
        "multiclass_cohen_kappa", preds, target,
        num_classes=4, weights=weights,
    )
    assert ours == pytest.approx(theirs, rel=2e-6)


@pytest.mark.parametrize(
    "function", ["binary_auroc", "binary_average_precision"]
)
def test_ranking_ties(function):
    preds = np.array([0.8, 0.8, 0.5, 0.5, 0.5, 0.1, 0.1])
    target = np.array([1, 0, 1, 0, 1, 0, 1])
    ours = getattr(mf, function)(preds, target)
    theirs = upstream(function, preds, target)
    assert ours == pytest.approx(theirs, rel=2e-6)


def test_ranking_sort_random_input():
    rng = np.random.default_rng(321)
    preds = rng.random(10_003)
    target = rng.integers(0, 2, size=10_003)
    assert mf.binary_auroc(preds, target) == pytest.approx(
        upstream("binary_auroc", preds, target), rel=2e-6
    )


@pytest.mark.parametrize(
    "ours_class,upstream_class",
    [
        (mc.BinaryAccuracy, tc.BinaryAccuracy),
        (mc.BinaryF1Score, tc.BinaryF1Score),
        (mc.BinaryMatthewsCorrCoef, tc.BinaryMatthewsCorrCoef),
        (mc.BinaryAUROC, tc.BinaryAUROC),
        (mc.BinaryAveragePrecision, tc.BinaryAveragePrecision),
    ],
)
def test_binary_stateful_batches(ours_class, upstream_class):
    preds = np.array([0.1, 0.9, 0.7, 0.2, 0.8, 0.6, 0.4, 0.5])
    target = np.array([0, 1, 0, 0, 1, 1, 1, 0])
    ours, theirs = ours_class(), upstream_class()
    for indices in (slice(0, 3), slice(3, None)):
        ours.update(preds[indices], target[indices])
        theirs.update(
            torch.as_tensor(preds[indices]), torch.as_tensor(target[indices])
        )
    assert ours.compute() == pytest.approx(theirs.compute().item(), rel=2e-6)
    ours.reset()
    assert ours(preds, target) == pytest.approx(
        upstream_class()(torch.as_tensor(preds), torch.as_tensor(target)).item(),
        rel=2e-6,
    )


def test_multiclass_stateful_and_factory():
    preds = np.array([0, 2, 1, 2, 0, 3, 3, 1, 0])
    target = np.array([0, 1, 1, 2, 3, 3, 2, 1, 0])
    ours = mc.F1Score(task="multiclass", num_classes=4, average="weighted")
    theirs = tc.F1Score(task="multiclass", num_classes=4, average="weighted")
    ours.update(preds[:4], target[:4])
    ours.update(preds[4:], target[4:])
    theirs.update(torch.tensor(preds[:4]), torch.tensor(target[:4]))
    theirs.update(torch.tensor(preds[4:]), torch.tensor(target[4:]))
    assert ours.compute() == pytest.approx(theirs.compute().item(), rel=2e-6)


@pytest.mark.parametrize(
    "name",
    [
        "BinaryAccuracy",
        "BinaryPrecision",
        "BinaryRecall",
        "BinarySpecificity",
        "BinaryF1Score",
        "BinaryJaccardIndex",
        "BinaryConfusionMatrix",
        "BinaryMatthewsCorrCoef",
        "BinaryCohenKappa",
    ],
)
def test_all_binary_confusion_classes(name):
    preds = np.array([0.1, 0.9, 0.7, 0.2, 0.8, 0.6, 0.4])
    target = np.array([0, 1, 0, 0, 1, 1, 1])
    ours, theirs = getattr(mc, name)(), getattr(tc, name)()
    ours.update(preds[:3], target[:3])
    ours.update(preds[3:], target[3:])
    theirs.update(torch.tensor(preds[:3]), torch.tensor(target[:3]))
    theirs.update(torch.tensor(preds[3:]), torch.tensor(target[3:]))
    assert np.allclose(ours.compute(), theirs.compute().numpy(), rtol=2e-6)


@pytest.mark.parametrize(
    "name",
    [
        "MulticlassAccuracy",
        "MulticlassPrecision",
        "MulticlassRecall",
        "MulticlassSpecificity",
        "MulticlassF1Score",
        "MulticlassJaccardIndex",
        "MulticlassConfusionMatrix",
        "MulticlassMatthewsCorrCoef",
        "MulticlassCohenKappa",
    ],
)
def test_all_multiclass_classes(name):
    preds = np.array([0, 2, 1, 2, 0, 3, 3, 1, 0])
    target = np.array([0, 1, 1, 2, 3, 3, 2, 1, 0])
    kwargs = {"num_classes": 4}
    ours, theirs = getattr(mc, name)(**kwargs), getattr(tc, name)(**kwargs)
    ours.update(preds[:4], target[:4])
    ours.update(preds[4:], target[4:])
    theirs.update(torch.tensor(preds[:4]), torch.tensor(target[:4]))
    theirs.update(torch.tensor(preds[4:]), torch.tensor(target[4:]))
    assert np.allclose(ours.compute(), theirs.compute().numpy(), rtol=2e-6)


def test_generic_functional_factories():
    preds = [0.2, 0.8, 0.7, 0.1]
    target = [0, 1, 0, 0]
    assert mf.accuracy(preds, target, task="binary") == pytest.approx(
        tf.accuracy(
            torch.tensor(preds), torch.tensor(target), task="binary"
        ).item()
    )


@pytest.mark.parametrize("target", [[0.0, 0.5], [0.0, np.nan], [0, 2**70]])
def test_binary_target_rejects_silent_integer_narrowing(target):
    with pytest.raises((ValueError, OverflowError)):
        mf.binary_accuracy([0.1, 0.9], target)


def test_multiclass_predictions_reject_silent_integer_narrowing():
    with pytest.raises(ValueError):
        mf.multiclass_accuracy([0.0, 1.5], [0, 1], num_classes=2)


def test_multiclass_score_dimension_is_validated():
    with pytest.raises(ValueError):
        mf.multiclass_accuracy(
            np.ones((2, 3)), np.array([0, 1]), num_classes=2
        )


def test_nonfinite_predictions_are_rejected_before_ffi():
    with pytest.raises(ValueError):
        mf.binary_accuracy([0.1, np.nan], [0, 1])
