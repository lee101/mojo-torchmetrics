"""Benchmark Mojo kernels against TorchMetrics on identical CPU data."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
import torch
import torchmetrics
import torchmetrics.functional.classification as tc
import torchmetrics.functional.regression as tr


sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import mojo_torchmetrics.functional.classification as mc  # noqa: E402
import mojo_torchmetrics.functional.regression as mr  # noqa: E402


def timeit(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cases():
    rng = np.random.default_rng(42)

    binary_scores = np.ascontiguousarray(rng.random(5_000_000))
    binary_target = np.ascontiguousarray(
        rng.integers(0, 2, 5_000_000, dtype=np.int64)
    )
    binary_scores_t = torch.from_numpy(binary_scores)
    binary_target_t = torch.from_numpy(binary_target)
    yield (
        "BinaryAccuracy (5M)",
        lambda: mc.binary_accuracy(binary_scores, binary_target),
        lambda: tc.binary_accuracy(binary_scores_t, binary_target_t),
    )

    class_preds = np.ascontiguousarray(
        rng.integers(0, 20, 3_000_000, dtype=np.int64)
    )
    class_target = np.ascontiguousarray(
        rng.integers(0, 20, 3_000_000, dtype=np.int64)
    )
    class_preds_t = torch.from_numpy(class_preds)
    class_target_t = torch.from_numpy(class_target)
    yield (
        "MulticlassF1 macro (3M, 20 classes)",
        lambda: mc.multiclass_f1_score(
            class_preds, class_target, num_classes=20
        ),
        lambda: tc.multiclass_f1_score(
            class_preds_t, class_target_t, num_classes=20
        ),
    )
    yield (
        "MulticlassConfusionMatrix (3M, 20 classes)",
        lambda: mc.multiclass_confusion_matrix(
            class_preds, class_target, num_classes=20
        ),
        lambda: tc.multiclass_confusion_matrix(
            class_preds_t, class_target_t, num_classes=20
        ),
    )

    rank_scores = binary_scores[:1_000_000]
    rank_target = binary_target[:1_000_000]
    rank_scores_t = binary_scores_t[:1_000_000]
    rank_target_t = binary_target_t[:1_000_000]
    yield (
        "BinaryAUROC exact (1M)",
        lambda: mc.binary_auroc(rank_scores, rank_target),
        lambda: tc.binary_auroc(rank_scores_t, rank_target_t),
    )

    prediction = np.ascontiguousarray(rng.normal(size=5_000_000))
    target = np.ascontiguousarray(
        prediction + rng.normal(scale=0.5, size=5_000_000)
    )
    prediction_t = torch.from_numpy(prediction)
    target_t = torch.from_numpy(target)
    yield (
        "MeanSquaredError (5M)",
        lambda: mr.mean_squared_error(prediction, target),
        lambda: tr.mean_squared_error(prediction_t, target_t),
    )
    yield (
        "R2Score (5M)",
        lambda: mr.r2_score(prediction, target),
        lambda: tr.r2_score(prediction_t, target_t),
    )
    yield (
        "PearsonCorrCoef (5M)",
        lambda: mr.pearson_corrcoef(prediction, target),
        lambda: tr.pearson_corrcoef(prediction_t, target_t),
    )


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    print(f"Machine: {cpu_name()}; {os.cpu_count()} logical CPUs; {platform.platform()}")
    print(
        f"Software: Mojo 1.1.0.dev2026081105; "
        f"TorchMetrics {torchmetrics.__version__}; PyTorch {torch.__version__}"
    )
    print()
    print("| case | mojo-torchmetrics | torchmetrics | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, theirs in cases():
        ours()
        theirs()
        mojo_seconds = timeit(ours)
        torch_seconds = timeit(theirs)
        ratio = torch_seconds / mojo_seconds
        result = (
            f"{ratio:.2f}x faster"
            if ratio >= 1
            else f"{1 / ratio:.2f}x slower"
        )
        print(
            f"| {name} | {mojo_seconds * 1e3:.2f} ms | "
            f"{torch_seconds * 1e3:.2f} ms | {result} |"
        )


if __name__ == "__main__":
    main()
