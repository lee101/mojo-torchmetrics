# mojo-torchmetrics

`mojo-torchmetrics` is a standalone Mojo implementation of the compute-heavy
classification and regression core of
[TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/). Its Python API
uses the same covered class and functional names and accepts NumPy arrays or
CPU PyTorch tensors.

The project is useful when metrics run on large CPU-resident arrays. It is not
a replacement for TorchMetrics' distributed synchronization or GPU-native
tensor integration: results are Python floats or NumPy arrays, and `.to()` is
a compatibility no-op.

## Covered subset

| area | functional API and stateful classes |
| --- | --- |
| Binary classification | accuracy, precision, recall, specificity, F1, Jaccard, confusion matrix, Matthews correlation, Cohen's kappa, exact AUROC, exact average precision |
| Multiclass classification | accuracy, precision, recall, specificity, F1, Jaccard, confusion matrix, Matthews correlation, Cohen's kappa |
| Regression | MSE/RMSE, MAE, R2, explained variance, MAPE, symmetric MAPE, weighted MAPE, MSLE, Pearson correlation, concordance correlation |
| Stateful operation | `update`, `compute`, `reset`, `forward`, `clone`, and callable metric objects; sufficient statistics accumulate across batches |

Classification reductions support `micro`, `macro`, `weighted`, and unaggregated
scores where TorchMetrics exposes them, normalized confusion matrices, ignored
targets, probability thresholds, binary logits, and multiclass score matrices.
The task factories (`Accuracy(task=...)`, `F1Score(task=...)`, and their
functional equivalents) are also covered for binary and multiclass tasks.

Not covered are multilabel metrics, multiclass/multilabel AUROC and average
precision, `top_k > 1`, samplewise multidimensional averaging, binned ranking
thresholds, partial AUROC, distributed state synchronization, CUDA tensors,
plots, metric collections, wrappers, image/audio/text metrics, and regression
metrics outside the table.

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi install` provides the pinned Mojo/MAX nightly, Python, NumPy, PyTorch,
and TorchMetrics 1.9.0. The build task writes
`dist/libmojo-torchmetrics.so`. Importing the Python package also rebuilds the
library when `src/kernels.mojo` is newer.

For packaging outside this checkout, build the library and set
`MOJO_TORCHMETRICS_LIB` to its absolute path.

## Usage

```python
import numpy as np
from mojo_torchmetrics.classification import BinaryAUROC
from mojo_torchmetrics.functional import mean_squared_error

scores = np.array([0.05, 0.90, 0.70, 0.20, 0.80])
target = np.array([0, 1, 0, 0, 1])

auroc = BinaryAUROC()
auroc.update(scores[:3], target[:3])
auroc.update(scores[3:], target[3:])
print(auroc.compute())

prediction = np.array([2.5, 0.0, 2.0, 8.0])
actual = np.array([3.0, -0.5, 2.0, 7.0])
print(mean_squared_error(prediction, actual))
```

The same names are available from
`mojo_torchmetrics.functional.classification` and
`mojo_torchmetrics.functional.regression`, matching TorchMetrics' module
layout.

## Performance

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz
(72 logical CPUs), Linux 6.8.0-136, Mojo
1.1.0.dev2026081105, TorchMetrics 1.9.0, and PyTorch 2.13.0. Each row uses
identical pre-created CPU data and reports the best of three warm runs.

| case | mojo-torchmetrics | torchmetrics | result |
| --- | ---: | ---: | ---: |
| BinaryAccuracy (5M) | 160.26 ms | 221.57 ms | 1.38x faster |
| MulticlassF1 macro (3M, 20 classes) | 46.83 ms | 125.69 ms | 2.68x faster |
| MulticlassConfusionMatrix (3M, 20 classes) | 35.67 ms | 115.59 ms | 3.24x faster |
| BinaryAUROC exact (1M) | 106.34 ms | 131.36 ms | 1.24x faster |
| MeanSquaredError (5M) | 9.93 ms | 13.59 ms | 1.37x faster |
| R2Score (5M) | 18.01 ms | 22.26 ms | 1.24x faster |
| PearsonCorrCoef (5M) | 44.54 ms | 81.42 ms | 1.83x faster |

Exact ranking uses Mojo's optimized in-place sort over a single score/target
scratch buffer. MSE and MAE share a dedicated SIMD reduction that also checks
input finiteness in the same memory pass. It splits inputs of at least one
million elements across eight CPU workers; smaller inputs stay serial to avoid
launch overhead. R2 uses a separate SIMD reduction. The benchmark prints its
machine and software versions so results from other hosts remain attributable.

No GPU path is provided. The targeted MSE reduction performs roughly three
arithmetic operations per 16 input bytes, far below the two-FLOP-per-byte
threshold where host/device transfers could pay off. The other benchmarked
kernels were already faster than TorchMetrics, so a GPU path was not justified.

## How it works

All kernels live in one compilation unit. Python owns every input, output, and
scratch allocation as C-contiguous `float64` or `int64` memory. `ctypes` passes
an address and shape metadata once per metric call; the C ABI exports accept
addresses as `Int` and rebuild
`UnsafePointer[..., AnyOrigin[mut=True]]` inside Mojo.

Confusion-matrix metrics share one linear counting kernel and reduce its small
matrix in Python. Exact binary AUROC and average precision copy into one
Python-owned pair scratch buffer, sort it in Mojo, group equal
thresholds, and integrate the ROC or precision-recall curve. Regression uses a
single pass over row-major memory to accumulate per-output sums, centered
moments, error totals, and correlations, with specialized SIMD R2 and
SIMD/parallel basic-error passes for single-output data. Stateful metrics add
those sufficient statistics rather than retaining prior batches; exact ranking
metrics retain scores because their global order is required.

## Verification

`pixi run test` runs 135 numerical and behavioral parity tests against the real
TorchMetrics 1.9.0 package. The suite covers functional and stateful APIs,
multiple state updates, logits, exact threshold behavior, ignored labels,
class averaging, normalized confusion matrices, tied ranking scores,
random ranking inputs, multi-output regression, adjusted R2, SIMD remainder
handling, the parallel basic-error threshold, the R2 threshold, and reset
behavior.

## License

MIT
