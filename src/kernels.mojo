from std.builtin.sort import sort
from std.math import isfinite, log, sqrt
from std.memory import UnsafePointer
from std.sys import simd_width_of


comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]


@fieldwise_init
struct RankPair(Copyable, Movable, Comparable):
    var score: Float64
    var target: Int64

    def __eq__(self, rhs: Self) -> Bool:
        return self.score == rhs.score and self.target == rhs.target

    def __lt__(self, rhs: Self) -> Bool:
        return self.score < rhs.score


comptime RankPairPtr = UnsafePointer[RankPair, AnyOrigin[mut=True]]


def f64p(address: Int) -> F64Ptr:
    return F64Ptr(unsafe_from_address=address)


def i64p(address: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=address)


def rank_pair_p(address: Int) -> RankPairPtr:
    return RankPairPtr(unsafe_from_address=address)


@export("mt_binary_confusion")
def mt_binary_confusion(
    preds_address: Int,
    target_address: Int,
    matrix_address: Int,
    n: Int,
    threshold: Float64,
    ignore_index: Int,
    use_ignore: Int,
) abi("C"):
    var preds = f64p(preds_address)
    var target = i64p(target_address)
    var matrix = f64p(matrix_address)
    for i in range(4):
        matrix[i] = 0.0
    for i in range(n):
        if use_ignore != 0 and Int(target[i]) == ignore_index:
            continue
        var prediction = 1 if preds[i] > threshold else 0
        matrix[Int(target[i]) * 2 + prediction] += 1.0


@export("mt_multiclass_confusion")
def mt_multiclass_confusion(
    preds_address: Int,
    target_address: Int,
    matrix_address: Int,
    n: Int,
    classes: Int,
    ignore_index: Int,
    use_ignore: Int,
) abi("C"):
    var preds = i64p(preds_address)
    var target = i64p(target_address)
    var matrix = f64p(matrix_address)
    for i in range(classes * classes):
        matrix[i] = 0.0
    for i in range(n):
        if use_ignore != 0 and Int(target[i]) == ignore_index:
            continue
        matrix[Int(target[i]) * classes + Int(preds[i])] += 1.0


@export("mt_binary_ranking")
def mt_binary_ranking(
    preds_address: Int,
    target_address: Int,
    pair_work_address: Int,
    result_address: Int,
    n: Int,
    ignore_index: Int,
    use_ignore: Int,
) abi("C") -> Int:
    var preds = f64p(preds_address)
    var target = i64p(target_address)
    var pairs = rank_pair_p(pair_work_address)
    var result = f64p(result_address)
    var kept = 0
    var positives = 0.0
    for i in range(n):
        if use_ignore != 0 and Int(target[i]) == ignore_index:
            continue
        pairs[kept] = RankPair(preds[i], target[i])
        positives += Float64(target[i])
        kept += 1

    var pair_span = Span(unsafe_ptr=pairs, length=kept)

    sort(pair_span)
    var negatives = Float64(kept) - positives
    var tp = 0.0
    var fp = 0.0
    var previous_tp = 0.0
    var previous_fp = 0.0
    var auc_area = 0.0
    var ap_area = 0.0
    var i = kept - 1
    while i >= 0:
        var score = pairs[i].score
        var group_positive = 0.0
        var group_total = 0.0
        while i >= 0 and pairs[i].score == score:
            group_positive += Float64(pairs[i].target)
            group_total += 1.0
            i -= 1
        tp += group_positive
        fp += group_total - group_positive
        auc_area += (fp - previous_fp) * (tp + previous_tp) * 0.5
        if positives > 0.0 and tp + fp > 0.0:
            ap_area += group_positive / positives * tp / (tp + fp)
        previous_tp = tp
        previous_fp = fp
    result[0] = auc_area / (positives * negatives) if positives > 0.0 and negatives > 0.0 else 0.0
    result[1] = ap_area if positives > 0.0 else 0.0
    return kept


def r2_chunk(
    preds: F64Ptr,
    target: F64Ptr,
    partials: F64Ptr,
    start: Int,
    end: Int,
    offset: Int,
):
    comptime W = simd_width_of[DType.float64]()
    var squared_error = SIMD[DType.float64, W](0.0)
    var target_sum = SIMD[DType.float64, W](0.0)
    var target_square_sum = SIMD[DType.float64, W](0.0)
    var vector_end = start + (end - start) // W * W
    for i in range(start, vector_end, W):
        var prediction = preds.load[width=W](i)
        var actual = target.load[width=W](i)
        var error = prediction - actual
        squared_error += error * error
        target_sum += actual
        target_square_sum += actual * actual
    var squared_error_tail = 0.0
    var target_sum_tail = 0.0
    var target_square_sum_tail = 0.0
    for i in range(vector_end, end):
        var prediction = preds[i]
        var actual = target[i]
        var error = prediction - actual
        squared_error_tail += error * error
        target_sum_tail += actual
        target_square_sum_tail += actual * actual
    partials[offset] = squared_error.reduce_add() + squared_error_tail
    partials[offset + 1] = target_sum.reduce_add() + target_sum_tail
    partials[offset + 2] = target_square_sum.reduce_add() + target_square_sum_tail


@export("mt_r2_reductions")
def mt_r2_reductions(
    preds_address: Int,
    target_address: Int,
    partials_address: Int,
    n: Int,
) abi("C"):
    comptime WORKERS = 8
    comptime PARALLEL_THRESHOLD = 1_000_000
    var preds = f64p(preds_address)
    var target = f64p(target_address)
    var partials = f64p(partials_address)
    if n < PARALLEL_THRESHOLD:
        r2_chunk(preds, target, partials, 0, n, 0)
        return

    for part in range(WORKERS):
        var start = part * n // WORKERS
        var end = (part + 1) * n // WORKERS
        r2_chunk(preds, target, partials, start, end, part * 3)
    var squared_error = 0.0
    var target_sum = 0.0
    var target_square_sum = 0.0
    for part in range(WORKERS):
        var offset = part * 3
        squared_error += partials[offset]
        target_sum += partials[offset + 1]
        target_square_sum += partials[offset + 2]
    partials[0] = squared_error
    partials[1] = target_sum
    partials[2] = target_square_sum


@export("mt_regression_reductions")
def mt_regression_reductions(
    preds_address: Int,
    target_address: Int,
    reductions_address: Int,
    rows: Int,
    outputs: Int,
    include_log: Int,
) abi("C"):
    var preds = f64p(preds_address)
    var target = f64p(target_address)
    var reductions = f64p(reductions_address)
    for i in range(outputs * 13):
        reductions[i] = 0.0
    for i in range(rows):
        for j in range(outputs):
            var index = i * outputs + j
            var prediction = preds[index]
            var actual = target[index]
            var error = prediction - actual
            var absolute_error = abs(error)
            var offset = j * 13
            reductions[offset] += 1.0
            reductions[offset + 1] += error * error
            reductions[offset + 2] += absolute_error
            reductions[offset + 3] += absolute_error / max(abs(actual), 1.17e-6)
            reductions[offset + 4] += 2.0 * absolute_error / max(abs(actual) + abs(prediction), 1.0e-6)
            reductions[offset + 5] += abs(actual)
            reductions[offset + 6] += actual
            reductions[offset + 7] += prediction
            reductions[offset + 8] += actual * actual
            reductions[offset + 9] += prediction * prediction
            reductions[offset + 10] += actual * prediction
            reductions[offset + 11] += error
            if include_log != 0:
                var log_error = log(1.0 + prediction) - log(1.0 + actual)
                reductions[offset + 12] += log_error * log_error


def basic_errors_chunk(
    preds: F64Ptr,
    target: F64Ptr,
    result: F64Ptr,
    n: Int,
    worker: Int,
    workers: Int,
):
    comptime W = simd_width_of[DType.float64]()
    var start = worker * n // workers
    var end = (worker + 1) * n // workers
    var vector_end = start + (end - start) // W * W
    var squared = SIMD[DType.float64, W](0.0)
    var absolute = SIMD[DType.float64, W](0.0)
    var valid = True
    for i in range(start, vector_end, W):
        var prediction = preds.load[width=W](i)
        var actual = target.load[width=W](i)
        if (
            not isfinite(prediction).reduce_and()
            or not isfinite(actual).reduce_and()
        ):
            valid = False
        var error = prediction - actual
        squared += error * error
        absolute += abs(error)
    var squared_tail = 0.0
    var absolute_tail = 0.0
    for i in range(vector_end, end):
        var prediction = preds[i]
        var actual = target[i]
        if not isfinite(prediction) or not isfinite(actual):
            valid = False
        var error = prediction - actual
        squared_tail += error * error
        absolute_tail += abs(error)
    result[worker * 3] = squared.reduce_add() + squared_tail
    result[worker * 3 + 1] = absolute.reduce_add() + absolute_tail
    result[worker * 3 + 2] = 1.0 if valid else 0.0


@export("mt_basic_errors")
def mt_basic_errors(
    preds_address: Int,
    target_address: Int,
    result_address: Int,
    n: Int,
) abi("C"):
    comptime WORKERS = 8
    comptime PARALLEL_THRESHOLD = 1_000_000
    var preds = f64p(preds_address)
    var target = f64p(target_address)
    var result = f64p(result_address)
    # Two flops per 16 bytes of input: bandwidth bound, so the partials are
    # accumulated over a serial chunk loop rather than fanned out.
    var workers = 1 if n < PARALLEL_THRESHOLD else WORKERS
    for worker in range(workers):
        basic_errors_chunk(preds, target, result, n, worker, workers)
    var squared = 0.0
    var absolute = 0.0
    var valid = True
    for worker in range(workers):
        squared += result[worker * 3]
        absolute += result[worker * 3 + 1]
        valid = valid and result[worker * 3 + 2] != 0.0
    result[0] = squared
    result[1] = absolute
    result[2] = 1.0 if valid else 0.0
