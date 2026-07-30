from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE = os.path.join(ROOT, "src", "kernels.mojo")
LIBRARY = os.environ.get("MOJO_TORCHMETRICS_LIB") or os.path.join(
    ROOT, "dist", "libmojo-torchmetrics.so"
)
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mt_binary_confusion": ([I, I, I, I, F, I, I], None),
    "mt_multiclass_confusion": ([I, I, I, I, I, I, I], None),
    "mt_binary_ranking": ([I, I, I, I, I, I, I], I),
    "mt_r2_reductions": ([I, I, I, I], None),
    "mt_regression_reductions": ([I, I, I, I, I, I], None),
    "mt_basic_errors": ([I, I, I, I], None),
}


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_TORCHMETRICS_LIB"):
        if os.path.exists(LIBRARY):
            return LIBRARY
        raise RuntimeError(f"MOJO_TORCHMETRICS_LIB does not exist: {LIBRARY}")
    stale = force or not os.path.exists(LIBRARY)
    if not stale:
        stale = os.path.getmtime(SOURCE) > os.path.getmtime(LIBRARY)
    if stale:
        process = subprocess.run(
            ["bash", os.path.join(ROOT, "build", "build.sh")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if process.returncode or not os.path.exists(LIBRARY):
            raise RuntimeError((process.stderr or process.stdout).strip()[:4000])
    return LIBRARY


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (arguments, result) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = arguments
            function.restype = result
    return _library


def addr(array: np.ndarray) -> int:
    return array.ctypes.data
