"""Loads the CUDA extension.

A prebuilt module (``python setup.py build_ext --inplace`` inside ``lib/fof``)
is used when present; otherwise the sources in ``csrc/`` are JIT-compiled with
``torch.utils.cpp_extension.load`` on first use and cached by PyTorch.
"""

import os
import threading

_lock = threading.Lock()
_ext = None
_error = None

_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
SOURCES = [os.path.join(_CSRC, n) for n in ("fof_ext.cpp", "fof_kernels.cu", "mc_kernels.cu")]


def load(verbose=False):
    """Returns the extension module, building it if necessary (raises on failure)."""
    global _ext, _error
    if _ext is not None:
        return _ext
    with _lock:
        if _ext is not None:
            return _ext
        try:
            from . import _C  # noqa: F401  (prebuilt)

            _ext = _C
            return _ext
        except ImportError:
            pass
        try:
            from torch.utils.cpp_extension import load as _jit_load

            _ext = _jit_load(
                name="fof_cuda_ext",
                sources=SOURCES,
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3"],
                verbose=verbose,
            )
        except Exception as e:  # keep the original error for diagnostics
            _error = e
            raise
        return _ext


def available():
    """True if the CUDA extension can be used (tries to build it once)."""
    if _ext is not None:
        return True
    if _error is not None:
        return False
    try:
        load()
        return True
    except Exception:
        return False


def last_error():
    return _error
