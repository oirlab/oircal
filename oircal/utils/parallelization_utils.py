"""
Shared helpers for parallel CPU work: shared-memory views and Numba thread limits.

Numba fixes its maximum thread count (``numba.config.NUMBA_NUM_THREADS``) at import,
from the process CPU affinity mask, which on CI runners can be lower than
``os.cpu_count()``. Requested thread counts are clamped to that maximum so
``set_num_threads`` never raises ``ValueError``.
"""
from __future__ import annotations

import ctypes
import os
import numpy as np
from contextlib import contextmanager

from numba import config, get_num_threads, set_num_threads

import typing
if typing.TYPE_CHECKING:
    from multiprocessing.sharedctypes import Array

__all__ = [
    "shared_array_to_numpy",
    "numba_thread_scope",
    "resolve_numba_threads",
]


def shared_array_to_numpy(
    shared_array : Array,
    shape : tuple | None = None,
    dtype : type | None = None
):
    """
    Covert a shared array to a numpy array

    Originally from pyklip Wang, J. J., Ruffio, J.-B., De Rosa, R. J., et al. 2015, ASCL, ascl:1506.001
    https://bitbucket.org/pyKLIP/pyklip/src/main/pyklip/parallelized.py

    Parameters
    ----------
    shared_array : multiprocessing.sharedctypes.Array
        The shared array to convert.
    shape : tuple of int, optional
        Shape of the returned array. Default is None (1D).
    dtype : type, optional
        The ctypes element type of the shared array. Default is ctypes.c_float.

    Returns
    -------
    numpy_array : np.ndarray
        The converted numpy array for vectorized operation. still points to the same memory.
    """
    if dtype is None:
        dtype = ctypes.c_float

    # if you passed in nothing you get nothing
    if shared_array is None:
        return None

    buf = shared_array.get_obj() if hasattr(shared_array, "get_obj") else shared_array
    # np.dtype understands ctypes types directly and raises on unsupported ones.
    numpy_array = np.frombuffer(buf, dtype=np.dtype(dtype))
    if shape is not None:
        # copy=False raises instead of silently copying, so the result always views the shared memory.
        numpy_array = numpy_array.reshape(shape, copy=False)

    return numpy_array


def resolve_numba_threads(max_cores: int | None = None) -> int:
    """
    Map a requested parallelism level to a thread count safe for this host and Numba build.

    - ``None``: use all CPUs available to this process, capped to Numba's maximum.
    - Positive ``max_cores``: use at most that many threads, capped the same way.

    Parameters
    ----------
    max_cores : int or None, optional
        Requested number of threads. Default is ``None``.

    Returns
    -------
    num_threads : int
        Thread count in ``1 <= num_threads <= numba.config.NUMBA_NUM_THREADS``.

    Raises
    ------
    ValueError
        If ``max_cores`` is less than 1.
    """
    if max_cores is not None and max_cores < 1:
        raise ValueError(f"max_cores must be None or >= 1, got {max_cores}")
    if not max_cores:
        max_cores = os.process_cpu_count() or 1
    return max(1, min(int(max_cores), config.NUMBA_NUM_THREADS))


@contextmanager
def numba_thread_scope(num_threads: int | None = None):
    """
    Temporarily set Numba's thread count, then restore the previous setting.

    The requested count is resolved with :func:`resolve_numba_threads`. Numba's thread
    count is per calling thread, so this only affects Numba parallel regions launched
    from the thread that enters the scope; it has no effect in multiprocessing workers.

    Parameters
    ----------
    num_threads : int or None, optional
        Explicit thread count, or ``None`` for all available CPUs (capped).
        Default is ``None``.

    Examples
    --------
    >>> with numba_thread_scope(4):
    ...     ...  # Numba parallel regions here
    """
    effective = resolve_numba_threads(num_threads)
    original_num_threads = get_num_threads()
    set_num_threads(effective)
    try:
        yield
    finally:
        set_num_threads(original_num_threads)
