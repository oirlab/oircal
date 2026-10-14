from numba import njit
import numpy as np


@njit(inline="always", nogil=True)
def seconds_to_mjd_offset(seconds: float) -> float:
    """Convert a time offset in seconds to fractional days (MJD units)."""
    return float(seconds) / 86400.0


@njit(nogil=True)
def mean_numba_if_not_zero(arr: np.ndarray) -> float:
    total = 0.0
    n = 0
    for i in range(arr.size):
        if arr[i] != 0:
            total += arr[i]
            n = n + 1
    if n != 0:
        return total / n
    else:
        return 0


@njit(nogil=True)
def bitwise_or_reduce(arr: np.ndarray) -> np.unsignedinteger:
    result = arr.dtype.type(0)
    for k in range(arr.shape[0]):
        result |= arr[k]
    return result
