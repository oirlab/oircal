from numba import njit, prange
import numpy as np

from ..utils.parallelization_utils import numba_thread_scope


__all__ = ["subtract_bias"]


def subtract_bias(
    input_data: np.ndarray,
    bias_data: np.ndarray,
    input_dq: np.ndarray | None = None,
    bias_dq: np.ndarray | None = None,
    max_cores: int = 1,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Subtract the bias from the ramps, in place.

    The bias is subtracted from every read. `input_data` must be floating point, since the
    result is written back into it. DQ propagation (bitwise OR) runs only if both `input_dq`
    and `bias_dq` are given. Otherwise `input_dq` is returned unchanged.

    Parameters
    ----------
    input_data
        3D array of shape (Nreads, Nrows, Ncols) of ramps (DN). Must be floating point. Modified in
        place.
    bias_data
        2D array of shape (Nrows, Ncols) of the bias (DN).
    input_dq
        2D array of shape (Nrows, Ncols) of per-pixel DQ flags. Modified in place. Default is None.
    bias_dq
        2D array of shape (Nrows, Ncols) of DQ flags from the bias calibration. Default is None.
    max_cores
        Maximum number of CPU cores to use. Default is 1.

    Returns
    -------
    input_data, input_dq
        The (in-place modified) input arrays; `input_dq` is returned as passed, including None.
    """
    assert not np.issubdtype(input_data.dtype, np.integer), (
        f"Ramps must be floating point, got {input_data.dtype}"
    )
    with numba_thread_scope(max_cores):
        _subtract_bias_numba(input_data, bias_data, input_dq, bias_dq)
    return input_data, input_dq


@njit(nogil=True, parallel=True)
def _subtract_bias_numba(data, bias, dq, bias_dq):
    Nreads, Nrows, Ncols = data.shape
    for j in prange(Nrows):
        for i in range(Ncols):
            for k in range(Nreads):
                data[k, j, i] -= bias[j, i]

            if dq is not None and bias_dq is not None:
                dq[j, i] |= bias_dq[j, i]
