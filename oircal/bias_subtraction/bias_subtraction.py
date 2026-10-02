import numpy as np
from numba import njit, prange

from ..utils.parallelization_utils import numba_thread_scope

__all__ = ['subtract_bias']

def subtract_bias(
    input_data: np.ndarray,
    bias_data: np.ndarray,
    input_dq: np.ndarray | None = None,
    bias_dq: np.ndarray | None = None,
    max_cores: int = 1,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Subtract the bias from the ramps, in place.

    DQ propagation (bitwise OR) runs only if both `input_dq` and `bias_dq`
    are given. Otherwise `input_dq` is returned unchanged.

    Parameters
    ----------
    input_data : np.ndarray
        Input ramp data with shape (Nreads, Ny, Nx). Modified in place.
    bias_data : np.ndarray
        Bias data with shape (Ny, Nx).
    input_dq : np.ndarray, optional
        Input DQ array with shape (Ny, Nx). Modified in place.
    bias_dq : np.ndarray, optional
        Bias DQ array with shape (Ny, Nx).
    max_cores : int
        Maximum number of CPU cores to use.

    Returns
    -------
    input_data, input_dq
        The (in-place modified) input arrays; `input_dq` is returned as passed, including None.
    """
    with numba_thread_scope(max_cores):
        _subtract_bias_numba(input_data, bias_data, input_dq, bias_dq)
    return input_data, input_dq


@njit(nogil=True, parallel=True, cache=True)
def _subtract_bias_numba(data, bias, dq, bias_dq):
    n_read, ny, nx = data.shape
    for j in prange(ny):
        for i in range(nx):
            for k in range(n_read):
                data[k, j, i] -= bias[j, i]

            if dq is not None and bias_dq is not None:
                dq[j, i] |= bias_dq[j, i]