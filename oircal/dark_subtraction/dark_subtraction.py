import numpy as np
from numba import njit, prange

from ..utils.parallelization_utils import numba_thread_scope

__all__ = ['subtract_dark_rate']

def subtract_dark_rate(
    input_rate: np.ndarray,
    dark_rate: np.ndarray,
    input_err: np.ndarray | None = None,
    input_dq: np.ndarray | None = None,
    dark_err: np.ndarray | None = None,
    dark_dq: np.ndarray | None = None,
    max_cores: int = 1,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Subtract the dark current rate from the input rate, in place.

    Rates are in DN/s. Errors are combined in quadrature and data quality
    flags with a bitwise OR. Dark subtraction should be done before gain
    correction.

    Error propagation runs only if both `input_err` and `dark_err` are given;
    DQ propagation runs only if both `input_dq` and `dark_dq` are given.
    Otherwise the corresponding input array is returned unchanged.

    Parameters
    ----------
    input_rate : np.ndarray
        2D array of rate values (DN/s). Modified in place.
    dark_rate : np.ndarray
        2D array of dark rate values (DN/s).
    input_err : np.ndarray, optional
        2D array of error values (DN/s). Modified in place.
    input_dq : np.ndarray, optional
        2D array of data quality flags. Modified in place.
    dark_err : np.ndarray, optional
        2D array of dark error values (DN/s).
    dark_dq : np.ndarray, optional
        2D array of dark data quality flags.
    max_cores : int
        Maximum number of CPU cores to use for parallel processing.

    Returns
    -------
    input_rate, input_err, input_dq
        The (in-place modified) input arrays; `input_err` and `input_dq`
        are returned as passed, including None.
    """
    with numba_thread_scope(max_cores):
        _subtract_dark_rate_numba(
            input_rate, dark_rate,
            input_err, dark_err,
            input_dq, dark_dq,
        )
    return input_rate, input_err, input_dq


@njit(nogil=True, parallel=True, cache=True)
def _subtract_dark_rate_numba(
    input_rate, dark_rate,
    input_err, dark_err,
    input_dq, dark_dq,
):
    ny, nx = input_rate.shape
    for j in prange(ny):
        for i in range(nx):
            input_rate[j, i] -= dark_rate[j, i]

            if input_err is not None and dark_err is not None:
                input_err[j, i] = np.sqrt(
                    input_err[j, i] ** 2 + dark_err[j, i] ** 2
                )

            if input_dq is not None and dark_dq is not None:
                input_dq[j, i] |= dark_dq[j, i]