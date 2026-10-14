from numba import njit, prange
import numpy as np

from ..utils.parallelization_utils import numba_thread_scope


__all__ = ["subtract_dark_rate"]


def subtract_dark_rate(
    input_data: np.ndarray,
    dark_rate: np.ndarray,
    input_err: np.ndarray | None = None,
    input_dq: np.ndarray | None = None,
    dark_err: np.ndarray | None = None,
    dark_dq: np.ndarray | None = None,
    max_cores: int = 1,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Subtract the dark current rate from the input rate, in place.

    `input_data` and `input_err` must be floating point, since the results are written back
    into them. Errors are combined in quadrature and DQ flags with a bitwise OR. Error
    propagation runs only if both `input_err` and `dark_err` are given; DQ propagation runs
    only if both `input_dq` and `dark_dq` are given. Otherwise the corresponding input array
    is returned unchanged.

    Parameters
    ----------
    input_data
        2D array of shape (Nrows, Ncols) of rates (DN/s). Must be floating point. Modified in place.
    dark_rate
        2D array of shape (Nrows, Ncols) of dark current rates (DN/s).
    input_err
        2D array of shape (Nrows, Ncols) of rate uncertainties (DN/s). Must be floating point.
        Modified in place. Default is None.
    input_dq
        2D array of shape (Nrows, Ncols) of per-pixel DQ flags. Modified in place. Default is None.
    dark_err
        2D array of shape (Nrows, Ncols) of dark rate uncertainties (DN/s). Default is None.
    dark_dq
        2D array of shape (Nrows, Ncols) of DQ flags from the dark calibration. Default is None.
    max_cores
        Maximum number of CPU cores to use for parallel processing. Default is 1.

    Returns
    -------
    input_data, input_err, input_dq
        The (in-place modified) input arrays; `input_err` and `input_dq`
        are returned as passed, including None.
    """
    assert not np.issubdtype(input_data.dtype, np.integer), (
        f"Rates must be floating point, got {input_data.dtype}"
    )
    assert input_err is None or not np.issubdtype(input_err.dtype, np.integer), (
        f"Rate errors must be floating point, got {input_err.dtype}"
    )
    with numba_thread_scope(max_cores):
        _subtract_dark_rate_numba(
            input_data,
            dark_rate,
            input_err,
            dark_err,
            input_dq,
            dark_dq,
        )
    return input_data, input_err, input_dq


@njit(nogil=True, parallel=True)
def _subtract_dark_rate_numba(
    input_rate,
    dark_rate,
    input_err,
    dark_err,
    input_dq,
    dark_dq,
):
    Nrows, Ncols = input_rate.shape
    for j in prange(Nrows):
        for i in range(Ncols):
            input_rate[j, i] -= dark_rate[j, i]

            if input_err is not None and dark_err is not None:
                input_err[j, i] = np.sqrt(
                    input_err[j, i] ** 2 + dark_err[j, i] ** 2,
                )

            if input_dq is not None and dark_dq is not None:
                input_dq[j, i] |= dark_dq[j, i]
