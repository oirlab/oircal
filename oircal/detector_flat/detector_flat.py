import numpy as np
from numba import njit, prange

from ..utils.parallelization_utils import numba_thread_scope

__all__ = ['correct_detector_flat']

def correct_detector_flat(
    input_rate: np.ndarray,
    detflat: np.ndarray,
    input_err: np.ndarray | None = None,
    input_dq: np.ndarray | None = None,
    detflat_err: np.ndarray | None = None,
    detflat_dq: np.ndarray | None = None,
    max_cores: int = 1,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Apply the detector-level flat field correction, in place.

    Pixels where the flat is not positive are left uncorrected. If
    `input_err` is given, it is propagated through the division; the flat
    contributes to the error only if `detflat_err` is also given, otherwise
    the flat is treated as noiseless. DQ propagation (bitwise OR) runs only
    if both `input_dq` and `detflat_dq` are given.

    Parameters
    ----------
    input_rate : np.ndarray
        2D input rate image (DN/s). Modified in place.
    detflat : np.ndarray
        2D detector flat field image.
    input_err : np.ndarray, optional
        2D error image associated with `input_rate`. Modified in place.
    input_dq : np.ndarray, optional
        2D data quality array associated with `input_rate`.Modified in place.
    detflat_err : np.ndarray, optional
        2D error image associated with `detflat`.
    detflat_dq : np.ndarray, optional
        2D data quality array associated with `detflat`.
    max_cores : int
        Maximum number of CPU cores to use.

    Returns
    -------
    input_rate, input_err, input_dq
        The (in-place modified) input arrays; `input_err` and `input_dq`
        are returned as passed, including None.
    """
    with numba_thread_scope(max_cores):
        _correct_detector_flat_numba(
            input_rate, detflat,
            input_err, detflat_err,
            input_dq, detflat_dq,
        )
    return input_rate, input_err, input_dq


@njit(nogil=True, parallel=True, cache=True)
def _correct_detector_flat_numba(
    input_rate, flat,
    input_err, flat_err,
    input_dq, flat_dq,
):
    ny, nx = input_rate.shape
    for j in prange(ny):
        for i in range(nx):
            f = flat[j, i]
            if f > 0:
                r = input_rate[j, i]
                input_rate[j, i] = r / f

                # Error propagation for r / f:
                # sigma^2 = (sigma_r / f)^2 + (r * sigma_f / f^2)^2
                if input_err is not None and flat_err is not None:
                    input_err[j, i] = np.sqrt(
                        (input_err[j, i] / f) ** 2
                        + (r * flat_err[j, i] / (f * f)) ** 2
                    )
                elif input_err is not None:
                    input_err[j, i] /= f

            if input_dq is not None and flat_dq is not None:
                input_dq[j, i] |= flat_dq[j, i]