from ..utils.parallelization_utils import numba_thread_scope
from numba import njit, prange
import numpy as np

__all__ = ['check_saturation']


def check_saturation(
    input_data : np.ndarray, input_dq_raw : np.ndarray, input_dq : np.ndarray,
    sat_thresh : np.ndarray, sat_dq : np.ndarray,
    dq_flags : tuple | None = None,
    max_cores : int = 1
):
    """
    Flag saturated pixels in ramps.

    Parameters
    ----------
    input_data : np.ndarray
        The input ramp data array of shape (Nreads, Ny, Nx).
    input_dq_raw : np.ndarray
        The input DQ_RAW array of shape (Nreads, Ny, Nx).
    input_dq : np.ndarray
        The input DQ array of shape (Ny, Nx).
    sat_thresh : np.ndarray
        The saturation threshold map of shape (Ny, Nx).
    sat_dq : np.ndarray
        The DQ map from the saturation model of shape (Ny, Nx).
    dq_flags : namedtuple, optional
        Named tuple of DQ bit values with integer fields SATURATED and DO_NOT_USE.
        If None (default), saturated reads are not flagged in `input_dq_raw`;
        `sat_dq` is still propagated to `input_dq`.
    max_cores : int, optional
        The maximum number of CPU cores to use. Default is 1.

    Returns
    -------
    input_dq_raw : np.ndarray
        The updated DQ_RAW array with saturation flags applied.
    input_dq : np.ndarray
        The updated DQ array with saturation flags applied.
    """
    with numba_thread_scope(max_cores):
        _check_saturation_numba(
            input_data, input_dq_raw, input_dq,
            sat_thresh, sat_dq, dq_flags,
        )
    return input_dq_raw, input_dq

@njit(nogil=True, parallel=True, cache=True)
def _check_saturation_numba(
    data : np.ndarray,
    dq_raw : np.ndarray,
    dq : np.ndarray,
    sat_thresh : np.ndarray,
    sat_dq : np.ndarray,
    dq_flags : tuple | None,
):
    n_read, ny, nx = data.shape
    for j in prange(ny):  # parallelize over spatial rows
        for i in range(nx):
            thresh = sat_thresh[j, i]
            dq[j, i] |= sat_dq[j, i]
            if dq_flags is not None:
                for r in range(n_read):
                    if data[r, j, i] >= thresh:
                        # Scalar writes: numba fails to compile a slice |= when the flag is a plain Python int.
                        for rr in range(r, n_read):
                            dq_raw[rr, j, i] |= (dq_flags.SATURATED | dq_flags.DO_NOT_USE)
                        break