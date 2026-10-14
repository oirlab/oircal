from numba import njit, prange
import numpy as np

from ..utils.dq_utils import dq_flags_to_namedtuple
from ..utils.parallelization_utils import numba_thread_scope


__all__ = ["check_saturation"]


def check_saturation(
    input_data: np.ndarray,
    input_dq_raw: np.ndarray,
    input_dq: np.ndarray,
    sat_thresh: np.ndarray,
    sat_dq: np.ndarray,
    dq_flags: tuple | None = None,
    max_cores: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flag saturated reads in up-the-ramp data, in place.

    For each pixel, the first read at or above the pixel's saturation threshold and every
    read after it are flagged SATURATED | DO_NOT_USE in `input_dq_raw`, so a ramp that drops
    back below the threshold stays flagged. The threshold (DN) is compared directly to
    `input_data`. `sat_dq` is OR'ed into `input_dq` for every pixel.

    Parameters
    ----------
    input_data
        3D array of shape (Nreads, Nrows, Ncols) of ramps (DN). Not modified.
    input_dq_raw
        3D array of shape (Nreads, Nrows, Ncols) of per-read DQ flags. Modified in place.
    input_dq
        2D array of shape (Nrows, Ncols) of per-pixel DQ flags. Modified in place.
    sat_thresh
        2D uint16 array of shape (Nrows, Ncols) of saturation thresholds (DN).
    sat_dq
        2D array of shape (Nrows, Ncols) of DQ flags from the saturation calibration.
    dq_flags
        Dict mapping DQ flag names to bit values, with keys SATURATED and DO_NOT_USE. Both
        values must fit in the dtype of `input_dq_raw`. If None, reads are not flagged and
        only `sat_dq` is propagated to `input_dq`. Default is None.
    max_cores
        Maximum number of CPU cores to use. Default is 1.

    Returns
    -------
    input_dq_raw : np.ndarray
        The per-read DQ flags with saturated reads flagged (same array as the input).
    input_dq : np.ndarray
        The per-pixel DQ flags with `sat_dq` propagated (same array as the input).
    """
    dq_flags = dq_flags_to_namedtuple(dq_flags)
    with numba_thread_scope(max_cores):
        _check_saturation_numba(
            input_data,
            input_dq_raw,
            input_dq,
            sat_thresh,
            sat_dq,
            dq_flags,
        )
    return input_dq_raw, input_dq


@njit(nogil=True, parallel=True)
def _check_saturation_numba(
    data: np.ndarray,
    dq_raw: np.ndarray,
    dq: np.ndarray,
    sat_thresh: np.ndarray,
    sat_dq: np.ndarray,
    dq_flags: tuple | None,
):
    Nreads, Nrows, Ncols = data.shape
    for j in prange(Nrows):  # parallelize over spatial rows
        for i in range(Ncols):
            thresh = sat_thresh[j, i]
            dq[j, i] |= sat_dq[j, i]
            if dq_flags is not None:
                for r in range(Nreads):
                    if data[r, j, i] >= thresh:
                        # Scalar writes: numba fails to compile a slice |= when the flag is a plain Python int.
                        for rr in range(r, Nreads):
                            dq_raw[rr, j, i] |= dq_flags.SATURATED | dq_flags.DO_NOT_USE
                        break
