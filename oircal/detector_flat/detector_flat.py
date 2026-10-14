from numba import njit, prange
import numpy as np

from ..utils.parallelization_utils import numba_thread_scope


__all__ = ["correct_detector_flat"]


def correct_detector_flat(
    input_data: np.ndarray,
    detflat: np.ndarray,
    input_err: np.ndarray | None = None,
    input_dq: np.ndarray | None = None,
    detflat_err: np.ndarray | None = None,
    detflat_dq: np.ndarray | None = None,
    max_cores: int = 1,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Apply the detector-level flat field correction, in place.

    `input_data` (and `input_err`) are divided by the flat. Both must be floating point, since
    the results are written back into them. Pixels where the flat is not positive (or is NaN)
    are left uncorrected. If `input_err` is given, it is propagated through the division; the
    flat contributes to the error only if `detflat_err` is also given, otherwise the flat is
    treated as noiseless. DQ propagation (bitwise OR) runs only if both `input_dq` and
    `detflat_dq` are given.

    Parameters
    ----------
    input_data
        2D array of shape (Nrows, Ncols) of rates (DN/s). Must be floating point. Modified in place.
    detflat
        2D array of shape (Nrows, Ncols) of the detector flat field (unitless, normalized).
    input_err
        2D array of shape (Nrows, Ncols) of rate uncertainties (DN/s). Must be floating point.
        Modified in place. Default is None.
    input_dq
        2D array of shape (Nrows, Ncols) of per-pixel DQ flags. Modified in place. Default is None.
    detflat_err
        2D array of shape (Nrows, Ncols) of flat field uncertainties (unitless). Default is None.
    detflat_dq
        2D array of shape (Nrows, Ncols) of DQ flags from the flat calibration. Default is None.
    max_cores
        Maximum number of CPU cores to use. Default is 1.

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
        _correct_detector_flat_numba(
            input_data,
            detflat,
            input_err,
            detflat_err,
            input_dq,
            detflat_dq,
        )
    return input_data, input_err, input_dq


@njit(nogil=True, parallel=True)
def _correct_detector_flat_numba(
    input_rate,
    flat,
    input_err,
    flat_err,
    input_dq,
    flat_dq,
):
    Nrows, Ncols = input_rate.shape
    for j in prange(Nrows):
        for i in range(Ncols):
            f = flat[j, i]
            if f > 0:
                r = input_rate[j, i]
                input_rate[j, i] = r / f

                # Error propagation for r / f:
                # sigma^2 = (sigma_r / f)^2 + (r * sigma_f / f^2)^2
                if input_err is not None and flat_err is not None:
                    input_err[j, i] = np.sqrt(
                        (input_err[j, i] / f) ** 2 + (r * flat_err[j, i] / (f * f)) ** 2,
                    )
                elif input_err is not None:
                    input_err[j, i] /= f

            if input_dq is not None and flat_dq is not None:
                input_dq[j, i] |= flat_dq[j, i]
