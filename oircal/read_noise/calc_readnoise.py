from numba import njit, prange
import numpy as np

from ..utils.dq_utils import dq_flags_to_namedtuple
from ..utils.parallelization_utils import numba_thread_scope


__all__ = ["calc_readnoise"]


def calc_readnoise(
    input_data: np.ndarray,
    input_dq_raw: np.ndarray | None = None,
    dq_flags: dict | None = None,
    mode: str = "per_pixel",
    max_cores: int = 1,
) -> np.ndarray | float:
    """
    Calculate the single-read read noise from read differences of darks.

    A read is usable unless it is flagged DO_NOT_USE or SATURATED or is not a finite number.
    The function works with the differences ``d_k = r[k+1] - r[k]`` between consecutive usable
    reads ``r`` of each ramp, so unusable reads are skipped: with every other read flagged, the
    differences span two reads. Any offset that is the same for every read of a ramp (such as
    kTC noise) cancels in these differences. The function assumes that the dark signal and its
    shot noise are negligible. Under that assumption, each difference has mean 0 and variance
    ``2 sigma^2``, where ``sigma`` is the single-read read noise. The function therefore
    estimates the read noise variance of each pixel as ``sigma^2 = sum d_k^2 / (2 n)``, where
    the sum runs over all ``n`` usable differences of the pixel, from all reads of all ramps.
    This estimate is unbiased when the dark signal is negligible. If the dark signal is not
    negligible, the estimate is too high, more so for differences that skip unusable reads.

    With ``mode="full_detector"``, the function returns the square root of the mean of the
    per-pixel variance estimates.

    A difference ``d_k`` is left out if usable read ``k+1``, or any unusable read between usable
    reads ``k`` and ``k+1``, is flagged JUMP_DET, since the jump happened between the two reads.

    Parameters
    ----------
    input_data
        4D array of shape (Nfiles, Nreads, Nrows, Ncols) or 3D array of shape
        (Nreads, Nrows, Ncols) of dark ramps (DN). May be an integer type. Not modified.
    input_dq_raw
        Array of the same shape as `input_data` of per-read DQ flags. Default is None.
    dq_flags
        Dict mapping DQ flag names to bit values, with keys DO_NOT_USE, SATURATED and JUMP_DET.
        If `input_dq_raw` or `dq_flags` is None, all finite reads are used. Default is None.
    mode
        ``"per_pixel"`` returns the read noise of each pixel; ``"full_detector"`` returns one
        read noise for all pixels. Default is ``"per_pixel"``.
    max_cores
        Maximum number of CPU cores to use. Default is 1.

    Returns
    -------
    rn : np.ndarray or float
        Single-read read noise (DN). For ``"per_pixel"``, a 2D float32 array of shape
        (Nrows, Ncols), NaN for pixels with no usable differences. For ``"full_detector"``, a
        float from the pixels with usable differences; NaN if there are none.
    """
    if mode not in ("per_pixel", "full_detector"):
        raise ValueError(f"mode must be 'per_pixel' or 'full_detector', got {mode!r}.")
    if input_data.ndim == 3:
        input_data = input_data[np.newaxis]
        if input_dq_raw is not None:
            input_dq_raw = input_dq_raw[np.newaxis]
    if input_data.ndim != 4:
        raise ValueError(f"input_data must be 3D or 4D, got {input_data.ndim}D.")
    Nfiles, Nreads, Nrows, Ncols = input_data.shape

    dq_flags = dq_flags_to_namedtuple(dq_flags)
    if dq_flags is None:
        input_dq_raw = None
    bad_flags = 0 if dq_flags is None else int(dq_flags.DO_NOT_USE | dq_flags.SATURATED)
    jump_flag = 0 if dq_flags is None else int(dq_flags.JUMP_DET)

    var = np.empty((Nrows, Ncols))
    with numba_thread_scope(max_cores):
        _calc_readnoise_var_numba(input_data, input_dq_raw, bad_flags, jump_flag, var)

    if mode == "full_detector":
        if np.isnan(var).all():
            return np.nan
        return float(np.sqrt(np.nanmean(var)))
    return np.sqrt(var).astype(np.float32)


@njit(nogil=True, parallel=True)
def _calc_readnoise_var_numba(data, dq_raw, bad_flags, jump_flag, var):
    """Write sum d^2 / (2 n) (DN^2) of each pixel to `var`, NaN if there are no differences."""
    Nfiles, Nreads, Nrows, Ncols = data.shape
    for j in prange(Nrows):
        for i in range(Ncols):
            sum_sq = 0.0
            n = 0
            for f in range(Nfiles):
                # Difference each usable read with the previous usable read, skipping unusable
                # reads. A jump on any read since the previous usable read, usable or not, is
                # inside the difference.
                has_prev = False
                prev = 0.0
                jump = False
                for k in range(Nreads):
                    if dq_raw is not None:
                        jump = jump or (dq_raw[f, k, j, i] & jump_flag) != 0
                        if dq_raw[f, k, j, i] & bad_flags:
                            continue
                    value = float(data[f, k, j, i])
                    if not np.isfinite(value):
                        continue
                    if has_prev and not jump:
                        d = value - prev
                        sum_sq += d * d
                        n += 1
                    has_prev = True
                    prev = value
                    jump = False
            var[j, i] = sum_sq / (2 * n) if n > 0 else np.nan
