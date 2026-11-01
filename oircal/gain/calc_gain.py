from numba import njit, prange
import numpy as np

from ..coadd import combine
from ..utils.dq_utils import dq_flags_to_namedtuple
from ..utils.parallelization_utils import numba_thread_scope


__all__ = ["calc_gain"]


def calc_gain(
    input_data: np.ndarray,
    rn: np.ndarray | float,
    input_dq_raw: np.ndarray | None = None,
    dq_flags: dict | None = None,
    mode: str = "per_pixel",
    n_sigma: float = 5.0,
    max_cores: int = 1,
) -> tuple[np.ndarray, np.ndarray] | tuple[float, float]:
    """
    Calculate the gain and its error from the mean and variance of read differences.

    The function works with the differences ``d_k = r[k+1] - r[k]`` between consecutive reads
    ``r`` of illuminated ramps with evenly spaced reads. Each difference has mean ``mu``, the
    signal per read interval (DN), and variance ``mu / g + 2 sigma^2``, where ``g`` is the gain
    and ``sigma`` is the single-read read noise `rn`. The differences of a ramp are split into
    runs of consecutive usable differences, and each run of ``n >= 2`` differences gives
    ``A = sum (d_k - dbar)^2 - 2 sigma^2 (n - 1/n)`` and ``B = (n - 1) dbar``, with expected
    values ``(n - 1) mu / g`` and ``(n - 1) mu``. The term ``n - 1/n`` (not ``n - 1``) holds
    because neighboring differences share a read. The gain of each pixel is ``sum B / sum A``
    over all runs of all ramps, so ramps may have different count rates.

    The error comes from the scatter of the runs about the gain, as for a ratio estimator:
    ``err^2 = N / (N - 1) * sum (B - g A)^2 / (sum A)^2``, where ``N`` is the number of runs.
    It is reliable only with many runs, e.g. 10 or more ramps.

    A read is unusable if it is flagged DO_NOT_USE or SATURATED or is not a finite number. A
    difference is usable if both of its reads are usable and read ``k+1`` is not flagged
    JUMP_DET. Unusable differences end a run, since a difference that skips a read has a
    different mean.

    With ``mode="full_detector"``, the inverse gains ``sum A / sum B`` of all pixels are
    sigma clipped at `n_sigma` and averaged with :func:`.combine`, and the gain is the inverse
    of the mean, with its error from the scatter of the kept pixels. The inverse gain is
    averaged rather than the gain because it is nearly linear in the noisy ``A``, so its mean
    is not biased by the noise of each pixel's gain.

    Parameters
    ----------
    input_data
        4D array of shape (Nfiles, Nreads, Nrows, Ncols) or 3D array of shape
        (Nreads, Nrows, Ncols) of illuminated ramps (DN). Should be linear (e.g. corrected
        for nonlinearity). May be an integer type. Not modified.
    rn
        Single-read read noise (DN), a float or a 2D array of shape (Nrows, Ncols), e.g. from
        `.calc_readnoise`.
    input_dq_raw
        Array of the same shape as `input_data` of per-read DQ flags. Default is None.
    dq_flags
        Dict mapping DQ flag names to bit values, with keys DO_NOT_USE, SATURATED and JUMP_DET.
        If `input_dq_raw` or `dq_flags` is None, all finite reads are used. Default is None.
    mode
        ``"per_pixel"`` returns the gain of each pixel; ``"full_detector"`` returns one gain
        for all pixels. Default is ``"per_pixel"``.
    n_sigma
        Sigma clipping threshold, in standard deviations, for the pixels combined with
        ``"full_detector"``. Default is 5.0.
    max_cores
        Maximum number of CPU cores to use. Default is 1.

    Returns
    -------
    gain : np.ndarray or float
        Gain (e-/DN). For ``"per_pixel"``, a 2D float32 array of shape (Nrows, Ncols), NaN for
        pixels without a run of 2 usable differences or with ``sum A <= 0`` or ``sum B <= 0``.
        For ``"full_detector"``, a float, NaN if no pixel has a run of 2 usable differences.
    gain_err : np.ndarray or float
        Error of `gain` (e-/DN), same shape as `gain`. NaN where `gain` is NaN or there are
        fewer than 2 runs.
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
    rn = np.broadcast_to(np.asarray(rn, dtype=np.float64), (Nrows, Ncols))

    dq_flags = dq_flags_to_namedtuple(dq_flags)
    if dq_flags is None:
        input_dq_raw = None
    bad_flags = 0 if dq_flags is None else int(dq_flags.DO_NOT_USE | dq_flags.SATURATED)
    jump_flag = 0 if dq_flags is None else int(dq_flags.JUMP_DET)

    # sum A, sum B, sum A^2, sum A B, sum B^2 and the number of runs of each pixel
    sums = np.zeros((6, Nrows, Ncols))
    with numba_thread_scope(max_cores):
        _calc_gain_sums_numba(input_data, input_dq_raw, rn, bad_flags, jump_flag, sums)

    sum_a, sum_b, sum_aa, sum_ab, sum_bb, Nruns = sums

    if mode == "full_detector":
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_gain = np.where(sum_b > 0, sum_a / sum_b, np.nan)
        inv_gain, inv_gain_err, _ = combine(
            inv_gain.ravel(),
            method="mean",
            err_method="scatter",
            do_sigma_clip=True,
            sigma_thresh_low=n_sigma,
            sigma_thresh_high=n_sigma,
            max_cores=max_cores,
        )
        if not inv_gain > 0:
            return np.nan, np.nan
        return float(1 / inv_gain), float(inv_gain_err / inv_gain**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where((sum_a > 0) & (sum_b > 0), sum_b / sum_a, np.nan)
        # sum (B - g A)^2, expanded so it can be summed in one pass
        sum_resid_sq = sum_bb - 2 * gain * sum_ab + gain**2 * sum_aa
        gain_err = np.sqrt(Nruns / (Nruns - 1) * np.maximum(sum_resid_sq, 0)) / sum_a
    gain_err = np.where(Nruns >= 2, gain_err, np.nan)
    return gain.astype(np.float32), gain_err.astype(np.float32)


@njit(nogil=True, parallel=True)
def _calc_gain_sums_numba(data, dq_raw, rn, bad_flags, jump_flag, sums):
    """Add sum A, sum B, sum A^2, sum A B, sum B^2 and the number of runs to `sums`."""
    Nfiles, Nreads, Nrows, Ncols = data.shape
    for j in prange(Nrows):
        for i in range(Ncols):
            two_var_rn = 2.0 * rn[j, i] ** 2
            for f in range(Nfiles):
                # Running sums of the current run of usable differences
                n = 0
                sum_d = 0.0
                sum_sq = 0.0
                prev_ok = False
                prev = 0.0
                for k in range(Nreads + 1):
                    ok = k < Nreads
                    value = 0.0
                    jump = False
                    if ok:
                        value = float(data[f, k, j, i])
                        ok = np.isfinite(value)
                        if dq_raw is not None:
                            jump = (dq_raw[f, k, j, i] & jump_flag) != 0
                            ok = ok and (dq_raw[f, k, j, i] & bad_flags) == 0
                    if ok and prev_ok and not jump:
                        d = value - prev
                        sum_d += d
                        sum_sq += d * d
                        n += 1
                    elif n > 0:
                        # The run ends here (also after the last read)
                        if n >= 2:
                            mean = sum_d / n
                            a = sum_sq - n * mean * mean - two_var_rn * (n - 1.0 / n)
                            b = (n - 1) * mean
                            sums[0, j, i] += a
                            sums[1, j, i] += b
                            sums[2, j, i] += a * a
                            sums[3, j, i] += a * b
                            sums[4, j, i] += b * b
                            sums[5, j, i] += 1
                        n = 0
                        sum_d = 0.0
                        sum_sq = 0.0
                    prev_ok = ok
                    prev = value
