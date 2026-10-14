"""
Iterative sigma clipping with biweight statistics.
"""

from numba import njit, prange
import numpy as np

from .parallelization_utils import numba_thread_scope
from .stats import biweight_location, biweight_midvariance


__all__ = [
    "sigma_clip",
    "sigma_clip_stack",
]


def sigma_clip_stack(
    input_data: np.ndarray,
    mask: np.ndarray | None = None,
    sigma_thresh_low: float | None = 5.0,
    sigma_thresh_high: float | None = 5.0,
    min_keep: int = 3,
    maxiters: int = 50,
    max_cores: int = 1,
) -> np.ndarray:
    """
    Sigma clip a stack along its first axis, independently for each element of the others.

    The stack can be e.g. frames of shape (Nframes, Nrows, Ncols) or measurements of shape
    (Nobs,) or (Nobs, Norders). :func:`sigma_clip` is applied to each 1D slice along the first
    axis, using the biweight location and midvariance.

    Parameters
    ----------
    input_data
        Array of shape (N, ...) of values.
    mask
        Boolean array of the same shape as `input_data` where True is good. Values already
        False are ignored. Not modified. If None, all finite values of `input_data` are good.
        Default is None.
    sigma_thresh_low
        Clip values more than this many standard deviations below the center. If None, values
        are not clipped on the low side. Default is 5.0.
    sigma_thresh_high
        Clip values more than this many standard deviations above the center. If None, values
        are not clipped on the high side. Default is 5.0.
    min_keep
        Minimum number of good values to keep in each slice. Default is 3.
    maxiters
        Maximum number of clipping iterations. Default is 50.
    max_cores
        Maximum number of CPU cores to use. Default is 1.

    Returns
    -------
    mask : np.ndarray
        Boolean array of the same shape as `input_data` where True is good, with clipped
        values set to False.
    """
    N = input_data.shape[0]
    mask = np.isfinite(input_data) if mask is None else mask & np.isfinite(input_data)
    data2d = np.ascontiguousarray(input_data.reshape(N, -1))
    mask2d = np.ascontiguousarray(mask.reshape(N, -1))
    with numba_thread_scope(max_cores):
        _sigma_clip_stack_numba(
            data2d,
            mask2d,
            sigma_thresh_low,
            sigma_thresh_high,
            min_keep,
            maxiters,
        )
    return mask2d.reshape(input_data.shape)


@njit(nogil=True, parallel=True)
def _sigma_clip_stack_numba(
    data: np.ndarray,
    mask: np.ndarray,
    sigma_thresh_low: float | None,
    sigma_thresh_high: float | None,
    min_keep: int,
    maxiters: int,
):
    N, M = data.shape
    for m in prange(M):
        # Copy the slice's mask so the clip operates on contiguous memory, then write back
        mask_m = mask[:, m].copy()
        sigma_clip(
            data[:, m],
            mask_m,
            sigma_thresh_low=sigma_thresh_low,
            sigma_thresh_high=sigma_thresh_high,
            maxiters=maxiters,
            min_keep=min_keep,
        )
        mask[:, m] = mask_m


@njit(nogil=True)
def sigma_clip(
    x: np.ndarray,
    mask: np.ndarray,
    sigma_thresh_low: float | None = None,
    sigma_thresh_high: float | None = None,
    maxiters: int = 50,
    min_keep: int = 3,
) -> tuple[np.ndarray, float, float, int]:
    """
    Iteratively sigma clip a 1D array using the biweight location and midvariance.

    Each iteration computes the biweight location (center) and the square root of the
    biweight midvariance (standard deviation) of the good values, and clips good values
    beyond the thresholds. Clipping stops when no values are clipped, when 3 or fewer good
    values remain, or after `maxiters` iterations. At most enough values are clipped to keep
    `min_keep`, clipping the most extreme first.

    Parameters
    ----------
    x
        1D array of values.
    mask
        1D boolean array of the same shape as `x` where True is good. Values already False are
        ignored. Modified in place.
    sigma_thresh_low
        Clip values more than this many standard deviations below the center. If None, values
        are not clipped on the low side. Default is None.
    sigma_thresh_high
        Clip values more than this many standard deviations above the center. If None, values
        are not clipped on the high side. Default is None.
    maxiters
        Maximum number of clipping iterations. Default is 50.
    min_keep
        Minimum number of good values to keep. Default is 3.

    Returns
    -------
    mask : np.ndarray
        The input mask with clipped values set to False.
    center : float
        Biweight location of the good values in the last iteration.
    stddev : float
        Square root of the biweight midvariance of the good values in the last iteration.
    n_iters : int
        Number of iterations performed.
    """
    center = np.nan
    stddev = np.nan
    min_batch_size = 3

    n_iters = 0
    for _ in range(maxiters):
        n_iters += 1
        x_masked = x[mask]
        if len(x_masked) <= min_batch_size:
            break

        center = biweight_location(x_masked)
        stddev = np.sqrt(biweight_midvariance(x_masked, M=center))

        n_good = np.sum(mask)
        _mask_outliers(
            x,
            center,
            stddev,
            mask,
            sigma_thresh_low=sigma_thresh_low,
            sigma_thresh_high=sigma_thresh_high,
            min_keep=min_keep,
        )
        if np.sum(mask) == n_good:
            break

    return mask, center, stddev, n_iters


@njit(nogil=True)
def _mask_outliers(
    x: np.ndarray,
    center: float,
    stddev: float,
    mask: np.ndarray,
    sigma_thresh_low: float | None = None,
    sigma_thresh_high: float | None = None,
    min_keep: int = 3,
) -> np.ndarray:
    """
    Mask values beyond the sigma thresholds, keeping at least `min_keep` good values.

    Parameters
    ----------
    x
        1D array of values.
    center
        Center of the values (e.g. the biweight location).
    stddev
        Standard deviation of the values.
    mask
        1D boolean array where True is good. Updated in place.
    sigma_thresh_low
        Low-side threshold in standard deviations. If None, no low-side clipping. Default is
        None.
    sigma_thresh_high
        High-side threshold in standard deviations. If None, no high-side clipping. Default is
        None.
    min_keep
        Minimum number of good values to keep. Candidates are ranked from most to least
        extreme, and the least extreme are kept when needed. Default is 3.

    Returns
    -------
    mask : np.ndarray
        The input mask, updated in place.
    """
    do_low = sigma_thresh_low is not None
    do_high = sigma_thresh_high is not None
    if do_low:
        low_thresh = -sigma_thresh_low * stddev
    if do_high:
        high_thresh = sigma_thresh_high * stddev

    # Pass 1: count good values and collect clipping candidates, with their absolute
    # residuals and indices, so they can be ranked by how extreme they are
    n_good = 0
    cand_resid = np.empty(x.size, dtype=np.float64)
    cand_idx = np.empty(x.size, dtype=np.int64)
    n_cand = 0

    for k in range(x.size):
        if mask[k]:
            n_good += 1
            res = x[k] - center
            if (do_low and res < low_thresh) or (do_high and res > high_thresh):
                cand_resid[n_cand] = abs(res)
                cand_idx[n_cand] = k
                n_cand += 1

    if n_cand == 0:
        return mask

    # How many can be clipped while keeping min_keep
    max_clip = n_good - min_keep
    if max_clip <= 0:
        return mask
    n_clip = min(n_cand, max_clip)

    # Pass 2: apply the mask
    if n_clip == n_cand:
        for k in range(n_cand):
            mask[cand_idx[k]] = False
    else:
        # Clip the n_clip most extreme candidates
        resids = cand_resid[:n_cand].copy()
        clip_threshold = np.partition(resids, n_cand - n_clip)[n_cand - n_clip]
        clipped = 0
        for k in range(n_cand):
            if clipped >= n_clip:
                break
            if cand_resid[k] >= clip_threshold:
                mask[cand_idx[k]] = False
                clipped += 1

    return mask
