"""
Combine a stack of measurements along its first axis.
"""

from numba import njit, prange
import numpy as np

from ..utils.parallelization_utils import numba_thread_scope
from ..utils.sigma_clipping import sigma_clip_stack


__all__ = ["combine"]

METHODS = {"wmean": 0, "mean": 1, "median": 2}
ERR_METHODS = {"se": 0, "scatter": 1}

# Variance of the median of N Gaussian values relative to the variance of their mean
MEDIAN_VAR_FACTOR = np.pi / 2


def combine(
    input_data: np.ndarray,
    input_err: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    method: str = "wmean",
    err_method: str = "se",
    do_sigma_clip: bool = False,
    sigma_thresh_low: float | None = 5.0,
    sigma_thresh_high: float | None = 5.0,
    min_keep: int | float = 3,
    max_cores: int = 1,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Combine a stack of measurements along its first axis.

    The stack can be e.g. frames of shape (Nframes, Nrows, Ncols), or measurements such as RVs
    of shape (Nobs,) or (Nobs, Norders). Each 1D slice along the first axis is combined
    independently, using its good values: those with `mask` True, finite data and (if
    `input_err` is given) finite errors, optionally sigma clipped. With N good values:

    `method` sets the combined value:

    - 'wmean': inverse-variance weighted mean, with weights w = ``1 / input_err**2``.
    - 'mean': mean.
    - 'median': median.

    `err_method` sets its uncertainty:

    - 'se': standard error propagated from `input_err`. For 'wmean' it is 1 / sqrt(sum(w));
      for 'mean' it is ``sqrt(sum(input_err**2))`` / N; for 'median' it is that of the mean
      times sqrt(pi / 2), the large-N ratio for Gaussian data. None if `input_err` is not
      given.
    - 'scatter': measured from the spread of the good values. For 'wmean' it is
      sqrt(sum(w * (x - wmean)**2) / ((N - 1) * sum(w))); for 'mean' it is the sample
      standard deviation (ddof=1) / sqrt(N); for 'median' it is that of the mean times
      sqrt(pi / 2). With N = 1 the value's own `input_err` is used, or NaN without
      `input_err`.

    With a single good value, it is used as is. With none, the value and error are NaN.

    Parameters
    ----------
    input_data
        Array of shape (N, ...) of measurements.
    input_err
        Array of the same shape as `input_data` of uncertainties. Required for `method`
        'wmean' and for `err_method` 'se'. Default is None.
    mask
        Boolean array of the same shape as `input_data` where True is good. Not modified. If
        None, all finite values are good. Default is None.
    method
        How to combine the good values: 'wmean', 'mean' or 'median'. Default is 'wmean'.
    err_method
        How to compute the uncertainty: 'se' (propagated standard error) or 'scatter'
        (measured from the spread of the good values). Default is 'se'.
    do_sigma_clip
        Sigma clip each slice's good values before combining, using the biweight location and
        midvariance (see :func:`.sigma_clip_stack`). Only done if N is more than 3. Default
        is False.
    sigma_thresh_low
        Low-side sigma clipping threshold in standard deviations. If None, no low-side
        clipping. Default is 5.0.
    sigma_thresh_high
        High-side sigma clipping threshold in standard deviations. If None, no high-side
        clipping. Default is 5.0.
    min_keep
        Minimum number of good values to keep per slice when sigma clipping. A value below 1
        is a fraction of N. Default is 3.
    max_cores
        Maximum number of CPU cores to use. Default is 1.

    Returns
    -------
    data : np.ndarray
        Array of shape ``input_data.shape[1:]`` of combined values (0-dimensional for 1D input).
    err : np.ndarray or None
        Array of the same shape as `data` of uncertainties. None for `err_method` 'se' without
        `input_err`.
    mask : np.ndarray
        Boolean array of the same shape as `input_data` of the values used (True), after
        sigma clipping. Use it to combine other quantities over the same values.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {list(METHODS)}, got {method!r}")
    if err_method not in ERR_METHODS:
        raise ValueError(f"err_method must be one of {list(ERR_METHODS)}, got {err_method!r}")
    if method == "wmean" and input_err is None:
        raise ValueError("method 'wmean' requires input_err")

    N = input_data.shape[0]
    good = np.isfinite(input_data)
    if mask is not None:
        good &= mask
    if input_err is not None:
        good &= np.isfinite(input_err)

    if do_sigma_clip and N > 3:
        if min_keep < 1:
            min_keep = int(np.ceil(min_keep * N))
        good = sigma_clip_stack(
            input_data,
            mask=good,
            sigma_thresh_low=sigma_thresh_low,
            sigma_thresh_high=sigma_thresh_high,
            min_keep=int(min_keep),
            max_cores=max_cores,
        )

    has_err = input_err is not None
    want_err = has_err or err_method == "scatter"
    data2d = input_data.reshape(N, -1)
    # Placeholder with the right shape when there are no errors; never read
    err2d = input_err.reshape(N, -1) if has_err else data2d
    good2d = good.reshape(N, -1)
    data_out = np.empty(data2d.shape[1], dtype=np.float64)
    err_out = np.empty(data2d.shape[1], dtype=np.float64)

    with numba_thread_scope(max_cores):
        _combine_numba(
            data2d,
            err2d,
            good2d,
            has_err,
            METHODS[method],
            ERR_METHODS[err_method],
            data_out,
            err_out,
        )

    shape = input_data.shape[1:]
    dtype = np.result_type(input_data.dtype, np.float32)
    data_out = data_out.astype(dtype).reshape(shape)
    err_out = err_out.astype(dtype).reshape(shape) if want_err else None
    return data_out, err_out, good


@njit(nogil=True, parallel=True)
def _combine_numba(
    data: np.ndarray,
    err: np.ndarray,
    good: np.ndarray,
    has_err: bool,
    method: int,
    err_method: int,
    data_out: np.ndarray,
    err_out: np.ndarray,
):
    M = data.shape[1]
    for m in prange(M):
        data_out[m], err_out[m] = combine_1d(
            data[:, m],
            err[:, m],
            good[:, m],
            has_err,
            method,
            err_method,
        )


@njit(nogil=True)
def combine_1d(
    x: np.ndarray,
    err: np.ndarray,
    good: np.ndarray,
    has_err: bool,
    method: int,
    err_method: int,
) -> tuple[float, float]:
    """
    Combine the good values of a 1D array (see :func:`combine` for the methods).

    Parameters
    ----------
    x
        1D array of values.
    err
        1D array of uncertainties of `x`. Only read if `has_err` is True.
    good
        1D boolean array where True is good.
    has_err
        Whether `err` holds uncertainties.
    method
        ``METHODS[name]`` of the combine method.
    err_method
        ``ERR_METHODS[name]`` of the uncertainty method.

    Returns
    -------
    value : float
        Combined value, NaN if there are no good values.
    error : float
        Its uncertainty, NaN if there are no good values or it cannot be computed.
    """
    idx = np.where(good)[0]
    n = len(idx)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return x[idx[0]], err[idx[0]] if has_err else np.nan

    v = x[idx].astype(np.float64)
    if method == 0:  # inverse-variance weighted mean
        w = 1 / err[idx].astype(np.float64) ** 2
        value = np.sum(w * v) / np.sum(w)
        if err_method == 0:
            error = 1 / np.sqrt(np.sum(w))
        else:
            error = np.sqrt(np.sum(w * (v - value) ** 2) / ((n - 1) * np.sum(w)))
        return value, error

    # Mean or median; the median's variance is that of the mean times MEDIAN_VAR_FACTOR
    scale = MEDIAN_VAR_FACTOR if method == 2 else 1.0
    value = np.median(v) if method == 2 else np.mean(v)
    if err_method == 0 and not has_err:
        error = np.nan
    elif err_method == 0:
        error = np.sqrt(scale * np.sum(err[idx].astype(np.float64) ** 2)) / n
    else:
        mean = np.mean(v)
        error = np.sqrt(scale * np.sum((v - mean) ** 2) / (n - 1) / n)
    return value, error


@njit(nogil=True)
def combine_variance_1d(var: np.ndarray, good: np.ndarray, method: int) -> float:
    """
    Combine variance components (e.g. Poisson or read noise) to match :func:`combine`.

    For 'wmean' the variances are combined as 1 / sum(1 / var); for 'mean' as
    sum(var) / N**2; for 'median' as that of the mean times pi / 2.

    Parameters
    ----------
    var
        1D array of variances.
    good
        1D boolean array where True is good.
    method
        ``METHODS[name]`` of the combine method.

    Returns
    -------
    variance : float
        Combined variance, NaN if there are no good values.
    """
    idx = np.where(good)[0]
    n = len(idx)
    if n == 0:
        return np.nan
    if n == 1:
        return var[idx[0]]
    v = var[idx].astype(np.float64)
    if method == 0:
        return 1 / np.sum(1 / v)
    scale = MEDIAN_VAR_FACTOR if method == 2 else 1.0
    return scale * np.sum(v) / n**2
