"""
Robust statistics.
"""

from numba import njit
import numpy as np
from scipy.interpolate import interp1d


__all__ = [
    "biweight_location",
    "biweight_midvariance",
    "get_err_from_posterior",
    "mad",
]


@njit(nogil=True)
def mad(x: np.ndarray, M: float | None = None) -> float:
    """
    Calculate the median absolute deviation of an array.

    Parameters
    ----------
    x
        Array of values.
    M
        Median of `x`. If None, it is calculated. Default is None.

    Returns
    -------
    mad : float
        Median absolute deviation of `x` about `M`, ignoring NaNs.
    """
    if M is None:
        M = np.nanmedian(x)
    return np.nanmedian(np.abs(x - M))


@njit(nogil=True)
def biweight_location(x: np.ndarray, M: float | None = None, c: float = 6.0) -> float:
    """
    Calculate the biweight location of an array.

    See https://docs.astropy.org/en/stable/api/astropy.stats.biweight_location.html.

    Parameters
    ----------
    x
        Array of values.
    M
        Initial guess of the location. If None, the median of `x` is used. Default is None.
    c
        Tuning constant of the biweight estimator. Default is 6.0.

    Returns
    -------
    location : float
        Biweight location of `x`. `M` if the MAD of `x` is 0 or not finite.
    """
    x = x.ravel()
    if M is None:
        M = np.nanmedian(x)

    mad_val = mad(x, M)
    if mad_val == 0.0 or not np.isfinite(mad_val):
        return M

    num = 0.0
    den = 0.0
    for i in range(x.size):
        u = (x[i] - M) / (c * mad_val)
        if np.abs(u) < 1:
            w = (1 - u**2) ** 2
            num += (x[i] - M) * w
            den += w

    if den == 0:
        return M
    return M + num / den


@njit(nogil=True)
def biweight_midvariance(x: np.ndarray, M: float | None = None, c: float = 9.0) -> float:
    """
    Calculate the biweight midvariance of an array.

    See https://docs.astropy.org/en/stable/api/astropy.stats.biweight_midvariance.html.

    Parameters
    ----------
    x
        Array of values.
    M
        Location about which to calculate the variance. If None, the median of `x` is used.
        Default is None.
    c
        Tuning constant of the biweight estimator. Default is 9.0.

    Returns
    -------
    midvariance : float
        Biweight midvariance of `x`. The MAD of `x` if it is 0 or not finite.
    """
    x = x.ravel()
    if M is None:
        M = np.nanmedian(x)

    d = x - M
    mad_val = mad(x, M=M)
    if mad_val == 0.0 or not np.isfinite(mad_val):
        return mad_val

    u = d / (c * mad_val)
    u2 = u * u

    good = np.where((np.abs(u) < 1.0) & np.isfinite(u))[0]
    n_good = len(good)

    f1 = 0.0
    f2 = 0.0
    for i in range(n_good):
        ii = good[i]
        t = 1 - u2[ii]
        f1 += d[ii] * d[ii] * t**4
        f2 += t * (1.0 - 5.0 * u2[ii])
    return n_good * f1 / (f2 * f2)


def get_err_from_posterior(x: np.ndarray, posterior: np.ndarray) -> tuple[float, float, float]:
    """
    Return the mode of a sampled 1D posterior and its left and right 68% errors.

    Copied from BREADS (https://github.com/jruffio/breads).

    Parameters
    ----------
    x
        1D array of the points the posterior is sampled at, increasing.
    posterior
        1D array of the posterior at `x`.

    Returns
    -------
    mode : float
        The value of `x` at the maximum of the posterior.
    left_err : float
        Distance from the mode to the lower bound of the 68% interval.
    right_err : float
        Distance from the mode to the upper bound of the 68% interval.
    """
    ind = np.argsort(posterior)
    cum_posterior = np.zeros(np.shape(posterior))
    cum_posterior[ind] = np.cumsum(posterior[ind])
    cum_posterior = cum_posterior / np.max(cum_posterior)
    argmax_post = np.argmax(cum_posterior)
    if len(x[0 : np.min([argmax_post + 1, len(x)])]) < 2:
        lx = x[0]
    else:
        tmp_cumpost = cum_posterior[0 : np.min([argmax_post + 1, len(x)])]
        tmp_x = x[0 : np.min([argmax_post + 1, len(x)])]
        deriv_tmp_cumpost = np.insert(
            tmp_cumpost[1::] - tmp_cumpost[0 : np.size(tmp_cumpost) - 1],
            np.size(tmp_cumpost) - 1,
            0,
        )
        try:
            whereinflection = np.where(deriv_tmp_cumpost < 0)[0][0]
            where2keep = np.where(
                (tmp_x <= tmp_x[whereinflection]) + (tmp_cumpost >= tmp_cumpost[whereinflection]),
            )
            tmp_cumpost = tmp_cumpost[where2keep]
            tmp_x = tmp_x[where2keep]
        except Exception:
            pass
        lf = interp1d(tmp_cumpost, tmp_x, bounds_error=False, fill_value=x[0])
        lx = lf(1 - 0.6827)
    if len(x[argmax_post::]) < 2:
        rx = x[-1]
    else:
        tmp_cumpost = cum_posterior[argmax_post::]
        tmp_x = x[argmax_post::]
        deriv_tmp_cumpost = np.insert(
            tmp_cumpost[1::] - tmp_cumpost[0 : np.size(tmp_cumpost) - 1],
            np.size(tmp_cumpost) - 1,
            0,
        )
        try:
            whereinflection = np.where(deriv_tmp_cumpost > 0)[0][0]
            where2keep = np.where(
                (tmp_x >= tmp_x[whereinflection]) + (tmp_cumpost >= tmp_cumpost[whereinflection]),
            )
            tmp_cumpost = tmp_cumpost[where2keep]
            tmp_x = tmp_x[where2keep]
        except Exception:
            pass
        rf = interp1d(tmp_cumpost, tmp_x, bounds_error=False, fill_value=x[-1])
        rx = rf(1 - 0.6827)
    return x[argmax_post], x[argmax_post] - lx, rx - x[argmax_post]
