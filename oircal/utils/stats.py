"""
Robust statistics.
"""

from numba import njit
import numpy as np


__all__ = [
    "biweight_location",
    "biweight_midvariance",
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
