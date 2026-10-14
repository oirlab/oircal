from numba import njit
import numpy as np


__all__ = [
    "polyval1d",
]


@njit(nogil=True)
def polyval1d(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Evaluate a polynomial with given coefficients at specified points using the Horner method.

    Parameters
    ----------
    coeffs
        1D array of shape (Ncoeffs,) of polynomial coefficients, lowest degree first (the
        reverse of `np.polyval`).
    x
        Array of any shape of points at which to evaluate the polynomial.

    Returns
    -------
    v : np.ndarray
        Array of the same shape as `x` and the dtype of `coeffs`, the polynomial evaluated at
        each point.
    """
    v = np.zeros(x.shape, dtype=coeffs.dtype)
    for coeff in coeffs[::-1]:
        v = v * x + coeff
    return v
