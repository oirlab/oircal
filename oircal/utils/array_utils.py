import numpy as np
from numba import njit

__all__ = [
    "polyval1d",
]


@njit(nogil=True)
def polyval1d(coeffs : np.ndarray, x : np.ndarray) -> np.ndarray:
    """
    Evaluate a polynomial with given coefficients at specified points using the Horner method.

    Parameters
    ----------
    coeffs : np.ndarray
        1D array of shape (n_coeffs,) of polynomial coefficients, ordered from lowest degree to highest (NOTE: numpy is reversed).
    x : np.ndarray
        Array of any shape of points at which to evaluate the polynomial.
    """
    v = np.zeros(x.shape, dtype=coeffs.dtype)
    for coeff in coeffs[::-1]:
        v = v * x + coeff
    return v
