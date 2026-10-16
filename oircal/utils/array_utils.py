import sys
from typing import Literal

from numba import njit
import numpy as np


__all__ = [
    "convert_endianness",
    "polyval1d",
]


def convert_endianness(
    arr: np.ndarray,
    output_endianness: Literal["little", "big"] = "little",
) -> np.ndarray:
    """
    Convert an array to the given byte order, keeping its values.

    oircal functions need little-endian arrays (Numba does not support non-native byte order),
    but data read from FITS files are big-endian.

    Parameters
    ----------
    arr
        Array of any shape and dtype.
    output_endianness
        Byte order of the output, ``"little"`` or ``"big"``.

    Returns
    -------
    out : np.ndarray
        Array with the same values in the requested byte order. `arr` itself is returned if it
        is already in that byte order or its dtype has no byte order (e.g. ``uint8``).
    """
    byteorder = arr.dtype.byteorder
    if byteorder == "|":
        return arr
    if byteorder == "=":
        current_endianness = sys.byteorder
    elif byteorder == "<":
        current_endianness = "little"
    else:
        current_endianness = "big"
    if current_endianness != output_endianness:
        return arr.byteswap().view(arr.dtype.newbyteorder(output_endianness))
    return arr


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
