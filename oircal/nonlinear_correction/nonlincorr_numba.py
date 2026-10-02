from numba import njit, prange
from ..utils.array_utils import polyval1d
import numpy as np
from ..utils.parallelization_utils import numba_thread_scope

__all__ = ['correct_nonlinearity']

@njit(nogil=True, parallel=True)
def _correct_nonlinearity_numba(
    ramps : np.ndarray,
    dq_raw : np.ndarray,
    dq : np.ndarray,
    coeffs : np.ndarray,
    nonlin_max : np.ndarray,
    nonlin_dq : np.ndarray,
    dq_flags : tuple | None
):
    n_reads, ny, nx = ramps.shape
    for i in prange(ny):
        for j in range(nx):
            first_read = ramps[0, i, j]
            nl_corr_ramp = first_read + polyval1d(coeffs[:, i, j], ramps[:, i, j] - first_read)
            ramps[:, i, j] = nl_corr_ramp
            if dq_flags is not None:
                for k in range(n_reads):
                    if ramps[k, i, j] > nonlin_max[i, j]:
                        dq_raw[k, i, j] |= (dq_flags.NON_LINEAR | dq_flags.DO_NOT_USE)
            dq[i, j] |= nonlin_dq[i, j]


def correct_nonlinearity(
    ramps : np.ndarray,
    dq_raw : np.ndarray,
    dq : np.ndarray,
    coeffs : np.ndarray,
    nonlin_max : np.ndarray,
    nonlin_dq : np.ndarray,
    dq_flags : tuple | None = None,
    max_cores : int = 1,
):
    """
    Apply non-linearity correction to up-the-ramps. Correction is done in place.
    The first read is subtracted before applying the polynomial correction, but it is added back after such that the output ramp still includes the bias.
    The dq arrays are updated in place to flag any reads that exceed the non-linearity limit.

    Parameters
    ----------
    ramps : np.ndarray
        3D array of shape (Nreads, Ny, Nx) containing the ramps. Must be floating point.
    dq_raw : np.ndarray
        3D array of shape (Nreads, Ny, Nx) containing the raw DQ flags.
        Any reads beyong `nonlin_max` will be flagged as "non linear" and not trustworthy.
    dq : np.ndarray
        2D array of shape (Ny, Nx) containing the DQ flags to be updated with non-linearity flags.
    coeffs : np.ndarray
        3D array of shape (Ncoeffs, Ny, Nx) containing the polynomial coefficients of P().
        such that, "corrected ramp (DN)" = P("measured ramp (DN)").
    nonlin_max : np.ndarray
        2D array of shape (Ny, Nx) containing the maximum DN value for which the non-linearity
        correction is good to within 1%.
    nonlin_dq : np.ndarray
        2D array of shape (Ny, Nx) containing the DQ flags from the non-linearity model
        to be applied to all pixels.
    dq_flags : namedtuple, optional
        Named tuple of DQ bit values with integer fields NON_LINEAR and DO_NOT_USE.
        If None (default), reads beyond `nonlin_max` are not flagged in `dq_raw`;
        `nonlin_dq` is still propagated to `dq`.
    max_cores : int, optional
        Maximum number of CPU cores to use. Default is 1.

    Returns
    -------
    ramps : np.ndarray
        The non-linearity corrected ramps (same as input `ramps` but modified in place).
    dq_raw : np.ndarray
        The updated raw DQ array (same as input `dq_raw` but modified in place).
    dq : np.ndarray
        The updated DQ array (same as input `dq` but modified in place).
    """
    assert not np.issubdtype(ramps.dtype, np.integer), f"Ramps must be floating point, got {ramps.dtype}"
    with numba_thread_scope(max_cores):
        _correct_nonlinearity_numba(ramps, dq_raw, dq, coeffs, nonlin_max, nonlin_dq, dq_flags)
    return ramps, dq_raw, dq
