from numba import njit, prange
import numpy as np

from ..utils.array_utils import polyval1d
from ..utils.dq_utils import dq_flags_to_namedtuple
from ..utils.parallelization_utils import numba_thread_scope


__all__ = ["correct_nonlinearity"]


@njit(nogil=True, parallel=True)
def _correct_nonlinearity_numba(
    ramps: np.ndarray,
    dq_raw: np.ndarray,
    dq: np.ndarray,
    coeffs: np.ndarray,
    nonlin_max: np.ndarray,
    nonlin_dq: np.ndarray,
    flag_reads: bool,
    nonlin_flag: int,
    propagate_dq: bool,
):
    Nreads, Nrows, Ncols = ramps.shape
    for i in prange(Nrows):
        for j in range(Ncols):
            first_read = ramps[0, i, j]
            nl_corr_ramp = first_read + polyval1d(coeffs[:, i, j], ramps[:, i, j] - first_read)
            ramps[:, i, j] = nl_corr_ramp
            if flag_reads:
                for k in range(Nreads):
                    if ramps[k, i, j] > nonlin_max[i, j]:
                        dq_raw[k, i, j] |= nonlin_flag
            if propagate_dq:
                dq[i, j] |= nonlin_dq[i, j]


def correct_nonlinearity(
    input_data: np.ndarray,
    coeffs: np.ndarray,
    input_dq_raw: np.ndarray | None = None,
    input_dq: np.ndarray | None = None,
    nonlin_max: np.ndarray | None = None,
    nonlin_dq: np.ndarray | None = None,
    dq_flags: tuple | None = None,
    max_cores: int = 1,
):
    """
    Apply non-linearity correction to up-the-ramps. Correction is done in place.
    The first read is subtracted before applying the polynomial correction, but it is added back after such that the output ramp still includes the bias.
    Reads beyond `nonlin_max` are flagged in `input_dq_raw` only if `input_dq_raw`,
    `nonlin_max` and `dq_flags` are all given. `nonlin_dq` is propagated (bitwise OR) to
    `input_dq` only if both `input_dq` and `nonlin_dq` are given.

    Parameters
    ----------
    input_data
        3D array of shape (Nreads, Nrows, Ncols) containing the ramps. Must be floating point.
    coeffs
        3D array of shape (Ncoeffs, Nrows, Ncols) containing the polynomial coefficients of P().
        such that, "corrected ramp (DN)" = P("measured ramp (DN)").
    input_dq_raw
        3D array of shape (Nreads, Nrows, Ncols) containing the raw DQ flags.
        Any reads beyond `nonlin_max` will be flagged as "non linear" and not trustworthy.
        Default is None.
    input_dq
        2D array of shape (Nrows, Ncols) containing the DQ flags to be updated with non-linearity
        flags.
        Default is None.
    nonlin_max
        2D array of shape (Nrows, Ncols) of the DN level above which reads are flagged, e.g. the
        `nonlin_max` returned by `fit_nonlin_coefs`. It is compared against the corrected ramp,
        including the first read. Reads are never flagged where it is NaN. Default is None.
    nonlin_dq
        2D array of shape (Nrows, Ncols) containing the DQ flags from the non-linearity model
        to be applied to all pixels. Default is None.
    dq_flags
        Dict mapping DQ flag names to bit values, with keys NON_LINEAR and DO_NOT_USE.
        Default is None.
    max_cores
        Maximum number of CPU cores to use. Default is 1.

    Returns
    -------
    input_data : np.ndarray
        The non-linearity corrected ramps (same array as the input, modified in place).
    input_dq_raw : np.ndarray or None
        The updated raw DQ array (same array as the input, modified in place).
    input_dq : np.ndarray or None
        The updated DQ array (same array as the input, modified in place).
    """
    dq_flags = dq_flags_to_namedtuple(dq_flags)
    assert not np.issubdtype(input_data.dtype, np.integer), (
        f"Ramps must be floating point, got {input_data.dtype}"
    )
    flag_reads = input_dq_raw is not None and nonlin_max is not None and dq_flags is not None
    propagate_dq = input_dq is not None and nonlin_dq is not None
    nonlin_flag = dq_flags.NON_LINEAR | dq_flags.DO_NOT_USE if flag_reads else 0
    # Numba needs arrays of fixed type, so pass empty placeholders for unused inputs
    _dq_raw = input_dq_raw if flag_reads else np.zeros((1, 1, 1), dtype=np.uint8)
    _nonlin_max = nonlin_max if flag_reads else np.zeros((1, 1), dtype=np.float32)
    _dq = input_dq if propagate_dq else np.zeros((1, 1), dtype=np.uint32)
    _nonlin_dq = nonlin_dq if propagate_dq else np.zeros((1, 1), dtype=np.uint32)
    with numba_thread_scope(max_cores):
        _correct_nonlinearity_numba(
            input_data,
            _dq_raw,
            _dq,
            coeffs,
            _nonlin_max,
            _nonlin_dq,
            flag_reads,
            nonlin_flag,
            propagate_dq,
        )
    return input_data, input_dq_raw, input_dq
