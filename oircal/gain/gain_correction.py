from numba import njit, prange
import numpy as np

from ..utils.parallelization_utils import numba_thread_scope


__all__ = ["correct_gain"]


def correct_gain(
    input_data: np.ndarray,
    gain: np.ndarray,
    input_err: np.ndarray | None = None,
    input_dq: np.ndarray | None = None,
    gain_err: np.ndarray | None = None,
    gain_dq: np.ndarray | None = None,
    input_var_poisson: np.ndarray | None = None,
    input_var_rnoise: np.ndarray | None = None,
    max_cores: int = 1,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Convert rates from DN/s to e-/s with the gain, in place.

    `input_data` (and `input_err`) are multiplied by the gain, and the variances
    `input_var_poisson` and `input_var_rnoise` by the gain squared. All must be floating
    point, since the results are written back into them. If `input_err` is given, it is
    propagated through the product; the gain contributes to the error only if `gain_err` is
    also given, otherwise the gain is treated as noiseless. `gain_err` does not enter the
    variances, so with it the variances no longer sum to ``input_err**2``. DQ propagation
    (bitwise OR) runs only if both `input_dq` and `gain_dq` are given.

    Parameters
    ----------
    input_data
        2D array of shape (Nrows, Ncols) of rates (DN/s). Must be floating point. Modified in
        place.
    gain
        2D array of shape (Nrows, Ncols) of the gain (e-/DN).
    input_err
        2D array of shape (Nrows, Ncols) of rate uncertainties (DN/s). Must be floating point.
        Modified in place. Default is None.
    input_dq
        2D array of shape (Nrows, Ncols) of per-pixel DQ flags. Modified in place. Default is
        None.
    gain_err
        2D array of shape (Nrows, Ncols) of gain uncertainties (e-/DN), e.g. the square root of
        ``cov[0, 0]`` from :func:`.fit_gain_and_readnoise`. Default is None.
    gain_dq
        2D array of shape (Nrows, Ncols) of DQ flags from the gain calibration. Default is None.
    input_var_poisson
        2D array of shape (Nrows, Ncols) of the Poisson variance of the rates ((DN/s)^2).
        Modified in place. Default is None.
    input_var_rnoise
        2D array of shape (Nrows, Ncols) of the read noise variance of the rates ((DN/s)^2).
        Modified in place. Default is None.
    max_cores
        Maximum number of CPU cores to use. Default is 1.

    Returns
    -------
    input_data, input_err, input_dq
        The (in-place modified) input arrays; `input_err` and `input_dq` are returned as
        passed, including None. `input_var_poisson` and `input_var_rnoise` are modified in
        place but not returned.
    """
    for name, arr in (
        ("Rates", input_data),
        ("Rate errors", input_err),
        ("Poisson variances", input_var_poisson),
        ("Read noise variances", input_var_rnoise),
    ):
        assert arr is None or not np.issubdtype(arr.dtype, np.integer), (
            f"{name} must be floating point, got {arr.dtype}"
        )
    with numba_thread_scope(max_cores):
        _correct_gain_numba(
            input_data,
            gain,
            input_err,
            gain_err,
            input_dq,
            gain_dq,
            input_var_poisson,
            input_var_rnoise,
        )
    return input_data, input_err, input_dq


@njit(nogil=True, parallel=True)
def _correct_gain_numba(
    input_rate,
    gain,
    input_err,
    gain_err,
    input_dq,
    gain_dq,
    input_var_poisson,
    input_var_rnoise,
):
    Nrows, Ncols = input_rate.shape
    for j in prange(Nrows):
        for i in range(Ncols):
            g = gain[j, i]
            r = input_rate[j, i]
            input_rate[j, i] = r * g

            # Error propagation for r * g:
            # sigma^2 = (g * sigma_r)^2 + (r * sigma_g)^2
            if input_err is not None and gain_err is not None:
                input_err[j, i] = np.sqrt((g * input_err[j, i]) ** 2 + (r * gain_err[j, i]) ** 2)
            elif input_err is not None:
                input_err[j, i] *= abs(g)

            if input_var_poisson is not None:
                input_var_poisson[j, i] *= g * g
            if input_var_rnoise is not None:
                input_var_rnoise[j, i] *= g * g

            if input_dq is not None and gain_dq is not None:
                input_dq[j, i] |= gain_dq[j, i]
