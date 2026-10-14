from numba import njit, prange
import numpy as np

from ..utils.dq_utils import dq_flags_to_namedtuple
from ..utils.parallelization_utils import numba_thread_scope
from .combine import METHODS, combine, combine_variance_1d


__all__ = ["coadd"]


def coadd(
    input_data: np.ndarray,
    input_err: np.ndarray | None = None,
    input_dq: np.ndarray | None = None,
    input_var_poisson: np.ndarray | None = None,
    input_var_rnoise: np.ndarray | None = None,
    input_mid_time: np.ndarray | None = None,
    dq_flags: dict | None = None,
    method: str = "wmean",
    err_method: str = "se",
    do_sigma_clip: bool = False,
    sigma_thresh_low: float | None = 5.0,
    sigma_thresh_high: float | None = 5.0,
    min_keep: int | float = 3,
    max_cores: int = 1,
) -> dict[str, np.ndarray | None]:
    """
    Coadd a stack of data products, such as frames or spectra, along its first axis.

    The stack has shape (N, ...), e.g. frames of shape (Nframes, Nrows, Ncols) or spectra of
    shape (Nspectra, Npix) or (Nspectra, Norders, Npix), and every element of the trailing
    axes (e.g. each pixel) is coadded independently. The good values of each element are those
    with `input_dq` equal to 0, finite data and (if `input_err` is given) finite errors,
    optionally sigma clipped. Their data and errors are combined with :func:`.combine`
    according to `method` ('wmean', 'mean' or 'median') and `err_method` ('se' or 'scatter').
    The variance components are combined to match `method` (1 / sum(1 / var) for 'wmean',
    sum(var) / N**2 for 'mean', and that times pi / 2 for 'median'), DQ flags with a bitwise
    OR, and mid times with a mean. With one good value, it is used as is. With none, the output
    is NaN and flagged DO_NOT_USE. A stack of one (N = 1) is returned unchanged. The inputs
    are not modified.

    Parameters
    ----------
    input_data
        Array of shape (N, ...) of data.
    input_err
        Array of the same shape as `input_data` of uncertainties. Required for `method`
        'wmean' and for `err_method` 'se'. Default is None.
    input_dq
        Array of the same shape as `input_data` of DQ flags. Nonzero values exclude the value.
        If None, all values are good. Default is None.
    input_var_poisson
        Array of the same shape as `input_data` of Poisson variances. Default is None.
    input_var_rnoise
        Array of the same shape as `input_data` of read noise variances. Default is None.
    input_mid_time
        Array of the same shape as `input_data` of mid times. Default is None.
    dq_flags
        Dict mapping DQ flag names to bit values, with key DO_NOT_USE. If None, elements with
        no good values are not flagged. Default is None.
    method
        How to combine the good values: 'wmean' (inverse-variance weighted mean), 'mean' or
        'median'. Default is 'wmean'.
    err_method
        How to compute the uncertainty: 'se' (standard error propagated from `input_err`) or
        'scatter' (measured from the spread of the good values). Default is 'se'.
    do_sigma_clip
        Sigma clip each element's good values before coadding, using the biweight location and
        midvariance (see :func:`.sigma_clip_stack`). Only done if N is more than 3. Default
        is False.
    sigma_thresh_low
        Low-side sigma clipping threshold in standard deviations. If None, no low-side
        clipping. Default is 5.0.
    sigma_thresh_high
        High-side sigma clipping threshold in standard deviations. If None, no high-side
        clipping. Default is 5.0.
    min_keep
        Minimum number of good values to keep per element when sigma clipping. A value below 1
        is a fraction of N. Default is 3.
    max_cores
        Maximum number of CPU cores to use. Default is 1.

    Returns
    -------
    output : dict[str, np.ndarray or None]
        Arrays of shape ``input_data.shape[1:]`` with keys data, err, var_poisson, var_rnoise,
        dq and mid_time. A key is None if it cannot be computed from the inputs given (dq is
        always returned).
    """
    dq_flags = dq_flags_to_namedtuple(dq_flags)
    N = input_data.shape[0]
    shape = input_data.shape[1:]

    # A stack of one is already coadded
    if N == 1:
        return {
            "data": input_data[0].copy(),
            "err": None if input_err is None else input_err[0].copy(),
            "var_poisson": None if input_var_poisson is None else input_var_poisson[0].copy(),
            "var_rnoise": None if input_var_rnoise is None else input_var_rnoise[0].copy(),
            "dq": np.zeros(shape, dtype=np.uint32) if input_dq is None else input_dq[0].copy(),
            "mid_time": None if input_mid_time is None else input_mid_time[0].copy(),
        }

    # Data and errors, and the values used for each element
    data_out, err_out, good = combine(
        input_data,
        input_err=input_err,
        mask=None if input_dq is None else input_dq == 0,
        method=method,
        err_method=err_method,
        do_sigma_clip=do_sigma_clip,
        sigma_thresh_low=sigma_thresh_low,
        sigma_thresh_high=sigma_thresh_high,
        min_keep=min_keep,
        max_cores=max_cores,
    )

    # DQ, variance components and mid times over the same values, with the trailing axes
    # flattened to (N, M)
    good2d = good.reshape(N, -1)
    dq_out = np.zeros(good2d.shape[1], dtype=np.uint32)
    do_not_use = 0 if dq_flags is None else dq_flags.DO_NOT_USE
    var_poisson_out = var_rnoise_out = mid_time_out = None
    with numba_thread_scope(max_cores):
        if input_dq is None:
            _flag_empty_numba(good2d, do_not_use, dq_out)
        else:
            _combine_dq_numba(input_dq.reshape(N, -1), good2d, do_not_use, dq_out)
        if input_var_poisson is not None:
            var_poisson_out = _combine_variance(input_var_poisson, good2d, method)
        if input_var_rnoise is not None:
            var_rnoise_out = _combine_variance(input_var_rnoise, good2d, method)
        if input_mid_time is not None:
            mid_time_out = np.empty(good2d.shape[1])
            _mean_numba(input_mid_time.reshape(N, -1), good2d, mid_time_out)

    return {
        "data": data_out,
        "err": err_out,
        "var_poisson": None if var_poisson_out is None else var_poisson_out.reshape(shape),
        "var_rnoise": None if var_rnoise_out is None else var_rnoise_out.reshape(shape),
        "dq": dq_out.reshape(shape),
        "mid_time": None if mid_time_out is None else mid_time_out.reshape(shape),
    }


def _combine_variance(var: np.ndarray, good2d: np.ndarray, method: str) -> np.ndarray:
    out = np.empty(good2d.shape[1])
    _combine_variance_numba(var.reshape(var.shape[0], -1), good2d, METHODS[method], out)
    return out.astype(var.dtype)


@njit(nogil=True, parallel=True)
def _combine_dq_numba(dq: np.ndarray, good: np.ndarray, do_not_use: int, dq_out: np.ndarray):
    for m in prange(dq.shape[1]):
        idx = np.where(good[:, m])[0]
        if len(idx) == 0:
            dq_out[m] = do_not_use
        else:
            result = np.uint32(0)
            for k in idx:
                result |= dq[k, m]
            dq_out[m] = result


@njit(nogil=True, parallel=True)
def _flag_empty_numba(good: np.ndarray, do_not_use: int, dq_out: np.ndarray):
    for m in prange(good.shape[1]):
        if not np.any(good[:, m]):
            dq_out[m] = do_not_use


@njit(nogil=True, parallel=True)
def _combine_variance_numba(var: np.ndarray, good: np.ndarray, method: int, out: np.ndarray):
    for m in prange(var.shape[1]):
        out[m] = combine_variance_1d(var[:, m], good[:, m], method)


@njit(nogil=True, parallel=True)
def _mean_numba(x: np.ndarray, good: np.ndarray, out: np.ndarray):
    for m in prange(x.shape[1]):
        idx = np.where(good[:, m])[0]
        out[m] = np.mean(x[idx, m]) if len(idx) > 0 else np.nan
