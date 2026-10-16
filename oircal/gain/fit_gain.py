import ctypes
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray

import numpy as np
from progressbar import progressbar

from ..ramp_fitting import fitramp_Brandt
from ..utils.dq_utils import dq_flags_to_namedtuple
from ..utils.parallelization_utils import numba_thread_scope, shared_array_to_numpy
from ..utils.stats import get_err_from_posterior
from . import fit_gain_Brandt


__all__ = ["fit_gain_and_readnoise"]

# Default grid offsets around the initial values, per method
_QUAD_DELTA_GAINS = np.array([-0.2, 0.0, 0.2])
_QUAD_DELTA_RNS = np.array([-0.5, 0.0, 0.5])
_GRID_DELTA_GAINS = np.linspace(-1, 1, 51)
_GRID_DELTA_RNS = np.linspace(-10, 10, 51)


def fit_gain_and_readnoise(
    times: np.ndarray,
    input_data: np.ndarray,
    init_gain: np.ndarray | float,
    init_rn: np.ndarray | float,
    input_dq_raw: np.ndarray | None = None,
    dq_flags: tuple | None = None,
    fit_gain: bool = True,
    fit_rn: bool = True,
    method: str = "quad",
    delta_gains: np.ndarray | None = None,
    delta_rns: np.ndarray | None = None,
    countratesidentical: bool = False,
    mode: str = "per_pixel",
    max_cores: int = 1,
) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray]:
    """
    Fit the detector gain and/or read noise from a set of up-the-ramp exposures.

    Maximizes the likelihood of the ramps, marginalized over each ramp's count rate, as a
    function of gain and read noise (Brandt 2025, PASP, 137, 125006). Either parameter can be
    held at its initial value: fit the read noise alone from darks (``fit_gain=False``), or
    the gain alone with a known read noise (``fit_rn=False``). Read differences touching a
    read flagged DO_NOT_USE or SATURATED, or ending on a read flagged JUMP_DET, are left out.

    Parameters
    ----------
    times
        1D array of shape (Nreads,) of the read times (s).
    input_data
        4D array of shape (Nfiles, Nreads, Nrows, Ncols) of ramps (DN).
    init_gain
        Initial gain (e-/DN), a float or a 2D array of shape (Nrows, Ncols). Fixed if
        `fit_gain` is False.
    init_rn
        Initial single-read read noise (DN), a float or a 2D array of shape (Nrows, Ncols).
        Fixed if `fit_rn` is False.
    input_dq_raw
        4D array of shape (Nfiles, Nreads, Nrows, Ncols) of per-read DQ flags. Default is None.
    dq_flags
        Dict mapping DQ flag names to bit values, with keys DO_NOT_USE, SATURATED and JUMP_DET.
        If `input_dq_raw` or `dq_flags` is None, all read differences are used. Default is None.
    fit_gain
        If True, fit the gain. Default is True.
    fit_rn
        If True, fit the read noise. Default is True.
    method
        ``"quad"`` fits a quadratic form to the log likelihood on a 3-point grid around the
        initial values, then again around the first result. ``"grid"`` evaluates the likelihood
        on a dense grid and takes the errors from the marginal posteriors. ``"both"`` runs the
        grid search, then the quadratic fit around its peak. Default is ``"quad"``.
    delta_gains
        1D array of gain offsets (e-/DN) from `init_gain` to evaluate. Must have 3 elements
        ``[-d, 0, d]`` for ``"quad"`` and more than 3 for ``"grid"`` and ``"both"``. Default is
        ``[-0.2, 0, 0.2]`` for ``"quad"`` and ``np.linspace(-1, 1, 51)`` otherwise.
    delta_rns
        1D array of read noise offsets (DN) from `init_rn`, as for `delta_gains`. Default is
        ``[-0.5, 0, 0.5]`` for ``"quad"`` and ``np.linspace(-10, 10, 51)`` otherwise.
    countratesidentical
        If True, assume every ramp of a pixel has the same count rate. Default is False.
    mode
        ``"per_pixel"`` fits each pixel; ``"full_detector"`` fits one gain and read noise to all
        pixels at once. Default is ``"per_pixel"``.
    max_cores
        Number of processes for ``"per_pixel"``. Default is 1.

    Returns
    -------
    gain : np.ndarray or float
        Best-fit gain (e-/DN), 2D float32 array of shape (Nrows, Ncols) for ``"per_pixel"``,
        else a float.
    rn : np.ndarray or float
        Best-fit read noise (DN), same shape as `gain`.
    cov : np.ndarray
        Covariance of (gain, rn), 4D array of shape (2, 2, Nrows, Ncols) for ``"per_pixel"``,
        else 2D of shape (2, 2). Entries of a fixed parameter are NaN, as are the off-diagonal
        terms unless both are fit with ``"quad"`` or ``"both"``. With ``"grid"``, the diagonal
        holds the errors rather than the variances, and `rn` is NaN unless it is fit.

    References
    ----------
    Brandt, T. D. (2025), "Computing the Electronic Gain for Detectors Read Out Up-the-ramp",
    PASP, 137, 125006.
    """
    if not (fit_gain or fit_rn):
        raise ValueError("At least one of fit_gain and fit_rn must be True.")
    if method not in ("quad", "grid", "both"):
        raise ValueError(f"method must be 'quad', 'grid' or 'both', got {method!r}.")
    if mode not in ("per_pixel", "full_detector"):
        raise ValueError(f"mode must be 'per_pixel' or 'full_detector', got {mode!r}.")
    if method == "grid" and not fit_gain:
        raise ValueError("method 'grid' needs fit_gain=True; use 'quad' or 'both' to fit rn alone.")

    if delta_gains is None:
        delta_gains = _QUAD_DELTA_GAINS if method == "quad" else _GRID_DELTA_GAINS
    if delta_rns is None:
        delta_rns = _QUAD_DELTA_RNS if method == "quad" else _GRID_DELTA_RNS
    for name, delta in (("delta_gains", delta_gains), ("delta_rns", delta_rns)):
        if method == "quad" and len(delta) != 3:
            raise ValueError(f"With method 'quad', {name} must have 3 elements [-d, 0, d].")
        if method != "quad" and len(delta) < 4:
            raise ValueError(f"With method {method!r}, {name} must have more than 3 elements.")
    # A fixed parameter is a single grid point at its initial value
    delta_gains = np.asarray(delta_gains, dtype=np.float64) if fit_gain else np.zeros(1)
    delta_rns = np.asarray(delta_rns, dtype=np.float64) if fit_rn else np.zeros(1)

    dq_flags = dq_flags_to_namedtuple(dq_flags)
    if input_dq_raw is None:
        dq_flags = None
    times = np.asarray(times, dtype=np.float64)
    Nfiles, Nreads, Nrows, Ncols = input_data.shape

    if mode == "full_detector":
        covar = fitramp_Brandt.Covar(times)
        gains_init = float(np.nanmedian(init_gain)) + delta_gains
        rns_init = float(np.nanmedian(init_rn)) + delta_rns
        gain, rn, cov = _fit_single(
            times,
            input_data,
            input_dq_raw,
            dq_flags,
            covar,
            gains_init,
            rns_init,
            method,
            countratesidentical,
        )
        return float(gain), float(rn), cov

    init_gain = np.broadcast_to(np.asarray(init_gain, dtype=np.float32), (Nrows, Ncols))
    init_rn = np.broadcast_to(np.asarray(init_rn, dtype=np.float32), (Nrows, Ncols))
    return _fit_per_pixel(
        times,
        input_data,
        input_dq_raw,
        dq_flags,
        init_gain,
        init_rn,
        delta_gains,
        delta_rns,
        method,
        countratesidentical,
        max_cores,
    )


def _fit_per_pixel(
    times,
    ramps,
    dq_raw,
    dq_flags,
    init_gain,
    init_rn,
    delta_gains,
    delta_rns,
    method,
    countratesidentical,
    max_cores,
):
    """Fit each pixel, in row blocks across processes when `max_cores` > 1."""
    Nfiles, Nreads, Nrows, Ncols = ramps.shape
    float_type = ctypes.c_float

    def shared(shape, values=None, ctype=float_type):
        raw = RawArray(ctype, int(np.prod(shape)))
        arr = shared_array_to_numpy(raw, shape, dtype=ctype)
        arr[:] = 0 if values is None else values
        return raw, arr

    gain_raw, gain = shared((Nrows, Ncols))
    rn_raw, rn = shared((Nrows, Ncols))
    cov_raw, cov = shared((2, 2, Nrows, Ncols))
    row_args = (times, delta_gains, delta_rns, method, countratesidentical, dq_flags)

    if max_cores > 1:
        ramps_raw, _ = shared(ramps.shape, ramps)
        dq_raw_raw = None
        if dq_raw is not None:
            dq_raw_raw, _ = shared(dq_raw.shape, dq_raw, ctypes.c_uint8)
        init_gain_raw, _ = shared((Nrows, Ncols), init_gain)
        init_rn_raw, _ = shared((Nrows, Ncols), init_rn)
        init_args = (
            (ramps_raw, ramps.shape),
            None if dq_raw is None else (dq_raw_raw, dq_raw.shape),
            (gain_raw, gain.shape),
            (rn_raw, rn.shape),
            (cov_raw, cov.shape),
            (init_gain_raw, (Nrows, Ncols)),
            (init_rn_raw, (Nrows, Ncols)),
        )
        block_size = max(1, Nrows // (4 * max_cores))
        ctx = mp.get_context("spawn")  # avoids unsafe fork after OpenMP init
        pool = ctx.Pool(
            processes=max_cores,
            initializer=_init_worker,
            initargs=init_args,
            maxtasksperchild=50,
        )
        try:
            with numba_thread_scope(max_cores):
                tasks = [
                    pool.apply_async(
                        _fit_rows_worker,
                        args=(start, min(start + block_size, Nrows), row_args),
                    )
                    for start in range(0, Nrows, block_size)
                ]
            for task in progressbar(tasks, desc="Fitting gain and read noise", fast=False):
                task.get()
        finally:
            pool.close()
            pool.join()
    else:
        for row_id in progressbar(range(Nrows), desc="Fitting gain and read noise", fast=False):
            gain[row_id], rn[row_id], cov[:, :, row_id] = _fit_row(
                ramps[:, :, row_id],
                None if dq_raw is None else dq_raw[:, :, row_id],
                init_gain[row_id],
                init_rn[row_id],
                *row_args,
            )

    return gain, rn, cov


_shared = {}


def _init_worker(ramps, dq_raw, gain, rn, cov, init_gain, init_rn):
    """Store the shared arrays (raw array, shape) in each worker process."""
    _shared.update(
        ramps=ramps,
        dq_raw=dq_raw,
        gain=gain,
        rn=rn,
        cov=cov,
        init_gain=init_gain,
        init_rn=init_rn,
    )


def _fit_rows_worker(row_start, row_end, row_args):
    """Fit rows [row_start, row_end) from the shared arrays and write the results back."""
    arrays = {}
    for name, value in _shared.items():
        if value is not None:
            raw, shape = value
            ctype = ctypes.c_uint8 if name == "dq_raw" else ctypes.c_float
            arrays[name] = shared_array_to_numpy(raw, shape, dtype=ctype)
    dq_raw = arrays.get("dq_raw")
    for row_id in range(row_start, row_end):
        out = _fit_row(
            arrays["ramps"][:, :, row_id],
            None if dq_raw is None else dq_raw[:, :, row_id],
            arrays["init_gain"][row_id],
            arrays["init_rn"][row_id],
            *row_args,
        )
        arrays["gain"][row_id], arrays["rn"][row_id], arrays["cov"][:, :, row_id] = out
    return row_start, row_end


def _fit_row(
    ramps_row,
    dq_raw_row,
    init_gain_row,
    init_rn_row,
    times,
    delta_gains,
    delta_rns,
    method,
    countratesidentical,
    dq_flags,
):
    """Fit every pixel of one detector row; ramps_row has shape (Nfiles, Nreads, Ncols)."""
    Ncols = ramps_row.shape[2]
    covar = fitramp_Brandt.Covar(times)
    gain = np.zeros(Ncols, dtype=np.float32)
    rn = np.zeros(Ncols, dtype=np.float32)
    cov = np.zeros((2, 2, Ncols), dtype=np.float32)
    for i in range(Ncols):
        gain[i], rn[i], cov[:, :, i] = _fit_single(
            times,
            ramps_row[:, :, i],
            None if dq_raw_row is None else dq_raw_row[:, :, i],
            dq_flags,
            covar,
            init_gain_row[i] + delta_gains,
            init_rn_row[i] + delta_rns,
            method,
            countratesidentical,
        )
    return gain, rn, cov


def _read_differences(times, ramps, dq_raw, dq_flags):
    """
    Read differences per unit time, of the ramps and of their squares, and the mask of
    differences to use. `ramps` has shape (Nfiles, Nreads, ...); extra axes are flattened with
    the files into one axis of ramps. Returns arrays of shape (Nreads - 1, Nramps).
    """
    Nreads = times.shape[0]
    ramps = np.asarray(ramps).swapaxes(0, 1).reshape(Nreads, -1)
    dt = np.diff(times)[:, None]
    diffs_lin = np.diff(ramps, axis=0).astype(np.float64) / dt
    diffs_quad = np.diff(ramps**2, axis=0).astype(np.float64) / dt
    if dq_raw is None or dq_flags is None:
        diffs2use = np.ones(diffs_lin.shape, dtype=np.uint8)
    else:
        dq_raw = np.asarray(dq_raw).swapaxes(0, 1).reshape(Nreads, -1)
        bad = (dq_raw & (dq_flags.DO_NOT_USE | dq_flags.SATURATED)) != 0
        jump = (dq_raw & dq_flags.JUMP_DET) != 0
        diffs2use = (~(bad[1:] | bad[:-1] | jump[1:])).astype(np.uint8)
    return diffs_lin, diffs_quad, diffs2use


def _fit_single(
    times,
    ramps,
    dq_raw,
    dq_flags,
    covar,
    gains_init,
    rns_init,
    method,
    countratesidentical,
):
    """Fit one gain and read noise to the ramps (Nfiles, Nreads, ...) with the given method."""
    diffs_lin, diffs_quad, diffs2use = _read_differences(times, ramps, dq_raw, dq_flags)
    args = (diffs_lin, diffs_quad, covar, diffs2use, countratesidentical)

    if method == "quad":
        return _fit_quad_iterated(*args, gains_init, rns_init)

    # Grid search over the positive grid points
    gains = gains_init[gains_init > 0]
    rns = rns_init[rns_init > 0]
    logL = fit_gain_Brandt.gen_likelihood_map(
        diffs_lin,
        diffs_quad,
        covar,
        rns,
        gains,
        [0],
        d2use=diffs2use,
        countratesidentical=countratesidentical,
    )[..., 0]
    i_rn, i_gain = np.unravel_index(np.argmax(logL), logL.shape)

    if method == "both":
        # Start the quadratic fit from the grid peak, with half the grid spacing
        gains_short = _three_points(gains, i_gain)
        rns_short = _three_points(rns, i_rn)
        return _fit_quad_iterated(*args, gains_short, rns_short)

    # method == "grid": best fit and errors from the marginal posteriors. As in hispecdrp, the
    # diagonal of cov holds the errors, not the variances.
    posterior = np.exp(logL - logL.max())
    gain, gain_err = _posterior_fit(gains, posterior.sum(axis=0))
    if rns.size > 3:
        rn, rn_err = _posterior_fit(rns, posterior.sum(axis=1))
    else:
        rn, rn_err = np.nan, np.nan
    cov = np.full((2, 2), np.nan)
    cov[0, 0] = gain_err
    cov[1, 1] = rn_err
    return gain, rn, cov


def _posterior_fit(grid, marginal):
    """Mode and larger 68% error of a marginal posterior sampled on `grid`."""
    marginal = marginal / marginal.sum()
    marginal /= np.diff(grid, prepend=grid[0] - (grid[1] - grid[0]))
    best, left_err, right_err = get_err_from_posterior(grid, marginal)
    return best, max(left_err, right_err)


def _three_points(grid, index):
    """Three points centered on grid[index], spaced by half the grid spacing."""
    if grid.size == 1:
        return grid
    delta = (grid[1] - grid[0]) / 2
    return np.array([grid[index] - delta, grid[index], grid[index] + delta])


def _fit_quad_iterated(
    diffs_lin,
    diffs_quad,
    covar,
    diffs2use,
    countratesidentical,
    gains_short,
    rns_short,
):
    """
    Fit a quadratic form to the log likelihood at 3 points per fitted parameter, then again
    at +-1 sigma around the first result.
    """
    args = (diffs_lin, diffs_quad, covar)
    gain, rn, cov = _fitquad_gain_rn(*args, rns_short, gains_short, diffs2use, countratesidentical)
    if len(gains_short) > 1:
        gain_err = np.sqrt(cov[0, 0])
        gains_short = [gain - gain_err, gain, gain + gain_err]
    if len(rns_short) > 1:
        rn_err = np.sqrt(cov[1, 1])
        rns_short = [rn - rn_err, rn, rn + rn_err]
    return _fitquad_gain_rn(*args, rns_short, gains_short, diffs2use, countratesidentical)


def _fitquad_gain_rn(diffs_lin, diffs_quad, covar, rns, gains, diffs2use, countratesidentical):
    """
    Best-fit gain and read noise and their covariance from a quadratic fit to the log
    likelihood. With one parameter fixed (a single grid point), fit a parabola in the other.
    """
    rns = np.asarray(rns, dtype=np.float64)
    gains = np.asarray(gains, dtype=np.float64)
    if len(rns) > 1 and len(gains) > 1:
        (gain, rn), cov, _ = fit_gain_Brandt.fitquad(
            diffs_lin,
            diffs_quad,
            covar,
            rns,
            gains,
            log=False,
            d2use=diffs2use,
            countratesidentical=countratesidentical,
        )
        return gain, rn, cov

    logL = fit_gain_Brandt.gen_likelihood_map(
        diffs_lin,
        diffs_quad,
        covar,
        rns,
        gains,
        [0],
        countratesidentical=countratesidentical,
        d2use=diffs2use,
    )
    cov = np.full((2, 2), np.nan)
    if len(rns) == 1:
        a, b, _ = _quad_1d_through_3pts(gains, logL[0, :, 0])
        gain, rn = -b / (2 * a), rns[0]
        cov[0, 0] = -1 / (2 * a)
    else:
        a, b, _ = _quad_1d_through_3pts(rns, logL[:, 0, 0])
        gain, rn = gains[0], -b / (2 * a)
        cov[1, 1] = -1 / (2 * a)
    return gain, rn, cov


def _quad_1d_through_3pts(x, y):
    """Coefficients (a, b, c) of the parabola y = a x^2 + b x + c through three points."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != (3,) or y.shape != (3,):
        raise ValueError("x and y must be length-3 arrays")
    if np.unique(x).size != 3:
        raise ValueError("x values must be distinct")

    x1, x2, x3 = x
    y1, y2, y3 = y
    d1 = (x1 - x2) * (x1 - x3)
    d2 = (x2 - x1) * (x2 - x3)
    d3 = (x3 - x1) * (x3 - x2)

    a = y1 / d1 + y2 / d2 + y3 / d3
    b = -(y1 * (x2 + x3) / d1 + y2 * (x1 + x3) / d2 + y3 * (x1 + x2) / d3)
    c = y1 * x2 * x3 / d1 + y2 * x1 * x3 / d2 + y3 * x1 * x2 / d3
    return a, b, c
