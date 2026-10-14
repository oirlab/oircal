import ctypes
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray

import numpy as np
from progressbar import progressbar

from ..utils.dq_utils import dq_flags_to_namedtuple
from ..utils.parallelization_utils import numba_thread_scope, shared_array_to_numpy
from . import calc_nonlin_coefs_Brandt as nonlin_Brandt


__all__ = ["fit_nonlin_coefs"]

_UINT16_MAX = np.iinfo(np.uint16).max


def fit_nonlin_coefs(
    input_data: np.ndarray,
    gain: np.ndarray,
    rn: np.ndarray,
    input_dq_raw: np.ndarray | None = None,
    dq_flags: tuple | None = None,
    order: int = 5,
    satval: np.ndarray | None = None,
    mode: str = "per_pixel",
    use_legendre: bool = True,
    readstart: int = 0,
    criterion_for_nonlin_thresh: float = 0.01,
    criterion_for_nonlin_max: float = 0.1,
    max_cores: int = 1,
):
    """
    Fit for the detector nonlinearity polynomial coefficients.

    Parameters
    ----------
    input_data
        4D array of shape (Nfiles, Nreads, Nrows, Ncols) containing the input ramps.
    gain
        2D array of shape (Nrows, Ncols) containing the gain map (e-/ADU).
    rn
        2D array of shape (Nrows, Ncols) containing the read noise map (in DN).
    input_dq_raw
        4D array of shape (Nfiles, Nreads, Nrows, Ncols) containing the input UTR data quality
        flags.
        Default is None.
    dq_flags
        Dict mapping DQ flag names to bit values, with keys DO_NOT_USE, SATURATED and JUMP_DET.
        If `input_dq_raw` or `dq_flags` is None, all read differences are used. Default is None.
    order
        Order of polynomial to fit for the nonlinearity correction. Default is 5.
    satval
        2D array of shape (Nrows, Ncols) of values beyond which reads will be ignored in the fit (in
        DN).
        Default is 64000 everywhere.
    mode
        'full_detector' to fit a single set of coefficients for the entire detector,
        'per_pixel' to fit each pixel independently. Default is 'per_pixel'.
    use_legendre
        If True, fit in the Legendre polynomial basis. Default is True.
    readstart
        First read (indexed from zero) that we use. Default is 0.
    criterion_for_nonlin_thresh
        Criterion to define the nonlinearity threshold.
        ie, if the data DN deviates from a linear ramp by more than this fraction, it is considered non-linear, but it can be corrected reasonably accurately.
        Default is 0.01.
    criterion_for_nonlin_max
        Criterion to define the nonlinearity maximum.
        ie, if the data DN deviates from a linear ramp by more than this fraction, it is considered too non-linear and cannot be corrected accurately.
        Default is 0.1.
    max_cores
        Number of cores for parallelization.
        Not used for mode == 'full_detector'. Default is 1 (no parallelization).

    Returns
    -------
    nonlin_coefs : np.ndarray
        Polynomial coefficients of P, lowest degree first, such that corrected DN =
        P(measured DN), both with the pedestal (first read) subtracted. Shape
        (order+1, Nrows, Ncols) if mode is 'per_pixel', else (order+1,).
    nonlin_thresh : np.ndarray or float
        Measured DN (pedestal subtracted) at which the correction departs from linear by
        `criterion_for_nonlin_thresh`. Shape (Nrows, Ncols) if mode is 'per_pixel', else a
        float. NaN where the criterion is not reached below `satval`.
    nonlin_max : np.ndarray or float
        Measured DN (pedestal subtracted) at which the correction departs from linear by
        `criterion_for_nonlin_max`. Shape (Nrows, Ncols) if mode is 'per_pixel', else a
        float. NaN where the criterion is not reached below `satval`.
    """
    dq_flags = dq_flags_to_namedtuple(dq_flags)
    Nfiles, Nreads, Nrows, Ncols = input_data.shape

    if input_dq_raw is None:
        input_dq_raw = np.zeros(input_data.shape, dtype=np.uint8)
        dq_flags = None

    if satval is None:
        satval = np.full((Nrows, Ncols), 64000)

    if mode.lower() == "full_detector":
        row_id = 0
        ramp_row = np.array(input_data).swapaxes(1, 3)  # Nfiles, Nrows, Ncols, Nreads
        dq_raw_row = np.array(input_dq_raw).swapaxes(1, 3)  # Nfiles, Nrows, Ncols, Nreads
        ramp_row = ramp_row.reshape(Nfiles * Nrows * Ncols, Nreads)[
            :,
            :,
            np.newaxis,
        ]  # shape (Nfiles*Nrows*Ncols,Nreads,1)
        dq_raw_row = dq_raw_row.reshape(Nfiles * Nrows * Ncols, Nreads)[
            :,
            :,
            np.newaxis,
        ]  # shape (Nfiles*Nrows*Ncols,Nreads,1)
        gains_row = np.array([np.nanmedian(gain)])
        rns_row = np.array([np.nanmedian(rn)])
        satval_row = np.array([np.nanmedian(satval)])

        args = (
            row_id,
            ramp_row,
            dq_raw_row,
            gains_row,
            rns_row,
            satval_row,
            order,
            use_legendre,
            readstart,
            criterion_for_nonlin_thresh,
            criterion_for_nonlin_max,
            dq_flags,
        )

        _out = _fit_nonlin_coefs_row(args)

        _, nonlin_coeffs, nonlin_thresh, nonlin_max = _out
        nonlin_coeffs = nonlin_coeffs[:, 0]
        nonlin_thresh = float(nonlin_thresh[0])
        nonlin_max = float(nonlin_max[0])

    elif mode.lower() == "per_pixel":
        Nfiles, Nreads, Nrows, Ncols = input_data.shape

        _out = _fit_nonlin_coefs_per_pixel(
            input_data,
            input_dq_raw,
            gain,
            rn,
            dq_flags=dq_flags,
            order=order,
            satval=satval,
            use_legendre=use_legendre,
            readstart=readstart,
            criterion_for_nonlin_thresh=criterion_for_nonlin_thresh,
            criterion_for_nonlin_max=criterion_for_nonlin_max,
            max_cores=max_cores,
        )
        nonlin_coeffs, nonlin_thresh, nonlin_max = _out

    return nonlin_coeffs, nonlin_thresh, nonlin_max


def _fit_nonlin_coefs_per_pixel(
    ramps: np.ndarray,
    dq_raw: np.ndarray,
    gain: np.ndarray,
    rn: np.ndarray,
    dq_flags: tuple | None = None,
    order: int = 5,
    satval: np.ndarray | None = None,
    use_legendre: bool = True,
    readstart: int = 0,
    criterion_for_nonlin_thresh: float = 0.01,
    criterion_for_nonlin_max: float = 0.1,
    max_cores: int = 1,
):
    """
    Fit for the nonlinearity polynomial coefficients on a per pixel basis.

    Parameters
    ----------
    ramps
        4D array of shape (Nfiles, Nreads, Nrows, Ncols) containing the input ramps.
    dq_raw
        4D array of shape (Nfiles, Nreads, Nrows, Ncols) containing the input UTR data quality
        flags.
    gain
        2D array of shape (Nrows, Ncols) containing the gain map (e-/ADU).
    rn
        2D array of shape (Nrows, Ncols) containing the read noise map (in DN).
    dq_flags
        Named tuple of DQ bit values with integer fields DO_NOT_USE, SATURATED and JUMP_DET.
        If None, dq_raw is ignored and all read differences are used. Default is None.
    order
        Order of polynomial to fit for the nonlinearity correction. Default is 5.
    satval
        2D array of shape (Nrows, Ncols) of values beyond which reads will be ignored in the fit (in
        DN).
        Default is 64000 everywhere.
    use_legendre
        If True, fit in the Legendre polynomial basis. Default is True.
    readstart
        First read (indexed from zero) that we use. Default is 0.
    criterion_for_nonlin_thresh
        Criterion to define the nonlinearity threshold.
        ie, if the data DN deviates from a linear ramp by more than this fraction, it is considered non-linear, but it can be corrected reasonably accurately.
        Default is 0.01.
    criterion_for_nonlin_max
        Criterion to define the nonlinearity maximum.
        ie, if the data DN deviates from a linear ramp by more than this fraction, it is considered too non-linear and cannot be corrected accurately.
        Default is 0.1.
    max_cores
        Number of cores for parallelization. Default is 1 (no parallelization).

    Returns
    -------
    nonlin_coefs : np.ndarray
        Nonlinearity coefficients. Size (order+1, Nrows, Ncols).
    nonlin_thresh : np.ndarray
        Nonlinearity threshold map (Nrows,Ncols) (in DN).
    nonlin_max : np.ndarray
        Nonlinearity maximum map (Nrows,Ncols) (in DN).
    """
    Nfiles, Nreads, Nrows, Ncols = ramps.shape

    mp_data_type = ctypes.c_float
    mp_float_type = ctypes.c_float
    mp_dq_raw_type = ctypes.c_uint8
    mp_dq_type = ctypes.c_uint
    types_tuple = (mp_data_type, mp_float_type, mp_dq_raw_type, mp_dq_type)

    nonlin_coeffs_mp = RawArray(mp_float_type, (order + 1) * Ncols * Nrows)
    nonlin_coeffs_shape = ((order + 1), Nrows, Ncols)
    nonlin_coeffs_np = shared_array_to_numpy(
        nonlin_coeffs_mp,
        nonlin_coeffs_shape,
        dtype=mp_float_type,
    )
    nonlin_coeffs_np[:] = np.zeros(nonlin_coeffs_shape, dtype=np.float32)

    nonlin_thresh_mp = RawArray(mp_float_type, Ncols * Nrows)
    nonlin_thresh_shape = (Nrows, Ncols)
    nonlin_thresh_np = shared_array_to_numpy(
        nonlin_thresh_mp,
        nonlin_thresh_shape,
        dtype=mp_float_type,
    )
    nonlin_thresh_np[:] = np.zeros(nonlin_thresh_shape, dtype=np.float32)

    nonlin_max_mp = RawArray(mp_float_type, Ncols * Nrows)
    nonlin_max_shape = (Nrows, Ncols)
    nonlin_max_np = shared_array_to_numpy(nonlin_max_mp, nonlin_max_shape, dtype=mp_float_type)
    nonlin_max_np[:] = np.zeros(nonlin_max_shape, dtype=np.float32)

    gains_mp = RawArray(mp_float_type, Ncols * Nrows)
    gains_shape = (Nrows, Ncols)
    gains_np = shared_array_to_numpy(gains_mp, gains_shape, dtype=mp_float_type)
    gains_np[:] = gain.astype(np.float32)

    rns_mp = RawArray(mp_float_type, Ncols * Nrows)
    rns_shape = (Nrows, Ncols)
    rns_np = shared_array_to_numpy(rns_mp, rns_shape, dtype=mp_float_type)
    rns_np[:] = rn.astype(np.float32)

    satval_mp = RawArray(mp_float_type, Ncols * Nrows)
    satval_shape = (Nrows, Ncols)
    satval_np = shared_array_to_numpy(satval_mp, satval_shape, dtype=mp_float_type)
    satval_np[:] = satval.astype(np.float32)

    if max_cores > 1:
        ramps_mp = RawArray(mp_data_type, Ncols * Nrows * Nreads * Nfiles)
        ramps_shape = (Nfiles, Nreads, Nrows, Ncols)
        ramps_np = shared_array_to_numpy(ramps_mp, ramps_shape, dtype=mp_data_type)
        ramps_np[:] = ramps  # copy original array into shared array
        ramps = ramps_np  # free the original array from memory to avoid duplicates

        dq_raw_mp = RawArray(mp_dq_raw_type, Ncols * Nrows * Nreads * Nfiles)
        dq_raw_shape = (Nfiles, Nreads, Nrows, Ncols)
        dq_raw_np = shared_array_to_numpy(dq_raw_mp, dq_raw_shape, dtype=mp_dq_raw_type)
        dq_raw_np[:] = dq_raw  # copy original array into shared array
        dq_raw = dq_raw_np  # free the original array from memory to avoid duplicates

        # Prepare blocks
        block_size = max(1, Nrows // (4 * max_cores)) if max_cores > 1 else Nrows

        _init_args = (
            ramps_mp,
            ramps_shape,
            dq_raw_mp,
            dq_raw_shape,
            nonlin_coeffs_mp,
            nonlin_coeffs_shape,
            nonlin_thresh_mp,
            nonlin_thresh_shape,
            nonlin_max_mp,
            nonlin_max_shape,
            gains_mp,
            gains_shape,
            rns_mp,
            rns_shape,
            satval_mp,
            satval_shape,
        )

        ctx = mp.get_context("spawn")  # avoids unsafe fork after OpenMP init
        tpool = ctx.Pool(
            processes=max_cores,
            initializer=_tpool_init_fit_nonlin_coefs,
            initargs=_init_args,
            maxtasksperchild=50,
        )

        args_list = []
        for block_start in range(0, Nrows, block_size):
            block_end = min(block_start + block_size, Nrows)

            args = (
                block_start,
                block_end,
                order,
                use_legendre,
                readstart,
                criterion_for_nonlin_thresh,
                criterion_for_nonlin_max,
                dq_flags,
                types_tuple,
            )
            args_list.append(args)

        try:
            with numba_thread_scope(max_cores):
                tasks = [
                    tpool.apply_async(_fit_nonlin_coefs_worker, args=(args_tuple,))
                    for args_tuple in args_list
                ]

            for t in progressbar(tasks, desc="Fitting Nonlinear coefficients", fast=False):
                t.wait()
        finally:
            tpool.close()
            tpool.join()
    else:
        # If not parallelizing, ie max_cores=1, just run directly without shared memory
        results = []

        for row_id in progressbar(
            np.arange(Nrows),
            desc="Fitting Nonlinear coefficients",
            fast=False,
        ):
            args = (
                row_id,
                ramps[:, :, row_id, :],
                dq_raw[:, :, row_id, :],
                gains_np[row_id, :],
                rns_np[row_id, :],
                satval_np[row_id, :],
                order,
                use_legendre,
                readstart,
                criterion_for_nonlin_thresh,
                criterion_for_nonlin_max,
                dq_flags,
            )

            results.append(_fit_nonlin_coefs_row(args))

        for row_element in results:
            row_id, nonlin_coeffs, nonlin_thresh, nonlin_max = row_element

            nonlin_coeffs_np[:, row_id, :] = nonlin_coeffs
            nonlin_thresh_np[row_id, :] = nonlin_thresh
            nonlin_max_np[row_id, :] = nonlin_max

    return nonlin_coeffs_np, nonlin_thresh_np, nonlin_max_np


def _tpool_init_fit_nonlin_coefs(
    data: ctypes.Array,
    data_shape: tuple,
    dq_raw: ctypes.Array,
    dq_raw_shape: tuple,
    nonlin_coeffs: ctypes.Array,
    nonlin_coeffs_shape: tuple,
    nonlin_thresh: ctypes.Array,
    nonlin_thresh_shape: tuple,
    nonlin_max: ctypes.Array,
    nonlin_max_shape: tuple,
    gains: ctypes.Array,
    gains_shape: tuple,
    rns: ctypes.Array,
    rns_shape: tuple,
    satval: ctypes.Array,
    satval_shape: tuple,
):
    """
    Initialize shared global variables for the multiprocessing pool.
    """

    global \
        shared_data, \
        shared_data_shape, \
        shared_dq_raw, \
        shared_dq_raw_shape, \
        shared_nonlin_coeffs, \
        shared_nonlin_coeffs_shape, \
        shared_nonlin_thresh, \
        shared_nonlin_thresh_shape, \
        shared_nonlin_max, \
        shared_nonlin_max_shape, \
        shared_gains, \
        shared_gains_shape, \
        shared_rns, \
        shared_rns_shape, \
        shared_satval, \
        shared_satval_shape

    shared_data = data
    shared_data_shape = data_shape

    shared_dq_raw = dq_raw
    shared_dq_raw_shape = dq_raw_shape

    shared_nonlin_coeffs = nonlin_coeffs
    shared_nonlin_coeffs_shape = nonlin_coeffs_shape

    shared_nonlin_thresh = nonlin_thresh
    shared_nonlin_thresh_shape = nonlin_thresh_shape

    shared_nonlin_max = nonlin_max
    shared_nonlin_max_shape = nonlin_max_shape

    shared_gains = gains
    shared_gains_shape = gains_shape

    shared_rns = rns
    shared_rns_shape = rns_shape

    shared_satval = satval
    shared_satval_shape = satval_shape


def _fit_nonlin_coefs_row(indata: tuple) -> tuple:
    """
    Helper function for parallelization fitting the gain and readnoise per detector row.
    """
    (
        row_id,
        ramp_row,
        dq_raw_row,
        gains_row,
        rns_row,
        satval_row,
        order,
        use_legendre,
        readstart,
        criterion_for_nonlin_thresh,
        criterion_for_nonlin_max,
        dq_flags,
    ) = indata
    Nfiles, Nreads, Ncols = ramp_row.shape

    # ramp_row should be (Nfiles, Nreads, Ncols)
    _ramps = ramp_row[:, readstart:, :]
    pedestal = np.nanmedian(_ramps[:, 0, :], axis=0)

    Nfiles, Nreads, n_ramps = dq_raw_row.shape
    if dq_flags is None:
        diffs2use = np.ones((Nfiles, Nreads - 1, n_ramps), dtype=np.uint8)
    else:
        mask = ((dq_raw_row & dq_flags.DO_NOT_USE) | (dq_raw_row & dq_flags.SATURATED)).astype(
            np.bool,
        )
        mask2 = (dq_raw_row & dq_flags.JUMP_DET).astype(np.bool)
        diffs2use = ~(mask[:, 1:Nreads, :] | mask[:, 0 : Nreads - 1, :] | mask2[:, 1:Nreads, :])
        diffs2use = diffs2use.astype(np.uint8)

    res = nonlin_Brandt.get_nonlin_coefs(
        _ramps,
        rns_row,
        gains_row,
        order,
        pedestal,
        satval=satval_row,
        d2u_list=diffs2use,
        use_legendre=use_legendre,
    )

    res = nonlin_Brandt.get_nonlin_coefs(
        _ramps,
        rns_row,
        gains_row,
        order,
        pedestal,
        satval=satval_row,
        d2u_list=diffs2use,
        use_legendre=use_legendre,
        last_pars=res,
    )

    nonlin_coefs = res.nonlin_coefs.T  # N_coeffs x Ncols
    nonlin_coefs = np.insert(nonlin_coefs, 0, np.zeros((Ncols,)), axis=0)  # add zero-th order term

    # figure out nonlin_thresh,nonlin_max
    nonlin_thresh = _find_first_valid_root_batch(
        nonlin_coefs[::-1, :],
        criterion_for_nonlin_thresh,
        satval=satval_row,
    )
    nonlin_max = _find_first_valid_root_batch(
        nonlin_coefs[::-1, :],
        criterion_for_nonlin_max,
        satval=satval_row,
    )

    return (row_id, nonlin_coefs, nonlin_thresh, nonlin_max)


def _fit_nonlin_coefs_worker(args: tuple) -> tuple:
    """
    Worker function for multiprocessing. Runs _fit_nonlin_coefs_row on a block of rows using shared arrays.

    Parameters
    ----------
    args
        Tuple containing: (block_start, block_end, order,use_legendre,readstart,criterion_for_nonlin_thresh,criterion_for_nonlin_max, dq_flags, types_tuple)

    Returns
    -------
    block_start : int
        Starting index of the block of rows processed.
    block_end : int
        Ending index of the block of rows processed.
    """
    (
        block_start,
        block_end,
        order,
        use_legendre,
        readstart,
        criterion_for_nonlin_thresh,
        criterion_for_nonlin_max,
        dq_flags,
        types_tuple,
    ) = args
    mp_data_type, mp_float_type, mp_dq_raw_type, mp_dq_type = types_tuple

    ramps_np = shared_array_to_numpy(shared_data, shared_data_shape, dtype=mp_data_type)
    dq_raw_np = shared_array_to_numpy(shared_dq_raw, shared_dq_raw_shape, dtype=mp_dq_raw_type)

    nonlin_coeffs_np = shared_array_to_numpy(
        shared_nonlin_coeffs,
        shared_nonlin_coeffs_shape,
        dtype=mp_float_type,
    )
    nonlin_thresh_np = shared_array_to_numpy(
        shared_nonlin_thresh,
        shared_nonlin_thresh_shape,
        dtype=mp_float_type,
    )
    nonlin_max_np = shared_array_to_numpy(
        shared_nonlin_max,
        shared_nonlin_max_shape,
        dtype=mp_float_type,
    )

    gains_np = shared_array_to_numpy(shared_gains, shared_gains_shape, dtype=mp_float_type)
    rns_np = shared_array_to_numpy(shared_rns, shared_rns_shape, dtype=mp_float_type)
    satval_np = shared_array_to_numpy(shared_satval, shared_satval_shape, dtype=mp_float_type)

    results = []
    for row_id in range(block_start, block_end):
        args_row = (
            row_id,
            ramps_np[:, :, row_id, :],
            dq_raw_np[:, :, row_id, :],
            gains_np[row_id, :],
            rns_np[row_id, :],
            satval_np[row_id, :],
            order,
            use_legendre,
            readstart,
            criterion_for_nonlin_thresh,
            criterion_for_nonlin_max,
            dq_flags,
        )
        results.append(_fit_nonlin_coefs_row(args_row))

    for row_element in results:
        row_id, nonlin_coeffs, nonlin_thresh, nonlin_max = row_element

        nonlin_coeffs_np[:, row_id, :] = nonlin_coeffs
        nonlin_thresh_np[row_id, :] = nonlin_thresh
        nonlin_max_np[row_id, :] = nonlin_max

    return (block_start, block_end)


def _find_first_valid_root_batch(
    nonlin_correction_coefs: np.ndarray,
    my_threshold: float = 0.01,
    satval: float | np.ndarray = _UINT16_MAX,
    min_root: float | np.ndarray = 1,
    atol: float = 1e-10,
):
    """
    Batch-solve for the smallest admissible real root x, independently for each column ix,
    where the polynomial P_ix(x) is defined by coefficients in ``nonlin_correction_coefs[:, ix]``
    (NumPy / ``np.polyval`` convention: highest degree first), and x satisfies:

        (P_ix(x) - x) / x = my_threshold

    This is equivalent to solving, for each ix:

        P_ix(x) - (1 + my_threshold) * x = 0

    and selecting the smallest real root in (min_root[ix], satval[ix]].

    Parameters
    ----------
    nonlin_correction_coefs
        2D array of shape (n_coeffs, Ncols). Column ix contains the polynomial coefficients
        for P_ix(x), ordered from highest degree to constant term.
    satval
        Upper bound(s) on acceptable roots. Either a scalar or a 1D array of shape (Ncols,).
        Default is max uint16.
    min_root
        Strict lower bound(s) on acceptable roots. Either a scalar or a 1D array of shape (Ncols,).
        Default is 1.
    my_threshold
        Threshold T in the equation (P(x) - x)/x = T. Default is 0.01.
    atol
        Absolute tolerance for deciding whether a root is (numerically) real. Default is 1e-10.

    Returns
    -------
    root0 : ndarray
        1D float array of shape (Ncols,). Each element is the smallest admissible real root
        for that column, or NaN if no admissible root exists (or if the polynomial is invalid).

    Notes
    -----
    - Uses ``np.roots`` per column; there is no true vectorized root-finding in NumPy.
    - If the polynomial degree is < 1 (i.e., no linear term), that column returns NaN.
    """
    coefs = np.asarray(nonlin_correction_coefs, dtype=float)
    if coefs.ndim != 2:
        raise ValueError("nonlin_correction_coefs must be 2D with shape (n_coeffs, Ncols).")

    n_coeffs, Ncols = coefs.shape
    if n_coeffs < 2:
        # No linear term exists, cannot subtract (1+T)*x robustly
        return np.full(Ncols, np.nan, dtype=float)

    # Broadcast satval/min_root to per-column arrays
    sat_arr = np.asarray(satval, dtype=float)
    min_arr = np.asarray(min_root, dtype=float)

    if sat_arr.ndim == 0:
        sat_arr = np.full(Ncols, float(sat_arr), dtype=float)
    if min_arr.ndim == 0:
        min_arr = np.full(Ncols, float(min_arr), dtype=float)

    if sat_arr.shape != (Ncols,):
        raise ValueError(f"satval must be a scalar or shape (Ncols,), got {sat_arr.shape}.")
    if min_arr.shape != (Ncols,):
        raise ValueError(f"min_root must be a scalar or shape (Ncols,), got {min_arr.shape}.")

    out = np.full(Ncols, np.nan, dtype=float)

    for ix in range(Ncols):
        # Build Q(x) = P(x) - (1+T)*x
        tmp = coefs[:, ix].copy()
        tmp[-2] -= 1.0 + my_threshold

        # If leading coefficients are zero, np.roots can behave poorly; trim them
        # (while preserving at least degree-1 if possible).
        nz = np.flatnonzero(tmp)
        if nz.size == 0:
            continue
        first_nz = nz[0]
        tmp_trim = tmp[first_nz:]

        # Need at least a linear polynomial after trimming
        if tmp_trim.size < 2:
            continue

        roots = np.roots(tmp_trim)

        # Keep (numerically) real roots
        real_roots = roots[np.isclose(roots.imag, 0.0, atol=atol)].real
        if real_roots.size == 0:
            continue

        # Enforce bounds: (min_root, satval]
        valid = real_roots[(real_roots > min_arr[ix]) & (real_roots <= sat_arr[ix])]
        if valid.size == 0:
            continue

        out[ix] = np.min(valid)

    return out
