from contextlib import contextmanager
import ctypes
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
from typing import Any

import numpy as np
from progressbar import progressbar
from stcal.ramp_fitting import ramp_fit
from stcal.ramp_fitting.ramp_fit_class import RampData

from ..jump_detection.jump_detection_jwst import _jump_detection_jwst_row
from ..utils.dq_utils import dq_flags_to_namedtuple
from ..utils.parallelization_utils import numba_thread_scope, shared_array_to_numpy
from . import fitramp_Brandt
from . import fitramp_cython_Brandt as fitramp_cython
from .midpoint import calculate_midpoint_time_deltas


__all__ = ["fit_ramps_jwst"]

# DQ bit values used when `dq_flags` is not given (JWST convention).
_DEFAULT_DQ_FLAGS = {
    "GOOD": 0,
    "DO_NOT_USE": 2**0,
    "SATURATED": 2**1,
    "JUMP_DET": 2**2,
    "OUTLIER": 2**4,
    "UNRELIABLE_SLOPE": 2**24,
    "UNRELIABLE_FLAT": 2**25,
    "REFERENCE_PIXEL": 2**31,
}


@contextmanager
def _stcal_likely_fit_allow_negative_rates():
    """
    Allow fitted / interim count rates to be negative in ``stcal`` likely_fit.

    Photon-noise covariance still uses ``max(rate, 0)`` (variance cannot be
    negative). The previous ``rate *= (rate > 0)`` pattern zeroed low-SNR /
    negative guesses and made empty background look like a hard floor at 0
    instead of a Gaussian around the true rate.
    """
    import stcal.ramp_fitting.likely_fit as lf

    orig_init = lf.initial_count_rate_guess
    orig_alphas = lf.compute_alphas_betas
    orig_fit_ramps = lf.fit_ramps

    def initial_count_rate_guess(diffs: np.ndarray, diffs2use: np.ndarray):
        num = np.sum((diffs * diffs2use), axis=0)
        den = np.sum(diffs2use, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return num / den

    def compute_alphas_betas(
        count_rate_guess: np.ndarray,
        gain: np.ndarray,
        rnoise: np.ndarray,
        covar: lf.Covar,
        rescale: bool,
        diffs: np.ndarray,
        dn_scale: float,
    ):
        guess = np.maximum(np.asarray(count_rate_guess, dtype=np.float64), 0.0)
        return orig_alphas(guess, gain, rnoise, covar, rescale, diffs, dn_scale)

    def fit_ramps(*args: Any, count_rate_guess: np.ndarray | None = None, **kwargs: Any):
        # Callers (jump search / final fit) may pass clipped guesses; restore a
        # non-negative *covariance* rate without forcing the solved countrate ≥ 0.
        if count_rate_guess is not None:
            count_rate_guess = np.maximum(np.asarray(count_rate_guess, dtype=np.float64), 0.0)
        return orig_fit_ramps(*args, count_rate_guess=count_rate_guess, **kwargs)

    lf.initial_count_rate_guess = initial_count_rate_guess
    lf.compute_alphas_betas = compute_alphas_betas
    lf.fit_ramps = fit_ramps
    try:
        yield
    finally:
        lf.initial_count_rate_guess = orig_init
        lf.compute_alphas_betas = orig_alphas
        lf.fit_ramps = orig_fit_ramps


def _fit_ramps_cython_row(indata: tuple) -> tuple:
    """
    Passes data for each row of a ramp to Tim Brandt's likelihood-based ramp fitting algorithm in Cython.

    Parameters
    ----------
    indata
        Tuple containing the following elements:
            ramp row index (int)
            read times (np.ndarray of shape (Nreads,))
            ramp data (np.ndarray of shape (1, Nreads, 1, Ncols))
            readnoise (np.ndarray of shape (1, Ncols))
            gain (np.ndarray of shape (1, Ncols))
            DQ cube (np.ndarray of shape (1, Nreads, 1, Ncols))
            DQ image (np.ndarray of shape (1, Ncols))
            dark current (np.ndarray of shape (1, Ncols))
            algorithm (str)
            DQ flags (namedtuple)

    Returns
    -------
    i : int
        Index of input row from the ramps, same as input.
    slope : np.ndarray
        Slope image (1, Ncols).
    dq_out : np.ndarray
        Output data quality image with flags from algorithm (1, Ncols).
    var_poisson : np.ndarray
        Output variance from poisson noise (1, Ncols).
    var_rnoise : np.ndarray
        Output variance from read noise (1, Ncols).
    err : np.ndarray
        Output error in slope image (1, Ncols).
    """

    (
        i,
        readtimes,
        data_3d,
        readnoise_2d,
        gain_2d,
        groupdq,
        pixeldq_2d,
        dark_current_2d,
        algorithm,
        dq_flags,
    ) = indata

    # We are implicitly fitting by row, so columns is only other axis and nrows=1
    nints, Nreads, Nrows, Ncols = data_3d.shape

    _ramps = data_3d[0, :, 0, :]
    _dq_raw = groupdq[0, :, 0, :]

    diff_times = np.diff(readtimes)[:, None]
    diffs = np.diff(_ramps, axis=0).astype(np.float64) / diff_times
    n0, n1 = diffs.shape

    Nreads, n_ramps = _dq_raw.shape
    mask = ((_dq_raw & dq_flags.DO_NOT_USE) | (_dq_raw & dq_flags.SATURATED)).astype(np.bool)
    mask2 = (_dq_raw & dq_flags.JUMP_DET).astype(np.bool)
    diffs2use = ~(mask[1:Nreads] | mask[0 : Nreads - 1] | mask2[1:Nreads])
    diffs2use = diffs2use.astype(np.uint8)

    # todo: try to put covar in shared memory?
    covar = fitramp_Brandt.Covar(readtimes.astype(np.float64))

    # Mean difference as count-rate guess. Do NOT floor negatives to 0 — empty /
    # low-SNR pixels must keep a signed guess so the solved rate is Gaussian
    # about the true background (including negative rates). Photon noise in the
    # Cython kernel uses max(cguess, 0) via the covariance construction below.
    with np.errstate(invalid="ignore", divide="ignore"):
        cguess = np.sum(diffs * diffs2use, axis=0) / np.sum(diffs2use, axis=0)
    cguess_phnoise = np.maximum(cguess, 0.0)

    cython_out = fitramp_cython.fit_ramps_row(
        diffs,
        diffs2use,
        covar.alpha_phnoise,
        covar.beta_phnoise,
        covar.alpha_readnoise,
        covar.beta_readnoise,
        readnoise_2d[0, :].astype(np.float64),
        gain_2d[0, :].astype(np.float64),
        cguess_phnoise,
        n1,
        n0,
    )

    if len(cython_out) >= 9:
        (
            logdetC_np,
            A_np,
            B_np,
            C_np,
            countrate_np,
            chisq_np,
            uncert_np,
            var_poisson_np,
            var_rnoise_np,
        ) = cython_out[:9]
        var_poisson = np.asarray(var_poisson_np, dtype=np.float64)
        var_rnoise = np.asarray(var_rnoise_np, dtype=np.float64)
        uncert = np.asarray(uncert_np, dtype=np.float64)
        # Prefer the GLS total when the split is incomplete / non-finite.
        total = np.square(uncert)
        split = var_poisson + var_rnoise
        bad = ~(np.isfinite(split) & (split >= 0.0)) | (split <= 0.0)
        # If split is usable, rebuild err from the split (stcal-like); else keep uncert.
        err = np.where(bad, uncert, np.sqrt(np.maximum(split, 0.0)))
        var_poisson = np.where(bad, 0.0, var_poisson)
        var_rnoise = np.where(bad, total, var_rnoise)
    else:
        logdetC_np, A_np, B_np, C_np, countrate_np, chisq_np, uncert_np = cython_out[:7]
        # Legacy cython: total uncertainty only — park it in var_rnoise.
        uncert = np.asarray(uncert_np, dtype=np.float64)
        var_rnoise = np.square(uncert)
        var_poisson = np.zeros_like(var_rnoise, dtype=np.float64)
        err = uncert

    dq = np.zeros((1, Ncols), dtype=np.uint32)

    return (i, countrate_np, dq, var_poisson, var_rnoise, err)


def _fit_ramps_jwst_row(indata: tuple) -> tuple:
    """
    Passes data for each row of a ramp to stcal.ramp_fitting.likely_fit algorithms.

    Parameters
    ----------
    indata
        Tuple containing the following elements:
            ramp row index (int)
            read times (np.ndarray of shape (Nreads,))
            ramp data (np.ndarray of shape (1, Nreads, 1, Ncols))
            readnoise (np.ndarray of shape (1, Ncols))
            gain (np.ndarray of shape (1, Ncols))
            DQ cube (np.ndarray of shape (1, Nreads, 1, Ncols))
            DQ image (np.ndarray of shape (1, Ncols))
            dark current (np.ndarray of shape (1, Ncols))
            algorithm (str)
            DQ flags (namedtuple)

    Returns
    -------
    i : int
        Index of input row from the ramps, same as input.
    slope : np.ndarray
        Slope image (1, Ncols).
    dq_out : np.ndarray
        Output data quality image with flags from algorithm (1, Ncols).
    var_poisson : np.ndarray
        Output variance from poisson noise (1, Ncols).
    var_rnoise : np.ndarray
        Output variance from read noise (1, Ncols).
    err : np.ndarray
        Output error in slope image (1, Ncols).
    """

    (
        i,
        times,
        data_3d,
        readnoise_2d,
        gain_2d,
        groupdq,
        pixeldq_2d,
        dark_current_2d,
        algorithm,
        dq_flags,
    ) = indata

    # We are implicitly fitting by row, so columns is only other axis and nrows=1
    nints, nframes, Nrows, Ncols = data_3d.shape

    # Set static parameters for stcal.ramp_fitting algorithms
    save_opt = False  # Save optional output
    weighting = "optimal"
    suppress_one_group = False  # Suppress saturated ramps with good 0th group

    # Initialize ramp data model
    ramp_data = RampData()

    # Explicitly set flags required by JWST pipeline
    ramp_data.flags_do_not_use = dq_flags.DO_NOT_USE
    ramp_data.flags_jump_det = dq_flags.JUMP_DET
    ramp_data.flags_saturated = dq_flags.SATURATED
    ramp_data.flags_no_gain_val = dq_flags.UNRELIABLE_FLAT  # Check if this is best/optimal flag
    ramp_data.flags_unreliable_slope = dq_flags.UNRELIABLE_SLOPE
    ramp_data.flags_chargeloss = (
        dq_flags.OUTLIER
    )  # Check if this is best/optimal flag - only required by FIXSEN_C

    # Explicitly set data arrays
    ramp_data.data = data_3d
    ramp_data.groupdq = groupdq
    ramp_data.pixeldq = pixeldq_2d
    ramp_data.average_dark_current = dark_current_2d

    # Get average exposure time
    diffs = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    group_time = np.average(diffs)

    # Set other required parameters
    ramp_data.algorithm = algorithm
    ramp_data.start_row = 0
    ramp_data.num_rows = Nrows
    ramp_data.nframes = 1  # this corresponds to nints instead of nframes as above

    # Required by FIXSEN_C for frame time
    ramp_data.frame_time = 0.0
    ramp_data.group_time = group_time
    ramp_data.groupgap = 0
    ramp_data.drop_frames1 = 0

    ramp_data.suppress_one_group_ramps = suppress_one_group

    # Times information is set as read_pattern parameter of RampData
    ramp_data.read_pattern = list(times)

    # todo: we are disabling the jump detection for now, because we cannot get the group DQ flags to return from the function.
    # and we might need it to estimate the mid_time correctly.
    ramp_data.rejection_threshold = np.inf

    # Run ramp fitting
    if algorithm == "LIKELY":
        if nframes < ramp_fit.likely_fit.LIKELY_MIN_NGROUPS:
            raise Exception(
                f"Input ramps have {nframes} frames which is less than LIKELY algorithm requirement of {ramp_fit.likely_fit.LIKELY_MIN_NGROUPS} frames",
            )
        # stcal.likely_fit divides readnoise by √2 because JWST READNOISE
        # reference files store CDS (pairwise-difference) noise
        # (σ_CDS = √2 σ_single). HISPEC RN calibrations and create_ramp use
        # single-read σ, so convert to CDS units before calling stcal; otherwise
        # rate_var_rnoise is under-reported by exactly 2×.
        readnoise_for_stcal = np.asarray(readnoise_2d, dtype=np.float64) * np.sqrt(2.0)
        # Call directly to likely ramp fit code (allow signed / low-SNR rates)
        with _stcal_likely_fit_allow_negative_rates():
            image_info, integ_info, opt_info = ramp_fit.likely_fit.likely_ramp_fit(
                ramp_data,
                readnoise_for_stcal,
                gain_2d,
            )
    elif algorithm == "FIXSEN_C":
        # stcal's OLS fitter takes the single-read read noise in electrons (stcal's ramp_fit
        # converts JWST's CDS read noise in DN with `readnoise_2d *= gain_2d / sqrt(2)` before
        # calling it). Our read noise is already single-read, so only convert DN to electrons.
        readnoise_e = (readnoise_2d * gain_2d).astype(readnoise_2d.dtype)
        image_info, _, _ = ramp_fit.ols_fit.ols_slope_fitter(
            ramp_data,
            gain_2d,
            readnoise_e,
            weighting,
            save_opt,
        )

    # Parse output image info that we need to pass to output data model
    outdata, dq, var_poisson, var_rnoise, err = image_info

    return (i, outdata, dq, var_poisson, var_rnoise, err)


def _fit_ramps_worker(args: tuple) -> tuple:
    """
    Worker function for multiprocessing. Runs _fit_ramps_jwst_row on a block of rows using shared arrays.

    Parameters
    ----------
    args
        (block_start, block_end, times, algorithm, dq_flags, types_tuple)

    Returns
    -------
    block_start : int
        Starting index of the block of rows processed.
    block_end : int
        Ending index of the block of rows processed.
    results : list
        List of results from processing each row in the block.
    """
    block_start, block_end, times, algorithm, dq_flags, types_tuple = args
    mp_data_type, mp_float_type, mp_dq_raw_type, mp_dq_type = types_tuple

    ramps_np = shared_array_to_numpy(shared_data, shared_data_shape, dtype=mp_data_type)
    dq_raw_np = shared_array_to_numpy(shared_dq_raw, shared_dq_raw_shape, dtype=mp_dq_raw_type)
    dq_image_np = shared_array_to_numpy(shared_dq_image, shared_dq_image_shape, dtype=mp_dq_type)
    gain_np = shared_array_to_numpy(shared_gain, shared_gain_shape, dtype=mp_float_type)
    dark_np = shared_array_to_numpy(shared_dark, shared_dark_shape, dtype=mp_float_type)
    rdnoise_np = shared_array_to_numpy(
        shared_read_noise,
        shared_read_noise_shape,
        dtype=mp_float_type,
    )
    start_times_np = shared_array_to_numpy(
        shared_start_times,
        shared_start_times_shape,
        dtype=mp_float_type,
    )

    slope_image_np = shared_array_to_numpy(
        shared_slope_image,
        shared_slope_image_shape,
        dtype=mp_float_type,
    )
    slope_errors_np = shared_array_to_numpy(
        shared_slope_errors,
        shared_slope_errors_shape,
        dtype=mp_float_type,
    )
    var_poisson_image_np = shared_array_to_numpy(
        shared_var_poisson_image,
        shared_var_poisson_image_shape,
        dtype=mp_float_type,
    )
    var_rnoise_image_np = shared_array_to_numpy(
        shared_var_rnoise_image,
        shared_var_rnoise_image_shape,
        dtype=mp_float_type,
    )
    mid_times_np = shared_array_to_numpy(
        shared_mid_times,
        shared_mid_times_shape,
        dtype=mp_float_type,
    )

    results = []
    for row in range(block_start, block_end):
        args_row = (
            row,
            times,
            ramps_np[np.newaxis, :, [row], :],
            rdnoise_np[[row], :],
            gain_np[[row], :],
            dq_raw_np[np.newaxis, :, [row], :],
            dq_image_np[[row], :],
            dark_np[[row], :],
            algorithm,
            dq_flags,
        )
        if algorithm == "CYTHON_LIKELY":
            results.append(_fit_ramps_cython_row(args_row))
        else:
            results.append(_fit_ramps_jwst_row(args_row))

    for row_element in results:
        row, outdata, dq, var_poisson, var_rnoise, err = row_element

        slope_image_np[row, :] = outdata
        slope_errors_np[row, :] = err
        dq_image_np[row, :] = dq_image_np[row, :] | dq[0]
        var_poisson_image_np[row, :] = var_poisson
        var_rnoise_image_np[row, :] = var_rnoise

    mid_times_np[block_start:block_end, :] = (
        start_times_np[block_start:block_end, :]
        + calculate_midpoint_time_deltas(times, dq_raw_np[:, block_start:block_end, :], dq_flags)
        / 86400.0
    )

    _dq_raw = dq_raw_np[:, block_start:block_end, :] - (
        dq_raw_np[:, block_start:block_end, :] & dq_flags.DO_NOT_USE
    )
    dq_image_np[block_start:block_end, :] |= np.bitwise_or.reduce(_dq_raw)

    return (block_start, block_end)


def _jump_detection_worker(args: tuple) -> tuple:
    """
    Worker function for multiprocessing. Runs _jump_detection_worker on a block of rows using shared arrays.

    Parameters
    ----------
    args
        Tuple containing: (block_start, block_end, times, rejection_threshold, dq_flags, types_tuple)

    Returns
    -------
    tuple
        (block_start, block_end)
    """
    block_start, block_end, times, rejection_threshold, dq_flags, types_tuple = args
    mp_data_type, mp_float_type, mp_dq_raw_type, mp_dq_type = types_tuple

    ramps_np = shared_array_to_numpy(shared_data, shared_data_shape, dtype=mp_data_type)
    dq_raw_np = shared_array_to_numpy(shared_dq_raw, shared_dq_raw_shape, dtype=mp_dq_raw_type)
    dq_image_np = shared_array_to_numpy(shared_dq_image, shared_dq_image_shape, dtype=mp_dq_type)
    gain_np = shared_array_to_numpy(shared_gain, shared_gain_shape, dtype=mp_float_type)
    rdnoise_np = shared_array_to_numpy(
        shared_read_noise,
        shared_read_noise_shape,
        dtype=mp_float_type,
    )

    results = []
    for row in range(block_start, block_end):
        args_row = (
            row,
            rejection_threshold,
            ramps_np[np.newaxis, :, [row], :],
            # Make sure to add a new axis to each array to maintain number of dimensions
            rdnoise_np[[row], :],
            gain_np[[row], :],
            dq_raw_np[np.newaxis, :, [row], :],
            dq_image_np[[row], :],
            dq_flags,
        )

        results.append(_jump_detection_jwst_row(args_row))

    for row_element in results:
        _row, _dq = row_element

        dq_raw_np[:, [_row], :] = _dq

    return (block_start, block_end)


def fit_ramps_jwst(
    times: np.ndarray,
    input_data: np.ndarray,
    dq_flags: tuple | None = None,
    input_dq_raw: np.ndarray | None = None,
    input_dq: np.ndarray | None = None,
    rdnoise: np.ndarray | None = None,
    gain: np.ndarray | None = None,
    dark_current: np.ndarray | None = None,
    start_times: np.ndarray | None = None,
    jump_detection: bool = False,
    jump_rejection_threshold: float = 4.0,
    max_cores: int = 1,
    algorithm: str = "CYTHON_LIKELY",
) -> dict:
    """
    Fit ramps using JWST ramp fitting algorithms from stcal.ramp_fitting.

    Parameters
    ----------
    times
        1D array of shape (Nreads,) containing the read times.
    input_data
        3D array of ramp data with shape (Nreads, Nrows, Ncols).
    dq_flags
        Dict mapping DQ flag names to bit values, with keys DO_NOT_USE, SATURATED, JUMP_DET,
        UNRELIABLE_FLAT, UNRELIABLE_SLOPE and OUTLIER, plus GOOD and REFERENCE_PIXEL when
        jump_detection is True. If None, the JWST bit values are used: GOOD=0,
        DO_NOT_USE=2**0, SATURATED=2**1, JUMP_DET=2**2, OUTLIER=2**4, UNRELIABLE_SLOPE=2**24,
        UNRELIABLE_FLAT=2**25 and REFERENCE_PIXEL=2**31. Default is None.
    input_dq_raw
        3D array of data quality flags across detector (Nreads, Nrows, Ncols) Default is None.
    input_dq
        2D array of data quality flags across detector (Nrows, Ncols) Default is None.
    rdnoise
        2D array of read noise across detector (Nrows, Ncols) Default is None.
    gain
        2D array of gain across detector (Nrows, Ncols) Default is None.
    dark_current
        2D array of dark current across detector (Nrows, Ncols) Default is None.
    start_times
        2D array of MJD times of the first read across detector (Nrows, Ncols). Default is None.
    jump_detection
        Whether to perform jump detection before ramp fitting. Default is False.
    jump_rejection_threshold
        Sigma threshold for jump detection. Default is 4.0.
    max_cores
        Number of cores for parallelization. Default is 1 (no parallelization).
    algorithm
        Ramp fitting algorithm to use. Options are 'JWST_LIKELY', 'JWST_FIXSEN' or 'CYTHON_LIKELY'.
        Default is 'CYTHON_LIKELY'.

    Returns
    -------
    output : dict[str, np.ndarray]
        Dictionary containing the L0 rate arrays.
    """

    Nreads, Nrows, Ncols = input_data.shape

    dq_flags = dq_flags_to_namedtuple(_DEFAULT_DQ_FLAGS if dq_flags is None else dq_flags)
    if input_dq_raw is None:
        input_dq_raw = np.zeros((Nreads, Nrows, Ncols), dtype=np.uint8)
    if start_times is None:
        start_times = np.zeros((Nrows, Ncols), dtype=np.float64)

    if algorithm.lower() in ("jwst_likely", "jwst_like", "jwst_likelihood"):
        algorithm = "LIKELY"
    elif algorithm.lower() in ("jwst_fixsen", "jwst_fixsen_c"):
        algorithm = "FIXSEN_C"
    elif algorithm.lower() in ("cython_likely", "cython_like", "cython_likelihood"):
        algorithm = "CYTHON_LIKELY"
    else:
        raise ValueError(f"Unrecognized JWST ramp fitting algorithm {algorithm}")

    mp_data_type = ctypes.c_ushort if input_data.dtype is np.uint16 else ctypes.c_float
    mp_float_type = ctypes.c_float
    mp_dq_raw_type = ctypes.c_uint8
    mp_dq_type = ctypes.c_uint
    types_tuple = (mp_data_type, mp_float_type, mp_dq_raw_type, mp_dq_type)

    slope_image_mp = RawArray(mp_float_type, Ncols * Nrows)
    slope_image_shape = (Nrows, Ncols)
    slope_image_np = shared_array_to_numpy(slope_image_mp, slope_image_shape, dtype=mp_float_type)
    slope_image_np[:] = np.zeros((Nrows, Ncols), dtype=np.float32)

    slope_errors_mp = RawArray(mp_float_type, Ncols * Nrows)
    slope_errors_shape = (Nrows, Ncols)
    slope_errors_np = shared_array_to_numpy(
        slope_errors_mp,
        slope_errors_shape,
        dtype=mp_float_type,
    )
    slope_errors_np[:] = np.zeros((Nrows, Ncols), dtype=np.float32)

    var_poisson_image_mp = RawArray(mp_float_type, Ncols * Nrows)
    var_poisson_image_shape = (Nrows, Ncols)
    var_poisson_image_np = shared_array_to_numpy(
        var_poisson_image_mp,
        var_poisson_image_shape,
        dtype=mp_float_type,
    )
    var_poisson_image_np[:] = np.zeros((Nrows, Ncols), dtype=np.float32)

    var_rnoise_image_mp = RawArray(mp_float_type, Ncols * Nrows)
    var_rnoise_image_shape = (Nrows, Ncols)
    var_rnoise_image_np = shared_array_to_numpy(
        var_rnoise_image_mp,
        var_rnoise_image_shape,
        dtype=mp_float_type,
    )
    var_rnoise_image_np[:] = np.zeros((Nrows, Ncols), dtype=np.float32)

    dq_image_mp = RawArray(mp_dq_type, Ncols * Nrows)
    dq_image_shape = (Nrows, Ncols)
    dq_image_np = shared_array_to_numpy(dq_image_mp, dq_image_shape, dtype=mp_dq_type)
    if input_dq is None:
        dq_image_np[:] = np.zeros((Nrows, Ncols), dtype=np.uint32)
    else:
        dq_image_np[:] = input_dq

    start_times_mp = RawArray(mp_float_type, Ncols * Nrows)
    start_times_shape = (Nrows, Ncols)
    start_times_np = shared_array_to_numpy(start_times_mp, start_times_shape, dtype=mp_float_type)
    if start_times is None:
        start_times_np[:] = np.zeros((Nrows, Ncols), dtype=np.float32)
    else:
        start_times_np[:] = start_times

    mid_times_mp = RawArray(mp_float_type, Ncols * Nrows)
    mid_times_shape = (Nrows, Ncols)
    mid_times_np = shared_array_to_numpy(mid_times_mp, mid_times_shape, dtype=mp_float_type)
    mid_times_np[:] = np.zeros((Nrows, Ncols), dtype=np.float32)

    gain_mp = RawArray(mp_float_type, Ncols * Nrows)
    gain_shape = (Nrows, Ncols)
    gain_np = shared_array_to_numpy(gain_mp, gain_shape, dtype=mp_float_type)
    if gain is None:
        # TODO? - optimize gain when array is not passed
        gain_np[:] = np.ones((Nrows, Ncols), dtype=np.float32)
    else:
        gain_np[:] = gain

    dark_mp = RawArray(mp_float_type, Ncols * Nrows)
    dark_shape = (Nrows, Ncols)
    dark_np = shared_array_to_numpy(dark_mp, dark_shape, dtype=mp_float_type)
    if dark_current is None:
        dark_np[:] = np.zeros((Nrows, Ncols), dtype=np.float32)
    else:
        dark_np[:] = dark_current

    rdnoise_mp = RawArray(mp_float_type, Ncols * Nrows)
    rdnoise_shape = (Nrows, Ncols)
    rdnoise_np = shared_array_to_numpy(rdnoise_mp, dark_shape, dtype=mp_float_type)
    if rdnoise is None:
        # TODO? - optimize rdnoise when array is not passed
        rdnoise_np[:] = np.ones((Nrows, Ncols), dtype=np.float32)
    else:
        rdnoise_np[:] = rdnoise

    if max_cores > 1:
        ramps_mp = RawArray(mp_data_type, Ncols * Nrows * Nreads)
        ramps_shape = (Nreads, Nrows, Ncols)
        ramps_np = shared_array_to_numpy(ramps_mp, ramps_shape, dtype=mp_data_type)
        ramps_np[:] = input_data  # copy original array into shared array
        input_data = ramps_np  # free the original array from memory to avoid duplicates

        dq_raw_mp = RawArray(mp_dq_raw_type, Ncols * Nrows * Nreads)
        dq_raw_shape = (Nreads, Nrows, Ncols)
        dq_raw_np = shared_array_to_numpy(dq_raw_mp, dq_raw_shape, dtype=mp_dq_raw_type)
        if input_dq_raw is None:
            dq_raw_np[:] = np.zeros((Nreads, Nrows, Ncols), dtype=np.uint8)
        else:
            dq_raw_np[:] = input_dq_raw  # copy original array into shared array
            input_dq_raw = dq_raw_np  # free the original array from memory to avoid duplicates

        # Prepare blocks
        block_size = max(1, Nrows // (4 * max_cores)) if max_cores > 1 else Nrows

        _init_args = (
            ramps_mp,
            ramps_shape,
            dq_raw_mp,
            dq_raw_shape,
            gain_mp,
            gain_shape,
            dark_mp,
            dark_shape,
            rdnoise_mp,
            rdnoise_shape,
            slope_image_mp,
            slope_image_shape,
            slope_errors_mp,
            slope_errors_shape,
            var_poisson_image_mp,
            var_poisson_image_shape,
            var_rnoise_image_mp,
            var_rnoise_image_shape,
            dq_image_mp,
            dq_image_shape,
            start_times_mp,
            start_times_shape,
            mid_times_mp,
            mid_times_shape,
        )

        ctx = mp.get_context("spawn")  # avoids unsafe fork after OpenMP init
        tpool = ctx.Pool(
            processes=max_cores,
            initializer=_tpool_init_jwst_ramp,
            initargs=_init_args,
            maxtasksperchild=50,
        )

        if jump_detection:
            args_list = []
            for block_start in range(0, Nrows, block_size):
                block_end = min(block_start + block_size, Nrows)

                args = (
                    block_start,
                    block_end,
                    times,
                    jump_rejection_threshold,
                    dq_flags,
                    types_tuple,
                )
                args_list.append(args)

            tasks = [
                tpool.apply_async(_jump_detection_worker, args=(args_tuple,))
                for args_tuple in args_list
            ]

            for t in progressbar(tasks, desc="Fitting Ramps", fast=False):
                t.wait()

        args_list = []
        for block_start in range(0, Nrows, block_size):
            block_end = min(block_start + block_size, Nrows)

            args = (block_start, block_end, times, algorithm, dq_flags, types_tuple)
            args_list.append(args)

        try:
            with numba_thread_scope(max_cores):
                tasks = [
                    tpool.apply_async(_fit_ramps_worker, args=(args_tuple,))
                    for args_tuple in args_list
                ]

            for t in progressbar(tasks, desc="Fitting Ramps", fast=False):
                t.wait()
        finally:
            tpool.close()
            tpool.join()
    else:
        # If not parallelizing, ie max_cores=1, just run directly without shared memory
        results = []
        for row in progressbar(np.arange(Nrows), desc="Fitting Ramps", fast=False):
            if jump_detection:
                args_row = (
                    row,
                    jump_rejection_threshold,
                    input_data[np.newaxis, :, [row], :],
                    # Make sure to add a new axis to each array to maintain number of dimensions
                    rdnoise_np[[row], :],
                    gain_np[[row], :],
                    input_dq_raw[np.newaxis, :, [row], :],
                    dq_image_np[[row], :],
                    dq_flags,
                )

                _row, _dq = _jump_detection_jwst_row(args_row)
                input_dq_raw[:, [_row], :] = _dq

            args = (
                row,
                times,
                # Reshape to add the nints axis that JWST pipeline requires
                input_data[np.newaxis, :, [row], :],
                # Make sure to add a new axis to each array to maintain number of dimensions
                rdnoise_np[[row], :],
                gain_np[[row], :],
                input_dq_raw[np.newaxis, :, [row], :],
                dq_image_np[[row], :],
                dark_np[[row], :],
                algorithm,
                dq_flags,
            )

            if algorithm == "CYTHON_LIKELY":
                results.append(_fit_ramps_cython_row(args))
            else:
                results.append(_fit_ramps_jwst_row(args))

        for row_element in results:
            row, outdata, dq, var_poisson, var_rnoise, err = row_element

            slope_image_np[row, :] = outdata
            slope_errors_np[row, :] = err
            dq_image_np[row, :] = dq_image_np[row, :] | dq[0]
            var_poisson_image_np[row, :] = var_poisson
            var_rnoise_image_np[row, :] = var_rnoise

        with numba_thread_scope(max_cores):
            mid_times_np = (
                start_times
                + calculate_midpoint_time_deltas(times, input_dq_raw, dq_flags) / 86400.0
            )

    output = {}
    output["rate"] = slope_image_np
    output["rate_err"] = slope_errors_np
    output["rate_var_rnoise"] = var_rnoise_image_np
    output["rate_var_poisson"] = var_poisson_image_np
    output["mid_time"] = mid_times_np
    output["dq"] = dq_image_np

    return output


def _tpool_init_jwst_ramp(
    data: ctypes.Array,
    data_shape: tuple,
    dq_raw: ctypes.Array,
    dq_raw_shape: tuple,
    gain: ctypes.Array,
    gain_shape: tuple,
    dark: ctypes.Array,
    dark_shape: tuple,
    read_noise: ctypes.Array,
    read_noise_shape: tuple,
    slope_image: ctypes.Array,
    slope_image_shape: tuple,
    slope_errors: ctypes.Array,
    slope_errors_shape: tuple,
    var_poisson_image: ctypes.Array,
    var_poisson_image_shape: tuple,
    var_rnoise_image: ctypes.Array,
    var_rnoise_image_shape: tuple,
    dq_image: ctypes.Array,
    dq_image_shape: tuple,
    start_times: ctypes.Array,
    start_times_shape: tuple,
    mid_times: ctypes.Array,
    mid_times_shape: tuple,
):
    """
    Initialize shared global variables for the multiprocessing pool.
    """

    global \
        shared_data, \
        shared_data_shape, \
        shared_dq_raw, \
        shared_dq_raw_shape, \
        shared_gain, \
        shared_gain_shape, \
        shared_dark, \
        shared_dark_shape, \
        shared_read_noise, \
        shared_read_noise_shape, \
        shared_slope_image, \
        shared_slope_image_shape, \
        shared_slope_errors, \
        shared_slope_errors_shape, \
        shared_var_poisson_image, \
        shared_var_poisson_image_shape, \
        shared_var_rnoise_image, \
        shared_var_rnoise_image_shape, \
        shared_dq_image, \
        shared_dq_image_shape, \
        shared_start_times, \
        shared_start_times_shape, \
        shared_mid_times, \
        shared_mid_times_shape

    shared_data = data
    shared_data_shape = data_shape

    shared_dq_raw = dq_raw
    shared_dq_raw_shape = dq_raw_shape

    shared_gain = gain
    shared_gain_shape = gain_shape

    shared_dark = dark
    shared_dark_shape = dark_shape

    shared_read_noise = read_noise
    shared_read_noise_shape = read_noise_shape

    shared_slope_image = slope_image
    shared_slope_image_shape = slope_image_shape

    shared_slope_errors = slope_errors
    shared_slope_errors_shape = slope_errors_shape

    shared_var_poisson_image = var_poisson_image
    shared_var_poisson_image_shape = var_poisson_image_shape

    shared_var_rnoise_image = var_rnoise_image
    shared_var_rnoise_image_shape = var_rnoise_image_shape

    shared_dq_image = dq_image
    shared_dq_image_shape = dq_image_shape

    shared_start_times = start_times
    shared_start_times_shape = start_times_shape

    shared_mid_times = mid_times
    shared_mid_times_shape = mid_times_shape
