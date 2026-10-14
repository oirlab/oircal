"""Tests for fitting the non-linearity correction.

Ported from hispecdrp's ``test_process_sat_flats_pipeline``, replacing the pipeline
with direct calls to ``check_saturation`` and ``fit_nonlin_coefs``.
"""

import numpy as np

from oircal.nonlinear_correction import fit_nonlin_coefs
from oircal.saturation import check_saturation
from oircal.simulations import create_ramp


def test_fit_nonlin_coefs(dq_flags):
    np.random.seed(12345)

    # --- Define parameters for the simulated ramp ---
    max_cores = 1
    max_slope = 1000  # e-/s
    readtime = 1.7  # seconds
    Nreads = 60
    Nfiles = 4
    shape = (2, 2)  # size of the detector
    read_noise = 0.1
    gain = 1.0
    bias = 1000.0
    kTC_noise = 0.0
    sat_thresh = np.full(shape, np.iinfo(np.uint16).max, dtype=np.uint16)
    satval_frac = 0.95

    # --- Non-linearity and its inverse polynomial coefficients ---

    # Create an ideal ramp
    ideal_ramp = np.linspace(0, 1.5 * np.iinfo(np.uint16).max, 1001, endpoint=True)
    nonlin_coeffs_inject = [-1e-6, 1, 0]
    ramp_nonlinear = np.polyval(nonlin_coeffs_inject, ideal_ramp)

    # Use a pre-validated, precise correction for comparison against the fitted correction below.
    nonlin_correction_coefs = np.array(
        [
            0.00000000e00,
            9.93166438e-01,
            9.93459922e-07,
            1.99687217e-12,
            4.28159139e-18,
            2.47963102e-23,
        ],
    )[::-1]
    nonlin_corrected_ramp = np.polyval(nonlin_correction_coefs, ramp_nonlinear)

    gain_map = np.full(shape, gain, dtype=np.float32)
    rn_map = np.full(shape, read_noise, dtype=np.float32) / gain_map
    bias_map = np.full(shape, bias, dtype=np.float32) / gain_map

    # --- Create fake UTR cubes and flag saturation ---
    electron_rate_map = np.full(shape, max_slope, dtype=np.float32)
    ramp_arr = []
    dq_raw_arr = []
    for _ in range(Nfiles):
        ramp_data = create_ramp(
            source=electron_rate_map,
            gain=gain_map,
            bias=bias,
            kTC_noise=kTC_noise,
            nonlin_coeffs=nonlin_coeffs_inject,
            dark=0,
            Nreads=Nreads,
            readtime=readtime,
            poisson_noise=False,
            read_noise=read_noise,
            convert_to_uint16=True,
            clip_ramps=True,
            max_cores=max_cores,
        )
        data = ramp_data["data"]
        dq_raw = np.zeros(data.shape, dtype=np.uint8)
        dq = np.zeros(shape, dtype=np.uint32)
        check_saturation(
            data,
            dq_raw,
            dq,
            sat_thresh,
            np.zeros(shape, dtype=np.uint32),
            dq_flags=dq_flags,
            max_cores=max_cores,
        )
        ramp_arr.append(data)
        dq_raw_arr.append(dq_raw)
    ramp_arr = np.array(ramp_arr)
    dq_raw_arr = np.array(dq_raw_arr)

    satval = satval_frac * (sat_thresh - bias_map)

    fit_kwargs = {
        "dq_flags": dq_flags,
        "order": 5,
        "satval": satval,
        "use_legendre": True,
        "readstart": 0,
        "criterion_for_nonlin_thresh": 0.01,
        "criterion_for_nonlin_max": 0.05,
        "max_cores": max_cores,
    }

    # Full detector mode
    coeffs, nonlin_thresh, nonlin_max = fit_nonlin_coefs(
        ramp_arr,
        gain_map,
        rn_map,
        input_dq_raw=dq_raw_arr,
        mode="full_detector",
        **fit_kwargs,
    )

    nonlin_corrected_ramp_4testing = np.polyval(coeffs[::-1], ramp_nonlinear)
    rel_err_ideal = (nonlin_corrected_ramp[1::] - ideal_ramp[1::]) / ideal_ramp[1::]
    rel_err_4testing = (nonlin_corrected_ramp_4testing[1::] - ideal_ramp[1::]) / ideal_ramp[1::]
    assert np.sum(np.abs(rel_err_ideal - rel_err_4testing) > 1e-2) == 0
    assert abs(nonlin_thresh - 18000) < 500
    assert abs(nonlin_max - 52600) < 500

    # Per pixel mode
    coeffs, nonlin_thresh, nonlin_max = fit_nonlin_coefs(
        ramp_arr,
        gain_map,
        rn_map,
        input_dq_raw=dq_raw_arr,
        mode="per_pixel",
        **fit_kwargs,
    )

    nonlin_corrected_ramp_4testing = np.polyval(coeffs[::-1, 0, 0], ramp_nonlinear)
    rel_err_ideal = (nonlin_corrected_ramp[1::] - ideal_ramp[1::]) / ideal_ramp[1::]
    rel_err_4testing = (nonlin_corrected_ramp_4testing[1::] - ideal_ramp[1::]) / ideal_ramp[1::]
    assert np.sum(np.abs(rel_err_ideal - rel_err_4testing) > 1e-2) == 0
    assert abs(nonlin_thresh[0, 0] - 18000) < 500
    assert abs(nonlin_max[0, 0] - 52600) < 500
