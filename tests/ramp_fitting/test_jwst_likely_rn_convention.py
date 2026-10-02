"""jwst_likely must treat HISPEC RN as single-read σ (stcal expects CDS)."""

from __future__ import annotations

import warnings

import numpy as np

from oircal.ramp_fitting.fit_ramp_jwst import fit_ramps_jwst
from oircal.simulations.utr import create_ramp


def test_jwst_likely_var_rnoise_matches_single_read_ols_theory(dq_flags):
    """
    stcal.likely_fit divides RN by √2 (JWST CDS convention). Without converting
    HISPEC single-read RN → CDS before the call, reported var_rnoise is 2× low.
    """
    readtime = 1.7
    n_reads = 10
    rn = 1.0
    ny = nx = 48
    truth = np.zeros((ny, nx))
    times = np.arange(n_reads, dtype=np.float32) * np.float32(readtime)
    t0 = times.astype(np.float64) - times.astype(np.float64).mean()
    var_ols = rn**2 / float(np.sum(t0**2))

    gain = np.ones((ny, nx), np.float32)
    rn_map = np.full((ny, nx), rn, np.float32)
    dark = np.zeros((ny, nx), np.float32)
    start = np.zeros((ny, nx), np.float64)
    dq_raw = np.zeros((n_reads, ny, nx), np.uint8)
    dq = np.zeros((ny, nx), np.uint32)

    rates = []
    vrs = []
    for seed in range(25):
        ramp = create_ramp(
            truth,
            readtime=readtime,
            n_reads=n_reads,
            gain=1.0,
            flat=1.0,
            bias=1000.0,
            dark=0.0,
            read_noise=rn,
            kTC_noise=0.0,
            poisson_noise=False,
            convert_to_uint16=False,
            clip_ramps=False,
            nonlin_coeffs=np.array([0.0, 1.0, 0.0]),
            max_cores=1,
            seed=seed,
        )
        data = np.asarray(ramp["data"], np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = fit_ramps_jwst(
                times,
                data,
                dq_flags,
                dq_raw,
                dq,
                rdnoise=rn_map,
                gain=gain,
                dark_current=dark,
                start_times=start,
                jump_detection=False,
                max_cores=1,
                algorithm="jwst_likely",
            )
        rates.append(np.asarray(out["rate"], float).ravel())
        vrs.append(np.asarray(out["rate_var_rnoise"], float).ravel())

    rates = np.concatenate(rates)
    vrs = np.concatenate(vrs)
    med_vr = float(np.median(vrs))
    emp_var = float(np.var(rates))

    # Reported RN variance must match OLS theory (not half of it).
    assert abs(med_vr / var_ols - 1.0) < 0.05, (med_vr, var_ols)
    assert abs(emp_var / var_ols - 1.0) < 0.15, (emp_var, var_ols)
    assert abs(emp_var / med_vr - 1.0) < 0.15, (emp_var, med_vr)
