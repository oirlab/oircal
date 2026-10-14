"""Compare cython_likely vs jwst_likely on rates and variance."""

from __future__ import annotations

import time
import warnings

import numpy as np

from oircal.ramp_fitting.fit_ramp_jwst import fit_ramps_jwst
from oircal.simulations.utr import create_ramp


def _simulate_and_fit(*, truth, rn, Nreads, seed, algorithm, poisson_noise, dq_flags):
    readtime = 1.7
    Nrows, Ncols = truth.shape
    times = np.arange(Nreads, dtype=np.float32) * np.float32(readtime)
    gain = np.ones((Nrows, Ncols), np.float32)
    rn_map = np.full((Nrows, Ncols), rn, np.float32)
    dark = np.zeros((Nrows, Ncols), np.float32)
    start = np.zeros((Nrows, Ncols), np.float64)
    dq_raw = np.zeros((Nreads, Nrows, Ncols), np.uint8)
    dq = np.zeros((Nrows, Ncols), np.uint32)

    ramp = create_ramp(
        truth,
        readtime=readtime,
        Nreads=Nreads,
        gain=1.0,
        flat=1.0,
        bias=1000.0,
        dark=0.0,
        read_noise=rn,
        kTC_noise=0.0,
        poisson_noise=poisson_noise,
        convert_to_uint16=False,
        clip_ramps=False,
        nonlin_coeffs=np.array([0.0, 1.0, 0.0]),
        max_cores=1,
        seed=seed,
    )
    data = np.asarray(ramp["data"], np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
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
            algorithm=algorithm,
        )
        dt = time.perf_counter() - t0
    return out, dt


def test_cython_parity_empty_sky_matches_jwst_likely_var_rnoise(dq_flags):
    """RN-only empty sky: cython parity var_rnoise ~ jwst_likely (~OLS)."""
    rn = 1.0
    Nreads = 10
    Nrows = Ncols = 32
    truth = np.zeros((Nrows, Ncols))
    times = np.arange(Nreads, dtype=np.float64) * 1.7
    t0 = times - times.mean()
    var_ols = rn**2 / float(np.sum(t0**2))

    out_j, _ = _simulate_and_fit(
        truth=truth,
        rn=rn,
        Nreads=Nreads,
        seed=0,
        algorithm="jwst_likely",
        poisson_noise=False,
        dq_flags=dq_flags,
    )
    out_c, _ = _simulate_and_fit(
        truth=truth,
        rn=rn,
        Nreads=Nreads,
        seed=0,
        algorithm="cython_likely",
        poisson_noise=False,
        dq_flags=dq_flags,
    )

    vr_j = float(np.median(out_j["rate_var_rnoise"]))
    vr_c = float(np.median(out_c["rate_var_rnoise"]))
    vp_j = float(np.median(out_j["rate_var_poisson"]))
    vp_c = float(np.median(out_c["rate_var_poisson"]))
    err_j = float(np.median(out_j["rate_err"]))
    err_c = float(np.median(out_c["rate_err"]))

    # Empty sky still has a tiny Wald Poisson term from a noisy positive cguess
    # (same as stcal); it should remain ≪ RN variance.
    assert vp_c < 0.2 * vr_c, (vp_c, vr_c)
    assert abs(vp_c - vp_j) < 0.2 * max(vr_j, 1e-12), (vp_c, vp_j, vr_j)
    assert abs(vr_j / var_ols - 1.0) < 0.08, (vr_j, var_ols)
    assert abs(vr_c / var_ols - 1.0) < 0.08, (vr_c, var_ols)
    assert abs(vr_c / vr_j - 1.0) < 0.05, (vr_c, vr_j)
    assert abs(err_c / err_j - 1.0) < 0.05, (err_c, err_j)


def test_cython_parity_rate_and_err_close_on_illuminated(dq_flags):
    """Illuminated field: rates and total rate_err within a few percent of jwst."""
    rn = 8.5
    Nreads = 10
    Nrows = Ncols = 48
    rng = np.random.default_rng(1)
    truth = rng.uniform(0.0, 50.0, size=(Nrows, Ncols))

    out_j, dt_j = _simulate_and_fit(
        truth=truth,
        rn=rn,
        Nreads=Nreads,
        seed=2,
        algorithm="jwst_likely",
        poisson_noise=True,
        dq_flags=dq_flags,
    )
    out_c, dt_c = _simulate_and_fit(
        truth=truth,
        rn=rn,
        Nreads=Nreads,
        seed=2,
        algorithm="cython_likely",
        poisson_noise=True,
        dq_flags=dq_flags,
    )

    rate_j = np.asarray(out_j["rate"], float)
    rate_c = np.asarray(out_c["rate"], float)
    err_j = np.asarray(out_j["rate_err"], float)
    err_c = np.asarray(out_c["rate_err"], float)
    vp_c = np.asarray(out_c["rate_var_poisson"], float)
    vr_c = np.asarray(out_c["rate_var_rnoise"], float)

    good = np.isfinite(rate_j) & np.isfinite(rate_c) & np.isfinite(err_j) & np.isfinite(err_c)
    good &= err_j > 0
    assert np.count_nonzero(good) > 0.9 * rate_j.size

    med_abs_rate = float(np.median(np.abs(rate_j[good])))
    med_rel_rate = float(
        np.median(np.abs(rate_c[good] - rate_j[good]) / np.maximum(err_j[good], 1e-12)),
    )
    med_err_ratio = float(np.median(err_c[good] / err_j[good]))
    split_ratio = float(
        np.median(np.sqrt(np.maximum(vp_c[good] + vr_c[good], 0.0)) / err_c[good]),
    )

    # Rates should agree at the ~noise level (same ramp, same theory).
    assert med_rel_rate < 0.25, (med_rel_rate, med_abs_rate)
    assert abs(med_err_ratio - 1.0) < 0.1, med_err_ratio
    assert abs(split_ratio - 1.0) < 0.05, split_ratio
    # Soft speed check: cython should not be dramatically slower than stcal here.
    # (Absolute speedup varies; we mainly guard against accidental regression.)
    assert dt_c < 5.0 * dt_j + 0.5, (dt_c, dt_j)
