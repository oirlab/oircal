"""Tests for failed fits in the Numba OLS and MCDS ramp fitting."""

import numpy as np
import pytest

from oircal.ramp_fitting import fit_ramps_mcds, fit_ramps_ols


Nreads = 10
times = np.arange(Nreads, dtype=np.float64)
ramp = (5.0 * times).astype(np.float32).reshape(Nreads, 1, 1)  # noiseless 5 DN/s
rdnoise = np.ones((1, 1), dtype=np.float32)


def _assert_failed(out, dq_flags):
    for key in ("rate", "rate_err", "rate_var_rnoise", "rate_var_poisson"):
        assert np.isnan(out[key][0, 0]), key
    assert out["dq"][0, 0] & dq_flags["DO_NOT_USE"]


@pytest.mark.parametrize("fit", [fit_ramps_ols, fit_ramps_mcds])
def test_all_reads_bad(fit, dq_flags):
    dq_raw = np.full((Nreads, 1, 1), dq_flags["DO_NOT_USE"], dtype=np.uint8)
    out = fit(times, ramp.copy(), input_dq_raw=dq_raw, dq_flags=dq_flags, rdnoise=rdnoise)
    _assert_failed(out, dq_flags)


def test_mcds_num_coadd_longer_than_segment(dq_flags):
    # Only reads 0 and 1 are usable, so averaging 2 reads at each end leaves no baseline
    dq_raw = np.zeros((Nreads, 1, 1), dtype=np.uint8)
    dq_raw[2:] = dq_flags["DO_NOT_USE"]
    out = fit_ramps_mcds(
        times,
        ramp.copy(),
        input_dq_raw=dq_raw,
        dq_flags=dq_flags,
        rdnoise=rdnoise,
        num_coadd=2,
    )
    _assert_failed(out, dq_flags)


def test_ols_bias_undefined_without_first_read(dq_flags):
    dq_raw = np.zeros((Nreads, 1, 1), dtype=np.uint8)
    dq_raw[0] = dq_flags["DO_NOT_USE"]
    out = fit_ramps_ols(times, ramp.copy(), input_dq_raw=dq_raw, dq_flags=dq_flags, rdnoise=rdnoise)
    np.testing.assert_allclose(out["rate"][0, 0], 5.0, rtol=1e-5)
    assert np.isnan(out["bias"][0, 0]) and np.isnan(out["bias_err"][0, 0])
    assert not out["dq"][0, 0] & dq_flags["DO_NOT_USE"]


@pytest.mark.parametrize("fit", [fit_ramps_ols, fit_ramps_mcds])
def test_good_ramp_not_flagged(fit, dq_flags):
    out = fit(times, ramp.copy(), dq_flags=dq_flags, rdnoise=rdnoise)
    np.testing.assert_allclose(out["rate"][0, 0], 5.0, rtol=1e-5)
    assert np.isfinite(out["rate_err"][0, 0])
    assert out["dq"][0, 0] == 0


@pytest.mark.parametrize("fit", [fit_ramps_ols, fit_ramps_mcds])
def test_invalid_gain_fails(fit, dq_flags):
    out = fit(
        times,
        ramp.copy(),
        dq_flags=dq_flags,
        rdnoise=rdnoise,
        gain=np.zeros((1, 1), dtype=np.float32),
    )
    _assert_failed(out, dq_flags)


@pytest.mark.parametrize("fit", [fit_ramps_ols, fit_ramps_mcds])
def test_zero_read_noise_has_no_read_noise_variance(fit, dq_flags):
    out = fit(times, ramp.copy(), dq_flags=dq_flags, rdnoise=np.zeros((1, 1), dtype=np.float32))
    assert out["rate_var_rnoise"][0, 0] == 0
    np.testing.assert_allclose(out["rate_err"][0, 0] ** 2, out["rate_var_poisson"][0, 0], rtol=1e-6)


@pytest.mark.parametrize("fit", [fit_ramps_ols, fit_ramps_mcds])
def test_noiseless_flat_ramp_is_exact(fit, dq_flags):
    # No read noise and no signal: zero variance, which is an exact fit, not a failure
    flat = np.zeros((Nreads, 1, 1), dtype=np.float32)
    out = fit(times, flat, dq_flags=dq_flags)
    assert out["rate"][0, 0] == 0 and out["rate_err"][0, 0] == 0
    assert out["dq"][0, 0] == 0
