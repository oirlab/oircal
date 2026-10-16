"""Tests for converting rates to e-/s with the gain."""

import numpy as np

from oircal.gain import correct_gain


def _inputs():
    rate = np.array([[10.0, 0.0], [-3.0, 40.0]], dtype=np.float32)  # DN/s
    err = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gain = np.array([[1.5, 2.0], [1.0, 3.0]], dtype=np.float32)  # e-/DN
    gain_err = np.array([[0.1, 0.2], [0.05, 0.3]], dtype=np.float32)
    return rate, err, gain, gain_err


def test_correct_gain_rate_err_and_dq():
    rate, err, gain, gain_err = _inputs()
    dq = np.zeros((2, 2), dtype=np.uint32)
    gain_dq = np.array([[0, 1], [2, 4]], dtype=np.uint32)
    r, e, d = rate.copy(), err.copy(), dq.copy()
    correct_gain(r, gain, input_err=e, input_dq=d, gain_err=gain_err, gain_dq=gain_dq)

    np.testing.assert_allclose(r, rate * gain)
    # sigma^2 = (g sigma_r)^2 + (r sigma_g)^2, finite at zero rate
    np.testing.assert_allclose(e, np.sqrt((gain * err) ** 2 + (rate * gain_err) ** 2), rtol=1e-6)
    assert np.isfinite(e).all()
    np.testing.assert_array_equal(d, gain_dq)


def test_correct_gain_noiseless_gain():
    rate, err, gain, _ = _inputs()
    e = err.copy()
    correct_gain(rate.copy(), gain, input_err=e)
    np.testing.assert_allclose(e, err * gain)


def test_correct_gain_variances():
    rate, err, gain, _ = _inputs()
    var_poisson = np.array([[0.5, 1.0], [2.0, 4.0]], dtype=np.float32)
    var_rnoise = err**2 - var_poisson
    vp, vr, e = var_poisson.copy(), var_rnoise.copy(), err.copy()
    correct_gain(rate.copy(), gain, input_err=e, input_var_poisson=vp, input_var_rnoise=vr)

    np.testing.assert_allclose(vp, var_poisson * gain**2)
    np.testing.assert_allclose(vr, var_rnoise * gain**2)
    # Without a gain error, the variances still sum to err**2
    np.testing.assert_allclose(vp + vr, e**2, rtol=1e-6)


def test_correct_gain_matches_hispecdrp_for_nonzero_rates():
    rate, err, gain, gain_err = _inputs()
    good = rate != 0
    e = err.copy()
    correct_gain(rate.copy(), gain, input_err=e, gain_err=gain_err)
    # hispecdrp: |r g| sqrt((sigma_r / r)^2 + (sigma_g / g)^2), undefined at zero rate
    r, g, se, sg = rate[good], gain[good], err[good], gain_err[good]
    expected = np.abs(r * g) * np.sqrt((se / r) ** 2 + (sg / g) ** 2)
    np.testing.assert_allclose(e[good], expected, rtol=1e-6)
