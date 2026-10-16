"""Tests for detector flat correction.

Ported from the detector flat section of hispecdrp's ``test_stage0_units``.
"""

import numpy as np

from oircal.detector_flat import correct_detector_flat


def _make_inputs():
    rate = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    err = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    dq = np.zeros((2, 2), dtype=np.uint32)
    flat = np.array([[2.0, 4.0], [0.5, 1.0]], dtype=np.float32)
    flat_err = np.array([[0.2, 0.4], [0.05, 0.1]], dtype=np.float32)
    flat_dq = np.array([[0, 1], [2, 4]], dtype=np.uint32)
    return rate, err, dq, flat, flat_err, flat_dq


def test_correct_detector_flat():
    rate, err, dq, flat, flat_err, flat_dq = _make_inputs()

    rate_out, err_out, dq_out = correct_detector_flat(
        rate.copy(),
        flat,
        err.copy(),
        dq.copy(),
        flat_err,
        flat_dq,
    )

    # Rate: elementwise division rate/flat
    np.testing.assert_array_equal(
        rate_out,
        np.array([[10.0 / 2.0, 20.0 / 4.0], [30.0 / 0.5, 40.0 / 1.0]], dtype=np.float32),
        err_msg="Flat correction (rate division) failed",
    )

    # DQ propagation: OR with flat_dq
    np.testing.assert_array_equal(
        dq_out,
        np.array([[0, 1], [2, 4]], dtype=np.uint32),
        err_msg="Flat DQ propagation failed",
    )

    for i in range(2):
        for j in range(2):
            np.testing.assert_almost_equal(
                err_out[i, j],
                rate[i, j]
                / flat[i, j]
                * np.sqrt((err[i, j] / rate[i, j]) ** 2 + (flat_err[i, j] / flat[i, j]) ** 2),
                decimal=5,
                err_msg="Error propagation incorrect in flat correction",
            )


def test_correct_detector_flat_noiseless_flat():
    rate, err, dq, flat, flat_err, flat_dq = _make_inputs()

    # Without flat_err the flat is noiseless: err is just scaled by 1/flat
    rate_out, err_out, dq_out = correct_detector_flat(rate.copy(), flat, err.copy())
    np.testing.assert_allclose(rate_out, rate / flat, rtol=1e-6)
    np.testing.assert_allclose(err_out, err / flat, rtol=1e-6)
    assert dq_out is None

    # Input DQ without flat DQ: left unchanged
    _, _, dq_out = correct_detector_flat(rate.copy(), flat, None, dq.copy())
    np.testing.assert_array_equal(dq_out, dq)


def test_correct_detector_flat_nonpositive_flat():
    rate, err, dq, flat, flat_err, flat_dq = _make_inputs()
    flat[0, 1] = 0.0
    flat[1, 0] = -1.0

    rate_out, err_out, dq_out = correct_detector_flat(
        rate.copy(),
        flat,
        err.copy(),
        dq.copy(),
        flat_err,
        flat_dq,
    )

    # Pixels with a non-positive flat are left uncorrected...
    for i, j in [(0, 1), (1, 0)]:
        assert rate_out[i, j] == rate[i, j]
        assert err_out[i, j] == err[i, j]
    # ...but the others are corrected, and DQ is still propagated everywhere
    np.testing.assert_allclose(rate_out[0, 0], rate[0, 0] / flat[0, 0])
    np.testing.assert_allclose(rate_out[1, 1], rate[1, 1] / flat[1, 1])
    np.testing.assert_array_equal(dq_out, flat_dq)


def test_correct_detector_flat_variances():
    rate, err, dq, flat, flat_err, flat_dq = _make_inputs()
    flat[0, 1] = 0.0  # not positive: left uncorrected
    var_poisson = np.array([[4.0, 9.0], [16.0, 25.0]], dtype=np.float32)
    var_rnoise = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    vp, vr, e = var_poisson.copy(), var_rnoise.copy(), err.copy()
    correct_detector_flat(
        rate.copy(),
        flat,
        input_err=e,
        input_var_poisson=vp,
        input_var_rnoise=vr,
    )

    safe_flat = np.where(flat > 0, flat, 1.0)
    expected_scale = 1 / safe_flat**2
    np.testing.assert_allclose(vp, var_poisson * expected_scale)
    np.testing.assert_allclose(vr, var_rnoise * expected_scale)
    # With a noiseless flat, the variances still sum to err**2 if they did before
    np.testing.assert_allclose(vp + vr, (var_poisson + var_rnoise) * expected_scale)
    np.testing.assert_allclose(e, err / safe_flat)
