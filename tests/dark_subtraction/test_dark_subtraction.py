"""Tests for dark subtraction.

Ported from the dark section of hispecdrp's ``test_stage0_units``.
"""
from oircal.dark_subtraction import subtract_dark_rate
import numpy as np


def _make_inputs():
    rate = np.array([[10.0, 20.0],
                     [30.0, 40.0]], dtype=np.float32)
    err = np.array([[1.0, 2.0],
                    [3.0, 4.0]], dtype=np.float32)
    dq = np.zeros((2, 2), dtype=np.uint32)
    dark_rate = np.array([[1.0, 2.0],
                          [3.0, 4.0]], dtype=np.float32)
    dark_err = np.array([[0.5, 0.5],
                         [1.0, 1.0]], dtype=np.float32)
    dark_dq = np.array([[0, 1],
                        [2, 4]], dtype=np.uint32)
    return rate, err, dq, dark_rate, dark_err, dark_dq


def test_subtract_dark_rate():
    rate, err, dq, dark_rate, dark_err, dark_dq = _make_inputs()

    rate_out, err_out, dq_out = subtract_dark_rate(
        rate.copy(), dark_rate, err.copy(), dq.copy(), dark_err, dark_dq
    )

    np.testing.assert_array_equal(
        rate_out,
        np.array([[9.0, 18.0],
                  [27.0, 36.0]], dtype=np.float32),
        err_msg="Dark subtraction failed"
    )
    np.testing.assert_array_equal(
        dq_out,
        np.array([[0, 1],
                  [2, 4]], dtype=np.uint32),
        err_msg="Dark DQ propagation failed"
    )
    np.testing.assert_allclose(
        err_out, np.sqrt(err**2 + dark_err**2), rtol=1e-6,
        err_msg="Error propagation incorrect"
    )


def test_subtract_dark_rate_without_err_or_dq():
    rate, err, dq, dark_rate, dark_err, dark_dq = _make_inputs()

    # No errors or DQ: rate still corrected, others returned as None
    rate_out, err_out, dq_out = subtract_dark_rate(rate.copy(), dark_rate)
    np.testing.assert_array_equal(rate_out, rate - dark_rate)
    assert err_out is None and dq_out is None

    # Input err/dq without the dark's: left unchanged
    _, err_out, dq_out = subtract_dark_rate(rate.copy(), dark_rate, err.copy(), dq.copy())
    np.testing.assert_array_equal(err_out, err)
    np.testing.assert_array_equal(dq_out, dq)
