"""Tests for making the dark calibration."""

import numpy as np

from oircal.dark_subtraction import make_dark, subtract_dark_rate


FLAGS = {"HOT": 2**11}


def test_make_dark_flags_hot_pixels():
    rate = np.array([[0.01, 6.0], [4.9, 100.0]], dtype=np.float32)
    err = np.full((2, 2), 0.1, dtype=np.float32)
    input_dq = np.array([[1, 0], [0, 0]], dtype=np.uint32)
    out = make_dark(rate, input_err=err, input_dq=input_dq, dq_flags=FLAGS, hot_pixel_threshold=5)

    np.testing.assert_array_equal(out["dark_rate"], rate)
    np.testing.assert_array_equal(out["dark_err"], err)
    np.testing.assert_array_equal(out["dark_dq"], [[1, 2**11], [0, 2**11]])
    assert out["dark_rate"] is not rate  # copies, inputs untouched
    assert input_dq[0, 1] == 0


def test_make_dark_output_feeds_subtract_dark_rate():
    dark = make_dark(np.full((2, 2), 0.5, dtype=np.float32), input_err=np.full((2, 2), 0.1))
    assert dark["dark_err"] is not None
    rate = np.full((2, 2), 2.0, dtype=np.float32)
    rate_err = np.full((2, 2), 0.1, dtype=np.float32)
    subtract_dark_rate(rate, input_err=rate_err, **dark)
    np.testing.assert_allclose(rate, 1.5)
    np.testing.assert_allclose(rate_err, np.sqrt(0.02))


def test_make_dark_without_err_or_flags():
    out = make_dark(np.full((1, 2), 10.0))
    assert out["dark_err"] is None
    assert not out["dark_dq"].any()
