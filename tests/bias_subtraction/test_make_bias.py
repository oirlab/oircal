"""Tests for making the bias calibration."""

import numpy as np

from oircal.bias_subtraction import make_bias


FLAGS = {"UNRELIABLE_BIAS": 2**22}


def test_make_bias_median_and_ktc():
    rng = np.random.default_rng(0)
    first_reads = 1000 + rng.normal(0, 20, (200, 3, 4))
    out = make_bias(first_reads)

    np.testing.assert_allclose(out["bias_data"], np.median(first_reads, axis=0))
    np.testing.assert_allclose(out["kTC"], 20, rtol=0.15)
    assert out["bias_dq"].dtype == np.uint32
    assert not out["bias_dq"].any()


def test_make_bias_clips_outliers():
    rng = np.random.default_rng(1)
    first_reads = 1000 + rng.normal(0, 10, (50, 2, 2))
    first_reads[0, 0, 0] += 10_000  # a single wild first read
    out = make_bias(first_reads)
    np.testing.assert_allclose(out["kTC"][0, 0], np.std(first_reads[1:, 0, 0]), rtol=0.05)


def test_make_bias_flags():
    first_reads = np.full((20, 2, 3), 500.0)
    first_reads[:, 0, 1] = 5000.0  # high bias
    first_reads[::2, 1, 2] += 400.0  # high kTC noise
    input_dq = np.zeros((2, 3), dtype=np.uint32)
    input_dq[1, 0] = 1
    out = make_bias(first_reads, input_dq=input_dq, dq_flags=FLAGS)

    expected = np.zeros((2, 3), dtype=np.uint32)
    expected[1, 0] = 1
    expected[0, 1] |= FLAGS["UNRELIABLE_BIAS"]
    expected[1, 2] |= FLAGS["UNRELIABLE_BIAS"]
    np.testing.assert_array_equal(out["bias_dq"], expected)
    assert input_dq[0, 1] == 0  # input DQ not modified


def test_make_bias_without_flags_does_not_flag():
    first_reads = np.full((5, 1, 1), 5000.0)
    assert not make_bias(first_reads)["bias_dq"].any()
