"""Tests for making the saturation calibration."""

import numpy as np

from oircal.saturation import check_saturation, make_saturation


def test_make_saturation_median_minus_offset():
    last_reads = np.array([[[60000.0, 50000.0]], [[60010.0, 50020.0]], [[59990.0, 49980.0]]])
    sat_thresh = make_saturation(last_reads, offset=100)
    np.testing.assert_allclose(sat_thresh, [[59900.0, 49900.0]])


def test_make_saturation_feeds_check_saturation(dq_flags):
    ramps = np.array([[[[100.0]], [[30000.0]], [[60000.0]], [[60000.0]]]] * 3)  # (3, 4, 1, 1)
    sat_thresh = make_saturation(ramps[:, -1])
    data = ramps[0].copy()
    dq_raw = np.zeros(data.shape, dtype=np.uint8)
    dq = np.zeros((1, 1), dtype=np.uint32)
    check_saturation(data, dq_raw, dq, sat_thresh, np.zeros((1, 1), np.uint32), dq_flags)
    assert list(dq_raw[:, 0, 0] & dq_flags["SATURATED"]) == [0, 0, 2, 2]
