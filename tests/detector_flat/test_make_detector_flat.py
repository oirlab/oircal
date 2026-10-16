"""Tests for making the detector flat."""

import numpy as np

from oircal.detector_flat import correct_detector_flat, make_detector_flat


FLAGS = {"COLD": 2**14}


def test_make_detector_flat_normalizes_and_flags():
    rate = np.array([[100.0, 50.0, 102.0], [98.0, 100.0, 89.0]])
    err = np.full(rate.shape, 2.0)
    out = make_detector_flat(rate, input_err=err, dq_flags=FLAGS, cold_pixel_threshold=0.9)

    assert out["median_rate"] == 99.0
    np.testing.assert_allclose(out["detflat"], rate / 99.0)
    np.testing.assert_allclose(out["detflat_err"], err / 99.0)
    np.testing.assert_array_equal(out["detflat_dq"], [[0, 2**14, 0], [0, 0, 2**14]])


def test_make_detector_flat_corrects_its_own_flat():
    rate = np.array([[100.0, 50.0], [200.0, 100.0]], dtype=np.float32)
    flat = make_detector_flat(rate)
    out = {key: value for key, value in flat.items() if key != "median_rate"}
    corrected = rate.copy()
    correct_detector_flat(corrected, **out)
    np.testing.assert_allclose(corrected, flat["median_rate"])
