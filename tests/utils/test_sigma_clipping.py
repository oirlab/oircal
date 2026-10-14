"""Tests for sigma clipping."""

import numpy as np

from oircal.utils.sigma_clipping import sigma_clip, sigma_clip_stack


def test_sigma_clip_removes_outliers():
    x = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 20.0, -15.0])
    mask = np.ones(x.size, dtype=np.bool_)
    mask, center, stddev, n_iters = sigma_clip(x, mask, 5.0, 5.0)
    np.testing.assert_array_equal(mask, [True] * 7 + [False, False])
    assert abs(center - 1.0) < 0.05 and stddev < 0.2 and n_iters >= 1


def test_sigma_clip_one_sided():
    x = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 20.0, -15.0])
    mask = np.ones(x.size, dtype=np.bool_)
    mask = sigma_clip(x, mask, None, 5.0)[0]
    assert not mask[7] and mask[8]  # only the high outlier is clipped


def test_sigma_clip_min_keep():
    x = np.array([1.0, 1.1, 0.9, 1.05, 30.0, 40.0, 50.0])
    mask = np.ones(x.size, dtype=np.bool_)
    mask = sigma_clip(x, mask, 3.0, 3.0, min_keep=6)[0]
    # Only one value can be clipped, and it is the most extreme
    np.testing.assert_array_equal(mask, [True] * 6 + [False])


def test_sigma_clip_stack():
    data = np.tile(np.linspace(0.9, 1.1, 10)[:, None, None], (1, 4, 4))
    data[3, 2, 1] = 50.0
    data[5, 0, 0] = np.nan
    mask = sigma_clip_stack(data)
    expected = np.ones(data.shape, dtype=np.bool_)
    expected[3, 2, 1] = False
    expected[5, 0, 0] = False  # not finite, so never good
    np.testing.assert_array_equal(mask, expected)


def test_sigma_clip_stack_1d():
    # Any stack shape works, e.g. RVs of shape (Nobs,)
    data = np.append(np.linspace(0.9, 1.1, 10), 25.0)
    mask = sigma_clip_stack(data)
    np.testing.assert_array_equal(mask, [True] * 10 + [False])
