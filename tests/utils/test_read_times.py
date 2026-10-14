"""Tests for detector read time helpers."""

import numpy as np
import pytest

from oircal.utils.read_times import get_delta_read_times, get_first_read_mjd


def test_delta_read_times():
    # 4 vertical channels of 8 x 4 pixels, read at 16 Hz -> 2 s per read
    read_times = get_delta_read_times(5, 8, 16, clock_rate=16.0, Nchannels=4)
    assert read_times.dtype == np.float32
    np.testing.assert_allclose(read_times, [0, 2, 4, 6, 8])

    # Same pixels per channel for horizontal channels
    read_times_h = get_delta_read_times(5, 16, 8, clock_rate=16.0, Nchannels=4, vertical=False)
    np.testing.assert_allclose(read_times_h, read_times)


def test_first_read_mjd_vertical():
    mjd_start = 61000.0
    clock_rate = 2.0
    first = get_first_read_mjd(mjd_start, 3, 4, clock_rate, Nchannels=2)
    assert first.shape == (3, 4)
    assert first.dtype == np.float64

    # Each 3 x 2 channel is read row by row, all channels in parallel
    ticks = np.array([[0, 1, 0, 1], [2, 3, 2, 3], [4, 5, 4, 5]])
    np.testing.assert_allclose((first - mjd_start) * 86400.0 * clock_rate, ticks, atol=1e-6)


def test_first_read_mjd_horizontal():
    mjd_start = 61000.0
    clock_rate = 2.0
    first = get_first_read_mjd(mjd_start, 4, 3, clock_rate, Nchannels=2, vertical=False)
    assert first.shape == (4, 3)

    # Horizontal channels are the transpose of vertical ones
    first_v = get_first_read_mjd(mjd_start, 3, 4, clock_rate, Nchannels=2)
    np.testing.assert_array_equal(first, first_v.T)


def test_first_read_mjd_readout_offset_is_sub_day():
    mjd_start = 61000.0
    first = get_first_read_mjd(mjd_start, 4096, 4160, clock_rate=10023152.94117647)
    max_offset_days = float(np.max(first) - mjd_start)
    assert max_offset_days < 10.0 / 86400.0


@pytest.mark.parametrize("func", [get_delta_read_times, get_first_read_mjd])
@pytest.mark.parametrize("vertical", [True, False])
def test_incompatible_channels(func, vertical):
    # 5 columns (vertical) or 5 rows (horizontal) don't split into 2 channels
    Nrows, Ncols = (4, 5) if vertical else (5, 4)
    with pytest.raises(ValueError, match="incompatible"):
        func(3, Nrows, Ncols, 1.0, Nchannels=2, vertical=vertical)
