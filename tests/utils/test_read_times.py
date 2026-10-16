"""Tests for detector read time helpers."""

import numpy as np
import pytest

from oircal.ramp_fitting import fit_ramps_ols
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

    # Each 3 x 2 channel is read row by row, all channels in parallel, the second channel
    # right to left
    ticks = np.array([[0, 1, 1, 0], [2, 3, 3, 2], [4, 5, 5, 4]])
    np.testing.assert_allclose((first - mjd_start) * 86400.0 * clock_rate, ticks, atol=1e-6)


@pytest.mark.parametrize(
    ("readout_direction", "ticks_row_0"),
    [
        ("forward", [0, 1, 0, 1, 0, 1]),
        ("reverse", [1, 0, 1, 0, 1, 0]),
        ("forward_reverse", [0, 1, 1, 0, 0, 1]),
        ("reverse_forward", [1, 0, 0, 1, 1, 0]),
    ],
)
def test_first_read_mjd_readout_direction(readout_direction, ticks_row_0):
    clock_rate = 2.0
    first = get_first_read_mjd(
        0.0,
        2,
        6,
        clock_rate,
        Nchannels=3,
        readout_direction=readout_direction,
    )
    ticks = first * 86400.0 * clock_rate
    np.testing.assert_allclose(ticks[0], ticks_row_0, atol=1e-6)
    # The second row continues where the first left off
    np.testing.assert_allclose(ticks[1], np.add(ticks_row_0, 2), atol=1e-6)


def test_first_read_mjd_readout_direction_horizontal():
    # Two horizontal channels of 2 rows x 3 columns; forward is toward increasing row
    clock_rate = 1.0
    first = get_first_read_mjd(0.0, 4, 3, clock_rate, Nchannels=2, vertical=False)
    ticks = np.array([[0, 2, 4], [1, 3, 5], [1, 3, 5], [0, 2, 4]])
    np.testing.assert_allclose(first * 86400.0 * clock_rate, ticks, atol=1e-6)


def test_first_read_mjd_invalid_readout_direction():
    with pytest.raises(ValueError, match="readout_direction"):
        get_first_read_mjd(0.0, 2, 4, 1.0, Nchannels=2, readout_direction="left")


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


def test_first_pixel_starts_at_mjd_start():
    mjd_start = 61209.79547434599
    first = get_first_read_mjd(mjd_start, 8, 16, clock_rate=1.5e5, Nchannels=4)
    # The first pixel of every channel is read at the start: the left end of the first row in
    # odd channels, the right end in even channels
    assert np.all(first[0, 0::8] == mjd_start)
    assert np.all(first[0, 7::8] == mjd_start)


def test_first_read_mjd_single_channel_row_major():
    # With one channel, pixels are read in row-major order, one per clock tick
    clock_rate = 4.0
    first = get_first_read_mjd(0.0, 3, 5, clock_rate)
    ticks = np.arange(15).reshape(3, 5)
    np.testing.assert_allclose(first * 86400.0 * clock_rate, ticks, atol=1e-9)


def test_channels_read_in_parallel():
    # Every vertical channel has the same timing, mirrored in every other channel
    first = get_first_read_mjd(61000.0, 6, 16, clock_rate=3.0, Nchannels=4)
    channel_1 = first[:, :4]
    for k in range(1, 4):
        expected = channel_1[:, ::-1] if k % 2 else channel_1
        np.testing.assert_array_equal(first[:, 4 * k : 4 * (k + 1)], expected)


def test_read_period_matches_channel_readout():
    # One read lasts as long as reading every pixel in a channel, so the last pixel of a read
    # is one clock tick before the next read starts
    Nrows, Ncols, clock_rate, Nchannels = 8, 16, 32.0, 4
    read_times = get_delta_read_times(3, Nrows, Ncols, clock_rate, Nchannels)
    first = get_first_read_mjd(0.0, Nrows, Ncols, clock_rate, Nchannels)
    last_pixel_s = first.max() * 86400.0
    np.testing.assert_allclose(read_times[1], last_pixel_s + 1 / clock_rate, rtol=1e-6)


def test_single_read():
    read_times = get_delta_read_times(1, 4, 4, clock_rate=1.0)
    np.testing.assert_array_equal(read_times, [0.0])


def test_read_times_drive_ramp_fit_mid_time():
    Nreads, Nrows, Ncols = 6, 4, 8
    mjd_start, clock_rate = 61000.0, 16.0
    times = get_delta_read_times(Nreads, Nrows, Ncols, clock_rate, Nchannels=2)
    start_times = get_first_read_mjd(mjd_start, Nrows, Ncols, clock_rate, Nchannels=2)
    rate = 5.0
    ramps = (rate * times[:, None, None] * np.ones((Nrows, Ncols))).astype(np.float32)

    out = fit_ramps_ols(times, ramps, start_times=start_times)
    np.testing.assert_allclose(out["rate"], rate, rtol=1e-5)
    # The midpoint of each pixel's ramp is its own first-read time plus the mean read offset
    expected_mid = start_times + np.mean(times) / 86400.0
    np.testing.assert_allclose(out["mid_time"], expected_mid, rtol=0, atol=1e-9)
