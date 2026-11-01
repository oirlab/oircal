"""Tests for calculating the gain from the mean and variance of read differences."""

import numpy as np

from oircal.gain import calc_gain


GAIN, RN = 1.5, 5.0  # e-/DN, DN
Nfiles, Nreads, Nrows, Ncols = 20, 10, 64, 64


def _ramps(rates, seed=0):
    """Ramps (DN) with read noise, a per-ramp offset and Poisson counts, rates in e- per read."""
    rng = np.random.default_rng(seed)
    shape = (len(rates), Nreads, Nrows, Ncols)
    mean_counts = np.asarray(rates, dtype=float)[:, None, None, None]
    increments = rng.poisson(mean_counts, (len(rates), Nreads - 1, Nrows, Ncols))
    signal = np.zeros(shape)
    signal[:, 1:] = np.cumsum(increments, axis=1) / GAIN
    offsets = rng.normal(1000.0, 300.0, (len(rates), 1, Nrows, Ncols))
    return offsets + signal + rng.normal(0.0, RN, shape)


def test_recovers_gain():
    # The shot noise (about 44 DN^2) is comparable to 2 RN^2, so the read noise term matters
    data = _ramps(np.full(Nfiles, 100.0))
    gain, gain_err = calc_gain(data, RN)
    assert gain.shape == gain_err.shape == (Nrows, Ncols)
    assert gain.dtype == gain_err.dtype == np.float32
    gain_detector, gain_err_detector = calc_gain(data, RN, mode="full_detector")
    assert abs(gain_detector / GAIN - 1) < 0.02
    assert abs(gain_detector - GAIN) < 3 * gain_err_detector
    assert abs(np.nanmedian(gain) / GAIN - 1) < 0.05

    # About 68% of the pixels are within their error of the true gain
    assert 0.6 < np.mean(np.abs(gain - GAIN) < gain_err) < 0.76


def test_rates_and_flags(dq_flags):
    # Ramps of different count rates, with flagged reads and jumps
    rng = np.random.default_rng(1)
    data = _ramps(np.linspace(50.0, 2000.0, Nfiles))
    dq_raw = np.zeros(data.shape, dtype=np.uint8)
    dq_raw[rng.random(data.shape) < 0.05] = dq_flags["DO_NOT_USE"]
    jumps = rng.random(data.shape) < 0.05
    dq_raw[jumps] |= dq_flags["JUMP_DET"]
    data += np.cumsum(jumps * rng.normal(0.0, 500.0, data.shape), axis=1)
    data[dq_raw & dq_flags["DO_NOT_USE"] > 0] = 1e6
    #gain = calc_gain(data, RN, dq_raw, dq_flags, mode="full_detector")
    gain, gain_err = calc_gain(data, RN, dq_raw, dq_flags, mode="per_pixel")
    assert abs(gain / GAIN - 1) < 0.02

    # A pixel with no run of 2 usable differences is NaN
    dq_raw[:, 1::2, 0, 0] = dq_flags["DO_NOT_USE"]
    gain, gain_err = calc_gain(data, RN, dq_raw, dq_flags)
    assert np.isnan(gain[0, 0])
    assert np.isnan(gain_err[0, 0])


def test_full_detector_clips_outliers():
    # Extra noise in a few pixels lowers their gain far below the rest; they are clipped
    data = _ramps(np.full(Nfiles, 100.0))
    gain, gain_err = calc_gain(data, RN, mode="full_detector")
    data[:, :, 0, :8] += np.random.default_rng(2).normal(0.0, 50.0, (Nfiles, Nreads, 8))
    gain_bad, gain_err_bad = calc_gain(data, RN, mode="full_detector")
    assert abs(gain_bad - gain) < 0.5 * gain_err
    gain_bad, _ = calc_gain(data, RN, mode="full_detector", n_sigma=np.inf)
    assert abs(gain_bad - gain) > 3 * gain_err
