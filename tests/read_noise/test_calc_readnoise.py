"""Tests for calculating the read noise from read differences of darks."""

import numpy as np
import pytest

from oircal.read_noise import calc_readnoise
from oircal.simulations import create_ramp


GAIN, RN = 1.5, 5.0  # e-/DN, DN
Nfiles, Nreads, Nrows, Ncols = 20, 5, 16, 16


def _darks(times, dark=0.0, ktc=300.0, seed=0):
    """Dark ramps (DN) with read noise, a per-ramp offset and Poisson dark current (e-/s)."""
    rng = np.random.default_rng(seed)
    shape = (Nfiles, len(times), Nrows, Ncols)
    signal = np.zeros(shape)
    if dark > 0:
        mean_counts = dark * np.diff(times)[None, :, None, None]  # e- per read difference
        increments = rng.poisson(mean_counts, (Nfiles, len(times) - 1, Nrows, Ncols))
        signal[:, 1:] = np.cumsum(increments, axis=1) / GAIN
    offsets = rng.normal(1000.0, ktc, (Nfiles, 1, Nrows, Ncols))
    return offsets + signal + rng.normal(0.0, RN, shape)


@pytest.fixture(scope="module")
def times():
    return np.arange(Nreads) * 1.7


def test_recovers_read_noise(times):
    rn = calc_readnoise(_darks(times))
    assert rn.shape == (Nrows, Ncols)
    assert rn.dtype == np.float32
    assert np.isfinite(rn).all()
    assert abs(rn.mean() - RN) < 0.2


def test_insensitive_to_ramp_offsets(times):
    data = _darks(times)
    offsets = np.random.default_rng(1).normal(0.0, 1e4, (Nfiles, 1, Nrows, Ncols))
    np.testing.assert_allclose(
        calc_readnoise(data + offsets),
        calc_readnoise(data),
        rtol=1e-5,
    )


def test_create_ramp_with_ktc(times):
    data = np.array(
        [
            create_ramp(
                source=np.zeros((Nrows, Ncols)),
                readtime=times[1],
                Nreads=Nreads,
                gain=GAIN,
                bias=1000.0 * GAIN,
                read_noise=RN * GAIN,
                kTC_noise=500.0,
                convert_to_uint16=False,
                seed=seed,
            )["data"]
            for seed in range(Nfiles)
        ],
    )
    assert abs(np.nanmean(calc_readnoise(data)) - RN) < 0.2


def test_variance_unbiased(times, dq_flags):
    # Flagged reads and jumps. The mean of the per-pixel variance estimates over 4096 pixels has
    # a relative error of about 0.3%.
    rng = np.random.default_rng(5)
    data = np.concatenate([_darks(times, seed=seed) for seed in range(16)], axis=2)
    dq_raw = np.zeros(data.shape, dtype=np.uint8)
    dq_raw[rng.random(data.shape) < 0.1] = dq_flags["DO_NOT_USE"]
    jumps = rng.random(data.shape) < 0.1
    dq_raw[jumps] |= dq_flags["JUMP_DET"]
    data += np.cumsum(jumps * rng.normal(0.0, 500.0, data.shape), axis=1)
    data[dq_raw & dq_flags["DO_NOT_USE"] > 0] = 1e6
    rn = calc_readnoise(data, dq_raw, dq_flags, mode="full_detector")
    assert abs(rn**2 / RN**2 - 1) < 0.015


def test_dark_signal_biases_high(times):
    # A dark rate R (DN/s) adds (R dt)^2 to the mean of d_k^2, so sigma^2 is high by (R dt)^2 / 2
    rate = 2.0  # DN/s
    data = _darks(times) + rate * times[None, :, None, None]
    rn = calc_readnoise(data, mode="full_detector")
    expected = RN**2 + (rate * times[1]) ** 2 / 2
    assert abs(rn**2 / expected - 1) < 0.03


def test_flagged_reads_left_out(times, dq_flags):
    data = _darks(times)
    dq_raw = np.zeros(data.shape, dtype=np.uint8)
    dq_raw[:5, 0, 3, 4] = dq_flags["DO_NOT_USE"]
    dq_raw[:5, 2, 3, 4] = dq_flags["SATURATED"]
    flagged = data.copy()
    flagged[dq_raw != 0] = 1e6
    rn = calc_readnoise(flagged, dq_raw, dq_flags)

    # Same as leaving those reads out as NaN
    with_nans = data.copy()
    with_nans[dq_raw != 0] = np.nan
    np.testing.assert_allclose(rn, calc_readnoise(with_nans))

    # The flagged reads are skipped: ramps 0-4 of pixel (3, 4) use reads 1, 3 and 4
    d = np.concatenate(
        [np.diff(data[:5, [1, 3, 4], 3, 4], axis=1), np.diff(data[5:, :, 3, 4], axis=1)],
        axis=None,
    )
    np.testing.assert_allclose(rn[3, 4] ** 2, np.sum(d**2) / (2 * d.size), rtol=1e-5)

    # A JUMP_DET on the first read removes nothing, since no difference ends there
    dq_raw[:] = 0
    dq_raw[:, 0, 0, 0] = dq_flags["JUMP_DET"]
    rn = calc_readnoise(data, dq_raw, dq_flags)
    np.testing.assert_allclose(rn, calc_readnoise(data))


def test_every_other_read_bad(times, dq_flags):
    # With negligible dark signal, differences across a bad read still have variance 2 sigma^2
    data = np.concatenate([_darks(times, seed=seed) for seed in range(4)], axis=2)
    dq_raw = np.zeros(data.shape, dtype=np.uint8)
    dq_raw[:, 1::2] = dq_flags["DO_NOT_USE"]
    flagged = data.copy()
    flagged[:, 1::2] = 1e6
    rn = calc_readnoise(flagged, dq_raw, dq_flags)
    assert np.isfinite(rn).all()
    np.testing.assert_allclose(rn, calc_readnoise(data[:, ::2]), rtol=1e-6)
    assert abs(calc_readnoise(flagged, dq_raw, dq_flags, mode="full_detector") - RN) < 0.1


def test_jump_on_skipped_read(times, dq_flags):
    # A jump on an unusable read is inside the difference that skips it
    data = _darks(times)
    dq_raw = np.zeros(data.shape, dtype=np.uint8)
    dq_raw[:, 2, 0, 0] = dq_flags["DO_NOT_USE"] | dq_flags["JUMP_DET"]
    rn = calc_readnoise(data, dq_raw, dq_flags)
    d = np.diff(data[:, [0, 1], 0, 0], axis=1), np.diff(data[:, [3, 4], 0, 0], axis=1)
    d = np.concatenate(d, axis=None)
    np.testing.assert_allclose(rn[0, 0] ** 2, np.sum(d**2) / (2 * d.size), rtol=1e-5)


def test_no_differences_is_nan(times, dq_flags):
    data = _darks(times)
    dq_raw = np.zeros(data.shape, dtype=np.uint8)
    dq_raw[:, :, 0, 0] = dq_flags["DO_NOT_USE"]  # no differences
    dq_raw[1:, :, 0, 1] = dq_flags["DO_NOT_USE"]
    dq_raw[0, 2:, 0, 1] = dq_flags["DO_NOT_USE"]  # one difference
    rn = calc_readnoise(data, dq_raw, dq_flags)
    assert np.isnan(rn[0, 0])
    assert np.isfinite(rn.ravel()[1:]).all()


def test_single_ramp_and_integer_input():
    data = _darks(np.arange(10) * 1.7)[0]
    rn = calc_readnoise(data)
    np.testing.assert_allclose(rn, calc_readnoise(data[np.newaxis]))
    rounded = np.round(data)
    np.testing.assert_allclose(
        calc_readnoise(rounded.astype(np.uint16)),
        calc_readnoise(rounded),
    )


def test_max_cores(times):
    data = _darks(times)
    np.testing.assert_array_equal(
        calc_readnoise(data, max_cores=2),
        calc_readnoise(data),
    )


def test_full_detector(times, dq_flags):
    data = _darks(times)
    rn = calc_readnoise(data, mode="full_detector")
    assert isinstance(rn, float)
    assert abs(rn - RN) < 0.1

    # Pixels with no estimate are left out of the mean
    dq_raw = np.zeros(data.shape, dtype=np.uint8)
    dq_raw[:, :, 0, 0] = dq_flags["DO_NOT_USE"]
    data[:, :, 0, 0] = 1e6
    rn = calc_readnoise(data, dq_raw, dq_flags, mode="full_detector")
    assert abs(rn - RN) < 0.1

    # No pixels with an estimate
    dq_raw[:] = dq_flags["DO_NOT_USE"]
    rn = calc_readnoise(data, dq_raw, dq_flags, mode="full_detector")
    assert np.isnan(rn)


def test_bad_inputs(times):
    data = _darks(times)
    with pytest.raises(ValueError, match="3D or 4D"):
        calc_readnoise(data[0, 0])
    with pytest.raises(ValueError, match="mode"):
        calc_readnoise(data, mode="mean")
