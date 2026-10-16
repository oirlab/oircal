"""Tests for fitting the gain and read noise."""

import numpy as np
import pytest

from oircal.gain import fit_gain_and_readnoise
from oircal.simulations import create_ramp


GAIN, RN = 1.5, 5.0  # e-/DN, DN


@pytest.fixture(scope="module")
def ramps():
    Nreads, readtime = 20, 1.7
    data = np.array(
        [
            create_ramp(
                source=np.full((2, 3), 200.0),
                readtime=readtime,
                Nreads=Nreads,
                gain=GAIN,
                bias=1000.0 * GAIN,
                read_noise=RN * GAIN,
                kTC_noise=0,
                convert_to_uint16=False,
                seed=seed,
            )["data"]
            for seed in range(40)
        ],
    )
    return np.arange(Nreads) * readtime, data


@pytest.mark.parametrize("method", ["quad", "grid", "both"])
def test_fit_gain_and_readnoise_recovers_truth(ramps, method):
    times, data = ramps
    gain, rn, cov = fit_gain_and_readnoise(times, data, 1.4, 4.5, method=method)
    assert gain.shape == rn.shape == (2, 3)
    assert cov.shape == (2, 2, 2, 3)
    # With "grid", cov holds the errors; otherwise the variances
    if method == "grid":
        gain_err, rn_err = cov[0, 0], cov[1, 1]
    else:
        gain_err, rn_err = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
    assert np.all(np.abs(gain - GAIN) < 4 * gain_err)
    assert np.all(np.abs(rn - RN) < 4 * rn_err)
    # Averaged over pixels, the fit is much closer than a single pixel's error
    assert abs(gain.mean() - GAIN) < 0.1
    assert abs(rn.mean() - RN) < 1.0


def test_fit_read_noise_only(ramps):
    times, data = ramps
    for method in ("quad", "both"):
        gain, rn, cov = fit_gain_and_readnoise(
            times,
            data,
            GAIN,
            4.5,
            fit_gain=False,
            method=method,
        )
        np.testing.assert_allclose(gain, GAIN)
        assert np.isnan(cov[0, 0]).all()
        assert abs(rn.mean() - RN) < 1.0


def test_fit_gain_only(ramps):
    times, data = ramps
    gain, rn, cov = fit_gain_and_readnoise(times, data, 1.4, RN, fit_rn=False)
    np.testing.assert_allclose(rn, RN)
    assert np.isnan(cov[1, 1]).all()
    assert abs(gain.mean() - GAIN) < 0.1


def test_full_detector(ramps):
    times, data = ramps
    gain, rn, cov = fit_gain_and_readnoise(times, data, 1.4, 4.5, mode="full_detector")
    assert isinstance(gain, float)
    assert cov.shape == (2, 2)
    assert abs(gain - GAIN) < 4 * np.sqrt(cov[0, 0])
    assert abs(rn - RN) < 4 * np.sqrt(cov[1, 1])


def test_masked_reads_are_left_out(ramps, dq_flags):
    times, data = ramps
    data = data.copy()
    dq_raw = np.zeros(data.shape, dtype=np.uint8)
    data[:, 10:, 0, 0] += 5000.0  # a jump in every ramp of one pixel
    dq_raw[:, 10, 0, 0] = dq_flags["JUMP_DET"]
    masked, _, _ = fit_gain_and_readnoise(times, data, 1.4, 4.5, dq_raw, dq_flags)
    unmasked, _, _ = fit_gain_and_readnoise(times, data, 1.4, 4.5)
    assert abs(masked[0, 0] - GAIN) < 0.5
    # Unmasked, the jump ruins the fit (NaN or far off)
    assert not abs(unmasked[0, 0] - GAIN) < 0.5


def test_parallel_matches_serial(ramps):
    times, data = ramps
    serial = fit_gain_and_readnoise(times, data, 1.4, 4.5)
    parallel = fit_gain_and_readnoise(times, data, 1.4, 4.5, max_cores=2)
    for a, b in zip(serial, parallel):
        np.testing.assert_allclose(a, b, equal_nan=True)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"fit_gain": False, "fit_rn": False}, "At least one"),
        ({"method": "newton"}, "method"),
        ({"mode": "channel"}, "mode"),
        ({"delta_gains": np.linspace(-1, 1, 5)}, "3 elements"),
        ({"method": "grid", "delta_rns": np.array([-1.0, 0.0, 1.0])}, "more than 3"),
        ({"method": "grid", "fit_gain": False}, "fit_gain=True"),
    ],
)
def test_invalid_arguments(ramps, kwargs, match):
    times, data = ramps
    with pytest.raises(ValueError, match=match):
        fit_gain_and_readnoise(times, data, 1.4, 4.5, **kwargs)
