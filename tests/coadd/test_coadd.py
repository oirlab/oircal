"""Tests for coadding stacks of frames and spectra.

The statistical test is ported from hispecdrp's ``test_coadd_frames_primitive``.
"""

import numpy as np
import pytest

from oircal.coadd import coadd


def _make_frames(Nframes=5, shape=(64, 64), rate=1.0, sigma_rnoise=0.1, sigma_poisson=0.1):
    rng = np.random.default_rng(1)
    sigma = np.sqrt(sigma_rnoise**2 + sigma_poisson**2)
    cube = (Nframes, *shape)
    data = rng.normal(loc=rate, scale=sigma, size=cube).astype(np.float32)
    data[data < 0] = 1.0
    return {
        "input_data": data,
        "input_err": np.full(cube, sigma, dtype=np.float32),
        "input_dq": np.zeros(cube, dtype=np.uint32),
        "input_var_poisson": np.full(cube, sigma_poisson**2, dtype=np.float32),
        "input_var_rnoise": np.full(cube, sigma_rnoise**2, dtype=np.float32),
        "input_mid_time": np.ones(cube, dtype=np.float64),
    }


@pytest.mark.parametrize("do_sigma_clip", [False, True])
def test_coadd_frames_statistics(do_sigma_clip, dq_flags):
    frames = _make_frames()
    out = coadd(
        **frames,
        dq_flags=dq_flags,
        do_sigma_clip=do_sigma_clip,
        sigma_thresh_low=4.0,
        sigma_thresh_high=4.0,
    )
    rel_residuals = np.abs(out["data"] - 1.0) / out["err"]
    assert np.sum(rel_residuals > 5) == 0
    assert abs(np.mean(rel_residuals < 1) - 0.683) < 0.1
    if not do_sigma_clip:  # clipping drops frames for some pixels
        np.testing.assert_allclose(out["var_poisson"], 0.1**2 / 5, rtol=1e-5)
        np.testing.assert_allclose(out["var_rnoise"], 0.1**2 / 5, rtol=1e-5)
    np.testing.assert_allclose(out["mid_time"], 1.0)
    assert np.all(out["dq"] == 0)


def test_coadd_frames_weighted_mean(dq_flags):
    data = np.array([1.0, 2.0, 4.0], dtype=np.float32).reshape(3, 1, 1)
    err = np.array([1.0, 1.0, 2.0], dtype=np.float32).reshape(3, 1, 1)
    out = coadd(data, input_err=err, dq_flags=dq_flags)
    w = 1 / err.ravel() ** 2
    np.testing.assert_allclose(out["data"][0, 0], np.sum(w * data.ravel()) / np.sum(w))
    np.testing.assert_allclose(out["err"][0, 0], 1 / np.sqrt(np.sum(w)))


def test_coadd_frames_bad_frames_excluded(dq_flags):
    data = np.array([1.0, 2.0, 100.0, np.nan], dtype=np.float32).reshape(4, 1, 1)
    dq = np.array([0, 0, 4, 0], dtype=np.uint32).reshape(4, 1, 1)
    dq_in = dq.copy()
    out = coadd(data, input_dq=dq, dq_flags=dq_flags, method="mean")
    # The flagged frame and the NaN frame are excluded
    np.testing.assert_allclose(out["data"][0, 0], 1.5)
    assert out["err"] is None
    assert out["dq"][0, 0] == 0
    np.testing.assert_array_equal(dq, dq_in)  # inputs are not modified


def test_coadd_frames_one_good_frame(dq_flags):
    data = np.array([1.0, 7.0], dtype=np.float32).reshape(2, 1, 1)
    dq = np.array([4, 0], dtype=np.uint32).reshape(2, 1, 1)
    err = np.array([0.1, 0.5], dtype=np.float32).reshape(2, 1, 1)
    out = coadd(data, input_err=err, input_dq=dq, dq_flags=dq_flags)
    assert out["data"][0, 0] == 7.0 and out["err"][0, 0] == np.float32(0.5)


def test_coadd_frames_no_good_frames(dq_flags):
    data = np.ones((3, 1, 1), dtype=np.float32)
    dq = np.full((3, 1, 1), 4, dtype=np.uint32)
    out = coadd(data, input_err=np.ones_like(data), input_dq=dq, dq_flags=dq_flags)
    assert np.isnan(out["data"][0, 0]) and np.isnan(out["err"][0, 0])
    assert out["dq"][0, 0] == dq_flags["DO_NOT_USE"]


def test_coadd_frames_single_frame():
    data = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
    dq = np.array([[[0, 4], [0, 0]]], dtype=np.uint32)
    out = coadd(data, input_dq=dq)
    # Returned unchanged, including DQ-flagged pixels
    np.testing.assert_array_equal(out["data"], data[0])
    np.testing.assert_array_equal(out["dq"], dq[0])


def test_coadd_frames_sigma_clip_rejects_outlier(dq_flags):
    data = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 50.0], dtype=np.float32).reshape(6, 1, 1)
    out = coadd(data, dq_flags=dq_flags, method="mean", do_sigma_clip=True)
    np.testing.assert_allclose(out["data"][0, 0], 1.0, rtol=1e-6)
    out = coadd(data, dq_flags=dq_flags, method="mean")
    assert out["data"][0, 0] > 5


def test_coadd_frames_mean_and_median(dq_flags):
    data = np.array([1.0, 2.0, 6.0], dtype=np.float32).reshape(3, 1, 1)
    err = np.array([1.0, 2.0, 2.0], dtype=np.float32).reshape(3, 1, 1)
    var = (err**2).astype(np.float32)
    kwargs = {"input_err": err, "input_var_poisson": var, "dq_flags": dq_flags}

    out = coadd(data, method="mean", **kwargs)
    np.testing.assert_allclose(out["data"][0, 0], 3.0)
    np.testing.assert_allclose(out["err"][0, 0], 3.0 / 3)  # sqrt(1 + 4 + 4) / 3
    np.testing.assert_allclose(out["var_poisson"][0, 0], 9.0 / 9)

    out = coadd(data, method="median", **kwargs)
    np.testing.assert_allclose(out["data"][0, 0], 2.0)
    np.testing.assert_allclose(out["err"][0, 0], np.sqrt(np.pi / 2), rtol=1e-6)
    np.testing.assert_allclose(out["var_poisson"][0, 0], np.pi / 2, rtol=1e-6)


def test_coadd_frames_median_resists_outlier():
    data = np.array([1.0, 1.1, 0.9, 100.0], dtype=np.float32).reshape(4, 1, 1)
    out = coadd(data, method="median")
    np.testing.assert_allclose(out["data"][0, 0], 1.05, rtol=1e-6)
    assert out["err"] is None


def test_coadd_frames_invalid_method():
    data = np.ones((3, 1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="requires input_err"):
        coadd(data)  # 'wmean' needs errors
    with pytest.raises(ValueError, match="method must be one of"):
        coadd(data, method="mode")


def test_coadd_frames_scatter_error():
    data = np.array([1.0, 2.0, 6.0], dtype=np.float32).reshape(3, 1, 1)
    out = coadd(data, method="mean", err_method="scatter")
    np.testing.assert_allclose(out["err"][0, 0], np.std([1, 2, 6], ddof=1) / np.sqrt(3), rtol=1e-6)


@pytest.mark.parametrize("shape", [(6, 50), (6, 3, 50)])
def test_coadd_spectra(shape, dq_flags):
    # Spectra of shape (Nspectra, Npix) or (Nspectra, Norders, Npix)
    data = np.full(shape, 100.0, dtype=np.float32)
    dq = np.zeros(shape, dtype=np.uint32)
    dq[0, ..., 10] = 4  # one spectrum flagged at pixel 10
    data[1, ..., 20] = 1e4  # a cosmic ray in another spectrum
    var = np.ones(shape, dtype=np.float32)
    out = coadd(
        data,
        input_err=np.sqrt(var),
        input_dq=dq,
        input_var_poisson=var,
        input_mid_time=np.arange(shape[0], dtype=np.float64)[:, None]
        .repeat(np.prod(shape[1:]), axis=1)
        .reshape(shape),
        dq_flags=dq_flags,
        do_sigma_clip=True,
    )
    for key in ("data", "err", "var_poisson", "dq", "mid_time"):
        assert out[key].shape == shape[1:], key
    np.testing.assert_allclose(out["data"], 100.0)  # cosmic ray clipped
    np.testing.assert_allclose(out["var_poisson"][..., 0], 1 / 6)
    np.testing.assert_allclose(out["var_poisson"][..., 10], 1 / 5)  # flagged spectrum excluded
    np.testing.assert_allclose(out["mid_time"][..., 10], np.mean(np.arange(1, 6)))
    assert np.all(out["dq"] == 0)


def test_coadd_flags_empty_without_input_dq(dq_flags):
    data = np.full((3, 2), np.nan, dtype=np.float32)
    data[:, 0] = 1.0
    out = coadd(data, method="mean", dq_flags=dq_flags)
    np.testing.assert_array_equal(out["dq"], [0, dq_flags["DO_NOT_USE"]])
