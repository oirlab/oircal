"""Tests for combining a stack of measurements along its first axis."""

import numpy as np
import pytest

from oircal.coadd import combine


x = np.array([1.0, 2.0, 6.0])
err = np.array([1.0, 2.0, 2.0])


def test_combine_se():
    w = 1 / err**2
    value, error, _ = combine(x, input_err=err, method="wmean", err_method="se")
    np.testing.assert_allclose(value, np.sum(w * x) / np.sum(w))
    np.testing.assert_allclose(error, 1 / np.sqrt(np.sum(w)))

    value, error, _ = combine(x, input_err=err, method="mean", err_method="se")
    np.testing.assert_allclose([value, error], [3.0, 1.0])  # sqrt(1 + 4 + 4) / 3

    value, error, _ = combine(x, input_err=err, method="median", err_method="se")
    np.testing.assert_allclose([value, error], [2.0, np.sqrt(np.pi / 2)])


def test_combine_scatter():
    w = 1 / err**2
    value, error, _ = combine(x, input_err=err, method="wmean", err_method="scatter")
    expected = np.sqrt(np.sum(w * (x - value) ** 2) / ((x.size - 1) * np.sum(w)))
    np.testing.assert_allclose(error, expected)

    # No input errors needed to measure the scatter of a mean or median
    sem = np.std(x, ddof=1) / np.sqrt(x.size)
    value, error, _ = combine(x, method="mean", err_method="scatter")
    np.testing.assert_allclose([value, error], [3.0, sem])
    value, error, _ = combine(x, method="median", err_method="scatter")
    np.testing.assert_allclose([value, error], [2.0, sem * np.sqrt(np.pi / 2)])


def test_combine_scatter_with_equal_weights_matches_mean():
    v1, e1, _ = combine(x, input_err=np.ones_like(x), method="wmean", err_method="scatter")
    v2, e2, _ = combine(x, method="mean", err_method="scatter")
    np.testing.assert_allclose([v1, e1], [v2, e2])


def test_combine_rvs_per_order():
    # e.g. RVs of shape (Nobs, Norders), combined over observations for each order
    rng = np.random.default_rng(0)
    rvs = rng.normal(10.0, 1.0, size=(20, 3))
    rvs[4, 1] = np.nan
    value, error, mask = combine(rvs, input_err=np.ones_like(rvs))
    assert value.shape == error.shape == (3,)
    np.testing.assert_allclose(value[0], rvs[:, 0].mean())
    assert not mask[4, 1] and mask.sum() == rvs.size - 1


def test_combine_mask_and_sigma_clip():
    values = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 50.0, 2.0])
    good = np.array([True] * 6 + [False])
    value, error, mask = combine(values, mask=good, method="mean", do_sigma_clip=True)
    np.testing.assert_allclose(value, 1.0)
    assert error is None  # 'se' without input_err
    np.testing.assert_array_equal(mask, [True] * 5 + [False, False])


def test_combine_one_and_no_good_values():
    value, error, _ = combine(np.array([4.0, np.nan]), input_err=np.array([0.5, 1.0]))
    assert value == 4.0 and error == 0.5
    value, error, _ = combine(np.array([np.nan, np.nan]), input_err=np.ones(2))
    assert np.isnan(value) and np.isnan(error)


def test_combine_invalid_arguments():
    with pytest.raises(ValueError, match="requires input_err"):
        combine(x)
    with pytest.raises(ValueError, match="method must be one of"):
        combine(x, input_err=err, method="mode")
    with pytest.raises(ValueError, match="err_method must be one of"):
        combine(x, input_err=err, err_method="std")
