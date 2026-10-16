"""Tests for the posterior error helper."""

import numpy as np

from oircal.utils.stats import get_err_from_posterior


def test_get_err_from_posterior_gaussian():
    x = np.linspace(-5, 5, 2001)
    posterior = np.exp(-0.5 * ((x - 1) / 0.5) ** 2)
    mode, left_err, right_err = get_err_from_posterior(x, posterior)
    assert mode == x[np.argmax(posterior)]
    np.testing.assert_allclose([left_err, right_err], 0.5, rtol=0.02)


def test_get_err_from_posterior_asymmetric():
    x = np.linspace(0, 10, 4001)
    posterior = np.where(
        x < 2,
        np.exp(-0.5 * ((x - 2) / 0.5) ** 2),
        np.exp(-0.5 * ((x - 2) / 2.0) ** 2),
    )
    _, left_err, right_err = get_err_from_posterior(x, posterior)
    assert left_err < right_err
