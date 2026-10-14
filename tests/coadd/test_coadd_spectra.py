"""Tests for coadding spectra on a common wavelength grid."""

import numpy as np
import pytest

from oircal.coadd import coadd, coadd_spectra


@pytest.mark.parametrize("shape", [(5, 40), (5, 3, 40)])
def test_coadd_spectra_matches_coadd(shape, dq_flags):
    rng = np.random.default_rng(3)
    flux = rng.normal(100.0, 2.0, size=shape).astype(np.float32)
    flux_err = np.full(shape, 2.0, dtype=np.float32)
    dq = np.zeros(shape, dtype=np.uint32)
    dq[2, ..., 7] = 4
    kwargs = {"input_err": flux_err, "input_dq": dq, "dq_flags": dq_flags, "method": "median"}
    out = coadd_spectra(flux, **kwargs)
    expected = coadd(flux, **kwargs)
    assert out["data"].shape == shape[1:]
    for key in ("data", "err", "dq"):
        np.testing.assert_array_equal(out[key], expected[key])
    np.testing.assert_allclose(
        out["data"][..., 7],
        np.median(np.delete(flux, 2, axis=0)[..., 7], axis=0),
    )


@pytest.mark.parametrize("shape", [(5,), (5, 2, 3, 40)])
def test_coadd_spectra_rejects_other_shapes(shape):
    with pytest.raises(ValueError, match="Nspectra"):
        coadd_spectra(np.ones(shape, dtype=np.float32), method="mean")
