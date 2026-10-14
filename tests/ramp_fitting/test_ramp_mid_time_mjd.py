"""Ramp mid-time must stay in MJD (days), not seconds added to MJD."""

import numpy as np
import pytest

from oircal.ramp_fitting import fit_ramps_jwst, fit_ramps_mcds, fit_ramps_ols
from oircal.ramp_fitting.utils import seconds_to_mjd_offset


def test_seconds_to_mjd_offset():
    assert seconds_to_mjd_offset(86400.0) == 1.0
    assert seconds_to_mjd_offset(0.0) == 0.0


@pytest.mark.parametrize("fit", [fit_ramps_ols, fit_ramps_mcds, fit_ramps_jwst])
def test_fit_mid_time_is_mjd(fit):
    # Ported from hispecdrp's test_ramp_exposure_mid_delta_is_sub_day
    mjd_start = 61209.79547434599
    Nreads, readtime = 10, 1.7  # s
    times = np.arange(Nreads) * readtime
    data = np.cumsum(np.ones((Nreads, 2, 2), dtype=np.float32), axis=0)
    out = fit(times, data, start_times=np.full((2, 2), mjd_start))
    expected = mjd_start + seconds_to_mjd_offset(np.mean(0.5 * (times[:-1] + times[1:])))
    np.testing.assert_allclose(out["mid_time"], expected, rtol=0, atol=1e-9)
    assert np.all(np.abs(out["mid_time"] - mjd_start) < seconds_to_mjd_offset(20.0))
