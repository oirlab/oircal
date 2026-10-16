"""Read noise and Poisson variances of the fitted rates, in each noise regime.

``rate_var_rnoise`` and ``rate_var_poisson`` are checked against the closed-form OLS and MCDS
formulas, and against ``rate_err``: for a single ramp segment, the two components sum to
``rate_err**2``.
"""

import numpy as np
import pytest

from oircal.ramp_fitting import fit_ramps_jwst, fit_ramps_mcds, fit_ramps_ols
from oircal.simulations import create_ramp


Nrows, Ncols = 6, 6
Nreads = 30
readtime = 1.7  # s
gain = 3.0  # e-/DN
read_noise = 8.5  # e-, single read
rdnoise = read_noise / gain  # DN
dark_current = 0.1  # DN/s
jump_read = 12  # first read of the second segment in the jump tests

# Source rates (e-/s) from read noise dominated to photon dominated
REGIMES = {
    "read_noise_dominated": 0.0,
    "comparable": 0.5,
    "mostly_poisson": 5.0,
    "photon_dominated": 2000.0,
}

ALGORITHMS = {
    "ols": (fit_ramps_ols, {}),
    "cds": (fit_ramps_mcds, {"num_coadd": 1}),
    "mcds": (fit_ramps_mcds, {"num_coadd": 3}),
    "jwst_fixsen": (fit_ramps_jwst, {"algorithm": "JWST_FIXSEN"}),
    "jwst_likely": (fit_ramps_jwst, {"algorithm": "JWST_LIKELY"}),
    "cython_likely": (fit_ramps_jwst, {"algorithm": "CYTHON_LIKELY"}),
}
JWST_ALGORITHMS = ["jwst_fixsen", "jwst_likely", "cython_likely"]


def _ramps(rate, jump):
    """Simulated ramps (DN) and per-read DQ, optionally with a flagged jump."""
    data = create_ramp(
        np.full((Nrows, Ncols), rate),
        readtime=readtime,
        Nreads=Nreads,
        gain=gain,
        bias=1000.0,
        read_noise=read_noise,
        kTC_noise=50.0,
        convert_to_uint16=False,
        seed=7,
    )["data"].astype(np.float32)
    dq_raw = np.zeros(data.shape, dtype=np.uint8)
    if jump:
        data[jump_read:] += 3000.0
        dq_raw[jump_read] = 4  # JUMP_DET in the conftest flags
    return np.arange(Nreads) * readtime, data, dq_raw


def _fit(algorithm, times, data, dq_raw, dq_flags):
    fit, kwargs = ALGORITHMS[algorithm]
    return fit(
        times,
        data.copy(),
        input_dq_raw=dq_raw.copy(),
        input_dq=np.zeros((Nrows, Ncols), dtype=np.uint32),
        dq_flags=dq_flags,
        rdnoise=np.full((Nrows, Ncols), rdnoise, dtype=np.float32),
        gain=np.full((Nrows, Ncols), gain, dtype=np.float32),
        **kwargs,
    )


def _ols_segment(t, S):
    """Rate and read noise / Poisson variances of an OLS fit to one segment."""
    n = len(t)
    delta = n * np.sum(t**2) - np.sum(t) ** 2
    rate = (n * np.sum(t * S) - np.sum(t) * np.sum(S)) / delta
    var_rnoise = n * rdnoise**2 / delta
    var_poisson = abs(rate + dark_current) / (gain * (t[-1] - t[0]))
    return var_rnoise, var_poisson


def _mcds_segment(t, S, m):
    """Rate and read noise / Poisson variances of MCDS on one segment."""
    dt = t[-m:].mean() - t[:m].mean()
    rate = (S[-m:].mean() - S[:m].mean()) / dt
    var_rnoise = 2 * rdnoise**2 / (m * dt**2)
    var_poisson = abs(rate + dark_current) / (gain * dt)
    return var_rnoise, var_poisson


@pytest.mark.parametrize("rate", list(REGIMES.values()), ids=list(REGIMES))
@pytest.mark.parametrize("algorithm", list(ALGORITHMS))
def test_variances_sum_to_err_single_segment(algorithm, rate, dq_flags):
    times, data, dq_raw = _ramps(rate, jump=False)
    out = _fit(algorithm, times, data, dq_raw, dq_flags)
    assert np.all(out["rate_var_rnoise"] > 0)
    assert np.all(out["rate_var_poisson"] >= 0)
    np.testing.assert_allclose(
        out["rate_var_rnoise"] + out["rate_var_poisson"],
        out["rate_err"] ** 2,
        rtol=1e-4,
    )


@pytest.mark.parametrize("rate", list(REGIMES.values()), ids=list(REGIMES))
@pytest.mark.parametrize("algorithm", JWST_ALGORITHMS)
def test_jwst_variances_sum_to_err_with_jump(algorithm, rate, dq_flags):
    times, data, dq_raw = _ramps(rate, jump=True)
    out = _fit(algorithm, times, data, dq_raw, dq_flags)
    np.testing.assert_allclose(
        out["rate_var_rnoise"] + out["rate_var_poisson"],
        out["rate_err"] ** 2,
        rtol=1e-4,
    )


@pytest.mark.parametrize("jump", [False, True], ids=["one_segment", "two_segments"])
@pytest.mark.parametrize("rate", list(REGIMES.values()), ids=list(REGIMES))
@pytest.mark.parametrize("algorithm", ["ols", "cds", "mcds"])
def test_ols_mcds_variances_match_formulas(algorithm, rate, jump, dq_flags):
    times, data, dq_raw = _ramps(rate, jump)
    fit, kwargs = ALGORITHMS[algorithm]
    out = fit(
        times,
        data.copy(),
        input_dq_raw=dq_raw.copy(),
        input_dq=np.zeros((Nrows, Ncols), dtype=np.uint32),
        dq_flags=dq_flags,
        rdnoise=np.full((Nrows, Ncols), rdnoise, dtype=np.float32),
        gain=np.full((Nrows, Ncols), gain, dtype=np.float32),
        dark_current=np.full((Nrows, Ncols), dark_current, dtype=np.float32),
        **kwargs,
    )
    segments = [slice(0, jump_read), slice(jump_read, Nreads)] if jump else [slice(0, Nreads)]

    for j in range(Nrows):
        for i in range(Ncols):
            per_segment = []
            for seg in segments:
                t, S = times[seg], data[seg, j, i].astype(np.float64)
                if algorithm == "ols":
                    per_segment.append(_ols_segment(t, S))
                else:
                    per_segment.append(_mcds_segment(t, S, kwargs["num_coadd"]))
            var_rnoise, var_poisson = np.array(per_segment).T

            # Each component combines like inverse variances; rate_err from the totals
            expected_rnoise = 1 / np.sum(1 / var_rnoise)
            expected_poisson = 1 / np.sum(1 / var_poisson)
            expected_err = np.sqrt(1 / np.sum(1 / (var_rnoise + var_poisson)))
            # Tolerance 1e-3 of the total variance (and of rate_err): OLS sums a segment's reads in
            # float32, which shifts the rate by ~1e-4 DN/s here, and near zero rate the Poisson
            # term |rate + dark_current| is sensitive to that
            atol = 1e-3 * expected_err**2
            np.testing.assert_allclose(
                out["rate_var_rnoise"][j, i],
                expected_rnoise,
                rtol=1e-4,
                atol=atol,
            )
            np.testing.assert_allclose(
                out["rate_var_poisson"][j, i],
                expected_poisson,
                rtol=1e-4,
                atol=atol,
            )
            np.testing.assert_allclose(out["rate_err"][j, i], expected_err, rtol=1e-3)
