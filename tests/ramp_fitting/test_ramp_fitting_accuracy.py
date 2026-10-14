"""Accuracy and noise properties of each ramp fitting algorithm.

Ported from hispecdrp's ``test_ramp_fitting_algorithms`` (read noise and photon dominated
regimes), using arrays directly instead of the stage 0 pipeline.
"""

import numpy as np
import pytest

from oircal.ramp_fitting import fit_ramps_jwst, fit_ramps_mcds, fit_ramps_ols
from oircal.simulations import create_ramp


Nrows, Ncols = 10, 10
Nreads = 30
readtime = 1.7  # s
gain = 3.0  # e-/DN
read_noise = 8.5  # e-, single read
bias = 1000.0  # e-
kTC_noise = 50.0  # e-

ALGORITHMS = {
    "ols": (fit_ramps_ols, {}),
    "cds": (fit_ramps_mcds, {"num_coadd": 1}),
    "mcds": (fit_ramps_mcds, {"num_coadd": 3}),
    "jwst_fixsen": (fit_ramps_jwst, {"algorithm": "JWST_FIXSEN"}),
    "jwst_likely": (fit_ramps_jwst, {"algorithm": "JWST_LIKELY"}),
    "cython_likely": (fit_ramps_jwst, {"algorithm": "CYTHON_LIKELY"}),
}


@pytest.mark.parametrize("rate", [0.0, 2000.0], ids=["read_noise_dominated", "photon_dominated"])
@pytest.mark.parametrize("algorithm", list(ALGORITHMS))
def test_ramp_fitting_accuracy_and_noise(algorithm, rate, dq_flags):
    ramp = create_ramp(
        np.full((Nrows, Ncols), rate),  # e-/s
        readtime=readtime,
        Nreads=Nreads,
        gain=gain,
        bias=bias,
        read_noise=read_noise,
        kTC_noise=kTC_noise,
        convert_to_uint16=False,
        seed=12345,
    )
    data = ramp["data"].astype(np.float32)  # DN
    times = np.arange(Nreads) * readtime

    fit, kwargs = ALGORITHMS[algorithm]
    out = fit(
        times,
        data,
        dq_flags=dq_flags,
        rdnoise=np.full((Nrows, Ncols), read_noise / gain, dtype=np.float32),  # DN
        gain=np.full((Nrows, Ncols), gain, dtype=np.float32),
        **kwargs,
    )

    good = (out["dq"] & dq_flags["DO_NOT_USE"]) == 0
    assert good.all()
    pulls = (out["rate"][good] - rate / gain) / out["rate_err"][good]
    N = pulls.size

    # Accuracy: unbiased rate
    assert abs(np.mean(pulls)) < 3 / np.sqrt(N)
    # Noise: the reported errors match the scatter of the rates
    assert abs(np.std(pulls) - 1) < 0.25
    # hispecdrp's criteria
    assert np.sum(np.abs(pulls) > 10) == 0
    assert abs(np.mean(np.abs(pulls) < 1) - 0.683) < 0.15
