"""Unit tests: jwst_likely recovers signed Gaussian background (no floor at 0)."""

from __future__ import annotations

import numpy as np

from oircal.ramp_fitting.fit_ramp_jwst import fit_ramps_jwst
from oircal.simulations import create_ramp


def test_jwst_likely_background_allows_negative_rates(dq_flags):
    """Empty pixels with read noise must yield a signed rate distribution about 0."""
    np.random.seed(42)
    ny, nx = 32, 128
    rate = np.zeros((ny, nx), dtype=np.float32)
    rate[:, 60:68] = 8.0  # bright strip

    ramp = create_ramp(
        rate,
        readtime=1.7,
        n_reads=8,
        gain=1.0,
        flat=1.0,
        dark=0.0,
        bias=1000.0,
        kTC_noise=20.0,
        read_noise=10.0,
        poisson_noise=True,
        convert_to_uint16=True,
    )
    data = np.asarray(ramp["data"], dtype=np.float32)
    dq_raw = np.asarray(ramp["dq_raw"])
    n_reads = data.shape[0]
    times = np.arange(n_reads, dtype=np.float64) * 1.7
    rdnoise = np.full((ny, nx), 10.0, dtype=np.float32)
    gain = np.ones((ny, nx), dtype=np.float32)
    start_times = np.zeros((ny, nx), dtype=np.float64)

    out = fit_ramps_jwst(
        times,
        data,
        dq_flags,
        dq_raw=dq_raw,
        rdnoise=rdnoise,
        gain=gain,
        start_times=start_times,
        max_cores=1,
        algorithm="jwst_likely",
        jump_detection=False,
    )
    fitted = np.asarray(out["rate"], dtype=np.float64)
    bg = np.concatenate([fitted[:, :55].ravel(), fitted[:, 75:].ravel()])

    assert bg.size > 1000
    assert np.mean(bg < 0) > 0.35
    assert np.mean(bg == 0) < 0.01
    assert abs(float(np.mean(bg))) < 0.2
    assert float(np.std(bg)) > 0.3
    assert float(np.mean(fitted[:, 60:68])) > 5.0
