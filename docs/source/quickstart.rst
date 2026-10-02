Quickstart
==========

Ensure you have followed the :doc:`installation instructions <installation>` before proceeding with this quickstart guide.

Every function in ``oircal`` operates on NumPy arrays. Up-the-ramp data are 3D arrays of shape ``(n_reads, ny, nx)``, and per-pixel images (rates, calibrations, data quality) are 2D arrays of shape ``(ny, nx)``. Most functions modify their inputs in place and also return them.

Data quality flags
------------------

Functions that read or set data quality (DQ) flags take a ``dq_flags`` argument: a named tuple whose fields give the bit value of each flag. Only the fields a function uses are required (see each function's docstring). Create it from a dict with :func:`~oircal.utils.dq_utils.dq_flags_to_namedtuple`:

.. code-block:: python

    from oircal.utils.dq_utils import dq_flags_to_namedtuple

    dq_flags = dq_flags_to_namedtuple({"DO_NOT_USE": 1, "SATURATED": 2, "JUMP_DET": 4})

.. note::

    Numba supports Python's built-in named tuples, which are more efficient than Numba's typed dict.

Processing a ramp
-----------------

The example below simulates an up-the-ramp exposure, then flags saturation, subtracts the bias, fits the ramps to get a rate image, and applies dark and detector flat corrections.

.. code-block:: python

    import numpy as np

    from oircal.simulations import create_ramp
    from oircal.saturation import check_saturation
    from oircal.bias_subtraction import subtract_bias
    from oircal.ramp_fitting import fit_ramps_ols
    from oircal.dark_subtraction import subtract_dark_rate
    from oircal.detector_flat import correct_detector_flat

    ny, nx = 64, 64
    n_reads, readtime = 10, 1.7  # number of reads, seconds between reads
    gain = np.full((ny, nx), 1.5, dtype=np.float32)  # e-/DN
    read_noise = np.full((ny, nx), 5.0, dtype=np.float32)  # DN
    bias = np.full((ny, nx), 1000.0, dtype=np.float32)  # DN

    # Simulate an up-the-ramp exposure of a flat 50 e-/s source
    source = np.full((ny, nx), 50.0, dtype=np.float32)
    ramp = create_ramp(
        source, readtime=readtime, n_reads=n_reads,
        gain=gain, bias=bias * gain, read_noise=read_noise * gain,
        convert_to_uint16=False, seed=0,
    )
    data = ramp["data"].astype(np.float32)  # (n_reads, ny, nx), DN
    dq_raw = np.zeros(data.shape, dtype=np.uint8)  # per-read DQ
    dq = np.zeros((ny, nx), dtype=np.uint32)  # per-pixel DQ

    # 1. Flag saturated reads
    sat_thresh = np.full((ny, nx), 65535.0, dtype=np.float32)
    check_saturation(data, dq_raw, dq, sat_thresh, np.zeros((ny, nx), np.uint32), dq_flags=dq_flags)

    # 2. Subtract the bias
    subtract_bias(data, bias)

    # 3. Fit the ramps to get a rate image (DN/s)
    times = np.arange(n_reads) * readtime
    result = fit_ramps_ols(
        times, data, dq_raw=dq_raw, dq=dq, dq_flags=dq_flags,
        rdnoise=read_noise, gain=gain,
    )
    rate, rate_err = result["rate"], result["rate_err"]

    # 4. Subtract a dark and apply a detector flat
    dark_rate = np.zeros((ny, nx), dtype=np.float32)
    flat = np.ones((ny, nx), dtype=np.float32)
    subtract_dark_rate(rate, dark_rate, input_err=rate_err)
    correct_detector_flat(rate, flat, input_err=rate_err)

    print(np.median(rate))  # ~33.3 DN/s = 50 e-/s / 1.5 e-/DN

Other ramp fitting algorithms are available through :func:`~oircal.ramp_fitting.fit_ramps_mcds` and :func:`~oircal.ramp_fitting.fit_ramps_jwst` (likelihood-based fitting via ``stcal`` or a Cython implementation). Jump detection is in :mod:`oircal.jump_detection`, and non-linearity fitting and correction are in :mod:`oircal.nonlinear_correction`. See the :doc:`API reference <api/api>` for the full list.
