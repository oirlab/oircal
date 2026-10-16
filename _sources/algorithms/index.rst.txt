Algorithms
==========

Algorithms in ``oircal`` are designed to work directly with NumPy arrays (see :doc:`../data_products`):

- Most functions modify their inputs in place and also return them.

- Most functions take a ``max_cores`` argument (default 1) to parallelize over detector rows.

Each page below shows one algorithm on its own. The examples on these pages continue from the setup below, which simulates a 64 x 64 pixel up-the-ramp exposure of a flat 50 e-/s source (see :doc:`../simulations`) and fits it with :func:`.fit_ramps_ols` to give the rate map ``rate`` and its error ``rate_err``.

.. code-block:: python

    import numpy as np

    from oircal.ramp_fitting import fit_ramps_ols
    from oircal.simulations import create_ramp
    from oircal.utils.read_times import get_delta_read_times, get_first_read_mjd

    dq_flags = {
        "GOOD": 0,
        "DO_NOT_USE": 2**0,
        "SATURATED": 2**1,
        "JUMP_DET": 2**2,
        "NON_LINEAR": 2**3,
        "OUTLIER": 2**4,
        "UNRELIABLE_SLOPE": 2**5,
        "UNRELIABLE_FLAT": 2**6,
        "REFERENCE_PIXEL": 2**7,
        "UNRELIABLE_BIAS": 2**8,
        "HOT": 2**9,
        "COLD": 2**10,
    }

    Nrows, Ncols = 64, 64
    Nreads, readtime = 10, 1.7  # number of reads, seconds between reads
    gain = np.full((Nrows, Ncols), 1.5, dtype=np.float32)  # e-/DN
    read_noise = np.full((Nrows, Ncols), 5.0, dtype=np.float32)  # DN
    bias = np.full((Nrows, Ncols), 1000.0, dtype=np.float32)  # DN

    source = np.full((Nrows, Ncols), 50.0, dtype=np.float32)  # e-/s
    ramp = create_ramp(
        source=source,
        readtime=readtime,
        Nreads=Nreads,
        gain=gain,
        bias=bias * gain,
        read_noise=read_noise * gain,
        convert_to_uint16=False,
        seed=0,
    )
    data = ramp["data"].astype(np.float32)  # (Nreads, Nrows, Ncols), DN
    # Readout timing from the clock rate and channel layout
    meta = ramp["meta"]
    readout = dict(
        clock_rate=meta["clock_rate"],
        Nchannels=meta["n_channels"],
        vertical=meta["channels_are_vertical"],
    )
    times = get_delta_read_times(Nreads, Nrows, Ncols, **readout)  # s since the first read
    start_times = get_first_read_mjd(meta["mjd_start"], Nrows, Ncols, **readout)  # MJD
    dq_raw = np.zeros(data.shape, dtype=np.uint8)  # per-read DQ
    dq = np.zeros((Nrows, Ncols), dtype=np.uint32)  # per-pixel DQ

    # Rate map for the examples that operate on rates
    ramp_fit_result = fit_ramps_ols(times=times, input_data=data, rdnoise=read_noise, gain=gain)
    rate, rate_err = ramp_fit_result["rate"], ramp_fit_result["rate_err"]  # (Nrows, Ncols), DN/s

.. toctree::
   :maxdepth: 1

   saturation
   bias_subtraction
   refpix
   nonlinear_correction
   jump_detection
   ramp_fitting
   gain
   dark_subtraction
   detector_flat
   coadd
