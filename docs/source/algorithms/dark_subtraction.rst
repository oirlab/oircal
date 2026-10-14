Dark Subtraction
================

:func:`.subtract_dark_rate` subtracts a dark current rate image (DN/s) from a rate image, in place.

.. code-block:: python

    from oircal.dark_subtraction import subtract_dark_rate
    from oircal.ramp_fitting import fit_ramps_ols

    result = fit_ramps_ols(times=times, input_data=data, rdnoise=read_noise, gain=gain)
    rate, rate_err = result["rate"], result["rate_err"]

    dark_rate = np.full((Nrows, Ncols), 0.01, dtype=np.float32)  # DN/s
    dark_err = np.full((Nrows, Ncols), 0.001, dtype=np.float32)  # DN/s
    subtract_dark_rate(
        input_data=rate,
        dark_rate=dark_rate,
        input_err=rate_err,
        dark_err=dark_err,
    )

Errors are combined in quadrature only if both ``input_err`` and ``dark_err`` are given. Likewise, pass both ``input_dq`` and ``dark_dq`` to OR the dark DQ into the image DQ.
