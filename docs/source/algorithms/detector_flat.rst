Detector Flat
=============

:func:`.correct_detector_flat` divides a rate image by the detector-level flat field, in place. Pixels where the flat is not positive (or is NaN) are left uncorrected.

.. code-block:: python

    from oircal.detector_flat import correct_detector_flat
    from oircal.ramp_fitting import fit_ramps_ols

    result = fit_ramps_ols(times=times, input_data=data, rdnoise=read_noise, gain=gain)
    rate, rate_err = result["rate"], result["rate_err"]

    flat = np.ones((Nrows, Ncols), dtype=np.float32)
    correct_detector_flat(input_data=rate, detflat=flat, input_err=rate_err)

If ``input_err`` is given, it is propagated through the division. The flat contributes to the error only if ``detflat_err`` is also given; otherwise it is treated as noiseless. Pass both ``input_dq`` and ``detflat_dq`` to OR the flat's DQ into the image DQ.
