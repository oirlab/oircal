Detector Flat
=============

Correcting the Flat
-------------------

:func:`.correct_detector_flat` divides a rate image by the detector-level flat field, in place. Pixels where the flat is not positive (or is NaN) are left uncorrected.

.. code-block:: python

    from oircal.detector_flat import correct_detector_flat

    flat = np.ones((Nrows, Ncols), dtype=np.float32)
    correct_detector_flat(
        input_data=rate,
        detflat=flat,
        input_err=rate_err,
        input_var_poisson=ramp_fit_result["rate_var_poisson"],
        input_var_rnoise=ramp_fit_result["rate_var_rnoise"],
    )

If ``input_err`` is given, it is propagated through the division. The flat contributes to the error only if ``detflat_err`` is also given; otherwise it is treated as noiseless. The variances ``input_var_poisson`` and ``input_var_rnoise`` are divided by the flat squared; ``detflat_err`` enters only ``input_err``. Pass both ``input_dq`` and ``detflat_dq`` to OR the flat's DQ into the image DQ.

Make Detector Flat
------------------

:func:`.make_detector_flat` makes the flat from a flat-field rate map, e.g. several exposures combined with :func:`.coadd`, divided by its median. Pixels below ``cold_pixel_threshold`` (0.9 by default) are flagged ``COLD``. It returns ``detflat``, ``detflat_err`` and ``detflat_dq``, plus ``median_rate`` for monitoring the lamp:

.. code-block:: python

    from oircal.detector_flat import make_detector_flat

    flat_rate = np.full((Nrows, Ncols), 1000.0, dtype=np.float32)  # DN/s
    flat_rate[5, 5] = 600.0  # a cold pixel
    flat_calib = make_detector_flat(flat_rate, dq_flags=dq_flags)
    flat, flat_dq = flat_calib["detflat"], flat_calib["detflat_dq"]
