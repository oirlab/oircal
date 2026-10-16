Gain and Read Noise
===================

The gain (e-/DN) and single-read read noise (DN) are inputs to ramp fitting and non-linearity fitting.

Converting to Electrons
-----------------------

:func:`.correct_gain` multiplies a rate image (DN/s) by the gain (e-/DN), in place, giving e-/s. The variances ``input_var_poisson`` and ``input_var_rnoise`` are multiplied by the gain squared.

.. code-block:: python

    from oircal.gain import correct_gain

    correct_gain(
        input_data=rate,
        gain=gain,
        input_err=rate_err,
        input_var_poisson=ramp_fit_result["rate_var_poisson"],
        input_var_rnoise=ramp_fit_result["rate_var_rnoise"],
    )

If ``input_err`` is given, it is propagated through the product. The gain contributes to the error only if ``gain_err`` is also given, e.g. ``np.sqrt(cov[0, 0])`` from :func:`.fit_gain_and_readnoise`; otherwise it is treated as noiseless. ``gain_err`` enters only ``input_err``. Pass both ``input_dq`` and ``gain_dq`` to OR the gain's DQ into the image DQ.

Make Gain and Read Noise
------------------------

:func:`.fit_gain_and_readnoise` fits the gain and read noise of each pixel from a stack of exposures of shape ``(Nfiles, Nreads, Nrows, Ncols)``. It maximizes the likelihood of the ramps, marginalized over each ramp's count rate (`Brandt 2025 <https://ui.adsabs.harvard.edu/abs/2025PASP..137l5006B/abstract>`_), starting from ``init_gain`` and ``init_rn``. ``method`` sets how the maximum is found:

- ``"quad"`` (default): fits a quadratic form to the log likelihood on a 3-point grid around the initial values, then again around the first result.
- ``"grid"``: evaluates the likelihood on a dense grid and takes the errors from the marginal posteriors.
- ``"both"``: runs the grid search, then the quadratic fit around its peak.

Read differences touching a read flagged ``DO_NOT_USE`` or ``SATURATED``, or ending on a read flagged ``JUMP_DET``, are left out.

In DN, the variance of a read difference is the signal in that difference divided by the gain, plus twice the read noise squared, so fitting the gain needs illuminated ramps. Each pixel is fit to all of its ramps. The example simulates 40 illuminated exposures of a 4 x 4 region with 20 reads each:

.. code-block:: python

    from oircal.gain import fit_gain_and_readnoise

    Nfiles, n_illum_reads = 40, 20
    illum_times = np.arange(n_illum_reads) * readtime  # s

    illum = []
    for seed in range(Nfiles):
        illum_ramp = create_ramp(
            source=np.full((4, 4), 200.0),  # e-/s
            readtime=readtime,
            Nreads=n_illum_reads,
            gain=1.5,  # e-/DN
            bias=1500.0,  # e-
            read_noise=7.5,  # e-, i.e. 5 DN
            kTC_noise=0,
            convert_to_uint16=False,
            seed=seed,
        )
        illum.append(illum_ramp["data"])
    illum = np.array(illum)  # (Nfiles, n_illum_reads, 4, 4)

    gain_map, rn_map, cov = fit_gain_and_readnoise(
        times=illum_times,
        input_data=illum,
        init_gain=1.4,
        init_rn=4.5,
    )
    gain_err, rn_err = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])

``gain_map`` and ``rn_map`` have shape ``(Nrows, Ncols)`` and ``cov`` has shape ``(2, 2, Nrows, Ncols)``. With ``method="grid"``, the diagonal of ``cov`` holds the errors rather than the variances.

To fit only the read noise, e.g. from dark exposures, hold the gain at a known value with ``fit_gain=False`` (with ``"quad"`` or ``"both"``); ``fit_rn=False`` fits only the gain. The fixed parameter's entries in ``cov`` are NaN. ``mode="full_detector"`` fits one gain and read noise to all pixels at once.
