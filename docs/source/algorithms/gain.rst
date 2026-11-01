Gain and Read Noise
===================

The gain (e-/DN) and single-read read noise (DN) are inputs to ramp fitting and non-linearity fitting.

Converting to Electrons
-----------------------

:func:`.correct_gain` multiplies a rate image (DN/s) by the gain (e-/DN), in place, giving e-/s. The variances :param:`correct_gain.input_var_poisson` and :param:`correct_gain.input_var_rnoise` are multiplied by the gain squared.

.. code-block:: python

    from oircal.gain import correct_gain

    correct_gain(
        input_data=rate,
        gain=gain,
        input_err=rate_err,
        input_var_poisson=ramp_fit_result["rate_var_poisson"],
        input_var_rnoise=ramp_fit_result["rate_var_rnoise"],
    )

If :param:`correct_gain.input_err` is given, it is propagated through the product. The gain contributes to the error only if :param:`correct_gain.gain_err` is also given, e.g. ``np.sqrt(cov[0, 0])`` from :func:`.fit_gain_and_readnoise`; otherwise it is treated as noiseless. :param:`correct_gain.gain_err` enters only :param:`correct_gain.input_err`. Pass both :param:`correct_gain.input_dq` and :param:`correct_gain.gain_dq` to OR the gain's DQ into the image DQ.

Make Gain and Read Noise
------------------------

:func:`.fit_gain_and_readnoise` fits the gain and read noise of each pixel from a stack of exposures of shape ``(Nfiles, Nreads, Nrows, Ncols)``. It maximizes the likelihood of the ramps, marginalized over each ramp's count rate (`Brandt 2025 <https://ui.adsabs.harvard.edu/abs/2025PASP..137l5006B/abstract>`_), starting from :param:`fit_gain_and_readnoise.init_gain` and :param:`fit_gain_and_readnoise.init_rn`. :param:`fit_gain_and_readnoise.method` sets how the maximum is found:

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

Photon Transfer Gain
--------------------

:func:`.calc_gain` calculates the gain of each pixel from the mean and variance of the differences between consecutive reads of illuminated ramps, holding the read noise at a known value :param:`calc_gain.rn` (e.g. from :func:`.calc_readnoise`). The reads must be evenly spaced. Let :math:`\mu` be the mean signal per read interval (DN) and :math:`\sigma_\mathrm{read}` the single-read read noise (DN). A read difference has variance :math:`\mu / g + 2 \sigma_\mathrm{read}^2`.

The differences of each ramp are split into runs of consecutive usable differences, ended by a read flagged ``DO_NOT_USE`` or ``SATURATED``, a non-finite read, or a read flagged ``JUMP_DET``. For a run of :math:`n \ge 2` differences :math:`d_k` with mean :math:`\bar d`,

.. math::

    A = \sum_k (d_k - \bar d)^2 - 2 \sigma_\mathrm{read}^2 \left(n - \frac{1}{n}\right), \qquad B = (n - 1) \, \bar d,

with expected values :math:`(n - 1) \mu / g` and :math:`(n - 1) \mu`. The read noise term has :math:`n - 1/n` rather than :math:`n - 1` because neighboring differences share a read. The gain is :math:`g = \sum B / \sum A` over all runs of all ramps of the pixel, so the ramps may have different count rates.

With ``mode="full_detector"``, the inverse gains :math:`\sum A / \sum B` of all pixels are combined with :func:`.combine`: sigma clipped at :param:`calc_gain.n_sigma` (5 by default), then averaged, with the error from the scatter of the kept pixels. The gain is the inverse of that mean. The inverse gain is averaged rather than the gain because it is nearly linear in the noisy :math:`A`, so its mean is not biased by the noise of each pixel's gain.

The error of the gain comes from the scatter of the runs about the gain, as for a ratio estimator. With :math:`N` runs,

.. math::

    \sigma_g^2 = \frac{N}{N - 1} \, \frac{\sum (B - g A)^2}{\left(\sum A\right)^2}.

It is reliable only with many runs, e.g. 10 or more ramps.

.. code-block:: python

    from oircal.gain import calc_gain

    gain_map, gain_err = calc_gain(input_data=illum, rn=5.0)  # (Nrows, Ncols), e-/DN
