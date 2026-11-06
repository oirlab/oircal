Read Noise
==========

:func:`.calc_readnoise` calculates the single-read read noise, in DN, from a set of dark exposures. The input is an array of shape ``(Nfiles, Nreads, Nrows, Ncols)``, or an array of shape ``(Nreads, Nrows, Ncols)`` for a single exposure.

The function does not need the gain. To fit the gain and the read noise together, use :func:`.fit_gain_and_readnoise` instead (see :doc:`gain`).

Method
------

A read is usable unless it is flagged ``DO_NOT_USE`` or ``SATURATED`` or is not a finite number. Let :math:`S_k` be usable read :math:`k` of a ramp, in DN, counting only the usable reads. The function works with the differences between consecutive usable reads,

.. math::

    d_k = S_{k+1} - S_k.

Without flagged reads, these are the differences between consecutive reads. Unusable reads are skipped rather than ending the differences: with every other read flagged, for example, each difference spans two reads.

Any offset that is the same for every read of a ramp (e.g., kTC noise) cancels in :math:`d_k`, so it does not affect the result.

Let :math:`\sigma_\mathrm{read}` be the single-read read noise, in DN. Each difference contains the read noise of two independent reads, so its read noise variance is :math:`2 \sigma_\mathrm{read}^2`. If the dark signal is negligible, :math:`d_k` has mean 0, and the expected value of :math:`d_k^2` is therefore :math:`2 \sigma_\mathrm{read}^2`. The function estimates the read noise variance of each pixel as

.. math::

    \hat{\sigma}_\mathrm{read}^2 = \frac{1}{2n} \sum_k d_k^2,

where the sum runs over all :math:`n` usable differences of the pixel, from all reads of all ramps. In this case, the estimator :math:`\hat{\sigma}_\mathrm{read}^2` is unbiased. Because all differences are combined into one sum, the result is a single read noise per pixel, averaged over all reads. The function does not estimate the read noise of individual reads.

The function does not subtract the dark signal, so any dark current biases the estimate towards larger values. Let :math:`R_{D}` be the dark rate of the pixel, in DN/s, let :math:`g` be the gain, in e-/DN, and let :math:`\Delta t_k` be the time between usable reads :math:`k` and :math:`k+1`. The difference :math:`d_k` then has mean :math:`R_{D} \, \Delta t_k` and an additional dark shot noise variance of :math:`R_{D} \, \Delta t_k / g`, so

.. math::

    E[d_k^2] = \underbrace{2 \sigma_\mathrm{read}^2}_\text{read noise} + \underbrace{\frac{R_{D} \, \Delta t_k}{g}}_\text{dark shot noise} + \underbrace{R_{D}^2 \Delta t_k^2}_\text{(mean dark signal)$^2$}.

The estimate :math:`\hat{\sigma}_\mathrm{read}^2` is therefore too high by the average over the usable differences of :math:`R_{D} \, \Delta t_k / (2g) + R_{D}^2 \Delta t_k^2 / 2`. For a dark rate of 0.01 DN/s, a gain of 3 e-/DN and 1.7 s between reads, this bias is 0.003 DN\ :sup:`2`, which is negligible compared with a read noise variance of 25 DN\ :sup:`2` (5 DN) or more. A difference that skips unusable reads spans a longer :math:`\Delta t_k`, so it is biased more.

Output
------

With ``mode="per_pixel"`` (the default), the function returns a ``(Nrows, Ncols)`` array with the read noise of each pixel. With ``mode="full_detector"``, it returns a single float for the whole detector. This value is the square root of the mean of the per-pixel variance estimates :math:`\hat{\sigma}_\mathrm{read}^2`.

Unusable reads are skipped as described in the Method section. A difference :math:`d_k` is left out of the sum if usable read :math:`k+1`, or any unusable read between usable reads :math:`k` and :math:`k+1`, is flagged ``JUMP_DET``, since the jump happened between the two reads. With ``mode="per_pixel"``, a pixel with no usable differences is NaN. With ``mode="full_detector"``, such pixels are left out of the mean.

Example
-------

The example simulates 20 dark exposures with a nominal gain of 3 e-/DN. Each has 10 reads, a dark current of 0.01 DN/s and 500 e- of kTC noise.

.. code-block:: python

    from oircal.read_noise import calc_readnoise

    Nfiles, n_dark_reads = 20, 10
    gain = np.full((Nrows, Ncols), 3.0, dtype=np.float32)  # e-/DN

    darks = []
    for seed in range(Nfiles):
        dark_ramp = create_ramp(
            source=np.zeros((Nrows, Ncols)),  # e-/s
            readtime=readtime,
            Nreads=n_dark_reads,
            gain=gain,
            bias=bias * gain,
            dark=0.01 * gain,  # e-/s
            read_noise=read_noise * gain,
            kTC_noise=500.0,  # e-
            convert_to_uint16=False,
            seed=seed,
        )
        darks.append(dark_ramp["data"])
    darks = np.array(darks)  # (Nfiles, n_dark_reads, Nrows, Ncols)

    rn_map = calc_readnoise(input_data=darks)  # (Nrows, Ncols), DN
    rn_detector = calc_readnoise(input_data=darks, mode="full_detector")  # DN
