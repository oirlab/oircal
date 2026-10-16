Up the Ramp Fitting
===================

Ramp fitting turns up-the-ramp data ``(Nreads, Nrows, Ncols)`` into a 2D rate image (DN/s). All three functions take the read times, the ramps, and optionally the DQ arrays, read noise and gain, and return a dict of 2D arrays:

- ``rate``, ``rate_err``: the rate and its uncertainty (DN/s).
- ``rate_var_rnoise``, ``rate_var_poisson``: the read noise and Poisson contributions to the variance.
- ``dq``: the per-pixel DQ, including flags propagated from ``dq_raw``.
- ``mid_time``: the mid time of the valid reads (MJD if ``start_times`` is given).

Pixels whose fit fails get NaN for the rate, its error and its variances, and are flagged ``DO_NOT_USE`` in ``dq`` when ``dq_flags`` is given.

Read times
----------

HxRG arrays integrate continuously and are read non-destructively: each readout channel samples its pixels one at a time at the pixel clock rate, so different pixels are read at different times. Ramp fitting needs this timing in two forms:

- ``times`` is the time of each read in seconds since the first read, shape ``(Nreads,)``. The same offsets apply to every pixel. :func:`.get_delta_read_times` computes them from the detector size, number of channels and clock rate.
- ``start_times`` (optional) is the MJD of each pixel's first read, shape ``(Nrows, Ncols)``, and puts ``mid_time`` in MJD. :func:`.get_first_read_mjd` computes it from the readout start, the channel layout and each channel's read direction.

Consecutive reads are one frame time apart, the time for each channel to read all of its pixels at one pixel per clock tick:

.. math::

   t_\mathrm{frame} = \frac{N_\mathrm{rows} \, N_\mathrm{cols,chan}}{f_\mathrm{clock}} = \frac{N_\mathrm{rows} \, N_\mathrm{cols}}{N_\mathrm{channels} \, f_\mathrm{clock}},

where :math:`f_\mathrm{clock}` is the pixel clock rate (``clock_rate``) and :math:`N_\mathrm{cols,chan} = N_\mathrm{cols} / N_\mathrm{channels}` is the width of each vertical channel. Read :math:`k` is then at ``times[k]`` :math:`= k \, t_\mathrm{frame}`, for :math:`k = 0, 1, \ldots, N_\mathrm{reads} - 1`.

.. _mid_time:

Each pixel's ``mid_time`` (MJD) is its first read plus the mean time of its usable read differences, not weighted by flux:

.. math::

   t_\mathrm{mid} = t_\mathrm{start} + \frac{1}{86400 \, |U|} \sum_{k \in U} \frac{t_k + t_{k+1}}{2},

where :math:`t_\mathrm{start}` is the pixel's ``start_times`` (MJD), :math:`t_k` is ``times[k]`` (s) and 86400 converts seconds to days. :math:`U` holds the differences :math:`k \to k+1` where neither read is flagged ``DO_NOT_USE`` or ``SATURATED`` and read :math:`k+1` is not flagged ``JUMP_DET``. With no flagged reads, this reduces to

.. math::

   t_\mathrm{mid} = t_\mathrm{start} + \frac{(N_\mathrm{reads} - 1) \, t_\mathrm{frame}}{2 \times 86400}.

The diagram below shows the readout timing of a small detector.

.. raw:: html
   :file: ../_includes/readout_pattern.html

A 4 x 9 detector with 3 vertical channels read in parallel in alternating directions. For a constant source, pixels read later have the same ramp, shifted in time.

Ordinary least squares
----------------------

:func:`.fit_ramps_ols` fits a line with a free intercept to each ramp segment. The output also has ``bias`` and ``bias_err``, the fitted intercept.

A ramp is split into segments at reads flagged ``JUMP_DET``, and reads flagged ``DO_NOT_USE`` or ``SATURATED`` are left out. For a segment of :math:`n` reads with times :math:`t_i` and signals :math:`S_i` (DN), the rate :math:`r` and intercept :math:`b` are

.. math::

   r = \frac{n \sum t_i S_i - \sum t_i \sum S_i}{\Delta}, \qquad
   b = \frac{\sum t_i^2 \sum S_i - \sum t_i \sum t_i S_i}{\Delta},

.. math::

   \Delta = n \sum t_i^2 - \Big(\sum t_i\Big)^2,

with read noise and Poisson variances

.. math::

   \sigma_{r,\mathrm{read}}^2 = \frac{n \, \sigma_\mathrm{read}^2}{\Delta}, \qquad
   \sigma_{r,\mathrm{Poisson}}^2 = \frac{|r + D|}{g \, (t_n - t_1)},

where :math:`\sigma_\mathrm{read}` is the single-read read noise (``rdnoise``, DN), :math:`g` the gain (``gain``, e-/DN) and :math:`D` the dark current (``dark_current``, DN/s). ``bias`` comes from the segment that starts at the first read, and ``bias_err`` is :math:`\sigma_b = \sigma_\mathrm{read} \sqrt{\sum t_i^2 / \Delta}`.

The segments are combined by inverse-variance weighting, with :math:`\sigma_{r,k}^2 = \sigma_{r,\mathrm{read},k}^2 + \sigma_{r,\mathrm{Poisson},k}^2` for segment :math:`k`:

.. math::

   r = \frac{\sum_k r_k / \sigma_{r,k}^2}{\sum_k 1 / \sigma_{r,k}^2}, \qquad
   \sigma_r^2 = \frac{1}{\sum_k 1 / \sigma_{r,k}^2}.

``rate_err`` is :math:`\sigma_r`, and ``rate_var_rnoise`` and ``rate_var_poisson`` combine :math:`\sigma_{r,\mathrm{read},k}^2` and :math:`\sigma_{r,\mathrm{Poisson},k}^2` the same way.

.. code-block:: python

    from oircal.ramp_fitting import fit_ramps_ols

    result = fit_ramps_ols(
        times=times,
        input_data=data,
        input_dq_raw=dq_raw,
        input_dq=dq,
        dq_flags=dq_flags,
        rdnoise=read_noise,
        gain=gain,
        start_times=start_times,
    )
    rate, rate_err = result["rate"], result["rate_err"]
    mid_time = result["mid_time"]  # MJD

Multiple correlated double sampling
-----------------------------------

:func:`.fit_ramps_mcds` computes the rate from the difference between the start and end of each ramp segment, averaging ``num_coadd`` reads at each end. It takes the same arguments as :func:`.fit_ramps_ols`, plus ``num_coadd``.

Set ``num_coadd=1`` to use CDS.

With :math:`m` = ``num_coadd``, let :math:`\bar S_1, \bar t_1` and :math:`\bar S_2, \bar t_2` be the mean signal and time of the first and last :math:`m` reads of a segment. Then

.. math::

   r = \frac{\bar S_2 - \bar S_1}{\bar t_2 - \bar t_1}, \qquad
   \sigma_{r,\mathrm{read}}^2 = \frac{2 \, \sigma_\mathrm{read}^2}{m \, (\bar t_2 - \bar t_1)^2}, \qquad
   \sigma_{r,\mathrm{Poisson}}^2 = \frac{|r + D|}{g \, (\bar t_2 - \bar t_1)},

and the segments are combined as for :func:`.fit_ramps_ols`.

.. code-block:: python

    from oircal.ramp_fitting import fit_ramps_mcds

    result = fit_ramps_mcds(
        times=times,
        input_data=data,
        input_dq_raw=dq_raw,
        input_dq=dq,
        dq_flags=dq_flags,
        rdnoise=read_noise,
        gain=gain,
    )

Maximum Likelihood
------------------

:func:`.fit_ramps_jwst` wraps the JWST ramp fitting algorithms:

- ``"CYTHON_LIKELY"`` (default): the same algorithm as ``"JWST_LIKELY"`` implemented in Cython.

- ``"JWST_LIKELY"``: likelihood-based fitting (`Brandt 2024 <https://arxiv.org/abs/2309.08753>`_) from `stcal <https://github.com/spacetelescope/stcal>`_. Needs at least 4 reads.

- ``"JWST_FIXSEN"``: quasi-likelihood-based fitting (`Fixsen et al. 2000 <https://arxiv.org/abs/astro-ph/0005486>`_).

.. code-block:: python

    from oircal.ramp_fitting import fit_ramps_jwst

    result = fit_ramps_jwst(
        times=times,
        input_data=data,
        dq_flags=dq_flags,
        input_dq_raw=dq_raw,
        input_dq=dq,
        rdnoise=read_noise,
        gain=gain,
        algorithm="CYTHON_LIKELY",
    )

With ``jump_detection=True``, :func:`.fit_ramps_jwst` also runs :func:`.jump_detection_jwst` on the ramps.
