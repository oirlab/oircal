Up the Ramp Fitting
===================

Ramp fitting turns up-the-ramp data ``(Nreads, Nrows, Ncols)`` into a 2D rate image (DN/s). All three functions take the read times, the ramps, and optionally the DQ arrays, read noise and gain, and return a dict of 2D arrays:

- ``rate``, ``rate_err``: the rate and its uncertainty (DN/s).
- ``rate_var_rnoise``, ``rate_var_poisson``: the read noise and Poisson contributions to the variance.
- ``dq``: the per-pixel DQ, including flags propagated from ``dq_raw``.
- ``mid_time``: the mid time of the valid reads (MJD if ``start_times`` is given).

Pixels whose fit fails get NaN for the rate, its error and its variances, and are flagged ``DO_NOT_USE`` in ``dq`` when ``dq_flags`` is given.

Ordinary least squares
----------------------

:func:`.fit_ramps_ols` fits a line with a free intercept to each ramp segment. The output also has ``bias`` and ``bias_err``, the fitted intercept.

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
    )
    rate, rate_err = result["rate"], result["rate_err"]

Multiple correlated double sampling
-----------------------------------

:func:`.fit_ramps_mcds` computes the rate from the difference between the start and end of each ramp segment, averaging ``num_coadd`` reads at each end. It takes the same arguments as :func:`.fit_ramps_ols`, plus ``num_coadd``.

Set ``num_coadd=1`` to use CDS.

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

- ``"JWST_LIKELY"``: likelihood-based fitting (Brandt 2024) from `stcal <https://github.com/spacetelescope/stcal>`_. Needs at least 4 reads.

- ``"JWST_FIXSEN"``: A quasi-likelihood-based fitting method as described in `Fixsen et al 2000 <https://ui.adsabs.harvard.edu/abs/2000PASP..112.1350F/exportcitation>`_.

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
