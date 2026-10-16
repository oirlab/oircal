Coadding
========

:func:`.coadd` and :func:`.combine` combine a stack of shape ``(N, ...)`` along its first axis, independently for every element of the other axes: frames of shape ``(Nframes, Nrows, Ncols)``, spectra of shape ``(Nspectra, Npix)`` or ``(Nspectra, Norders, Npix)``, or measurements such as RVs of shape ``(Nobs,)``. :func:`.coadd` coadds data products with their DQ, variances and mid times, using :func:`.combine` for the data and errors.

Methods
-------

``method`` sets how the good values are combined:

- ``"wmean"`` (default): inverse-variance weighted mean, with weights ``1 / input_err**2``.
- ``"mean"``: mean.
- ``"median"``: median.

``err_method`` sets the uncertainty:

- ``"se"`` (default): standard error propagated from ``input_err``.
- ``"scatter"``: measured from the spread of the good values, so it does not need ``input_err`` (except for ``"wmean"``, which needs it for the weights).

See :func:`.combine` for the formulas.

Coadding frames
---------------

:func:`.coadd` coadds each element (e.g. pixel) independently. Values with nonzero ``input_dq`` or non-finite data (or errors) are excluded. The data and errors of the remaining values are combined with ``method`` and ``err_method``; the variance components are combined to match, DQ flags with a bitwise OR, and mid times with a mean. Elements with no good values are NaN and flagged ``DO_NOT_USE``.

.. code-block:: python

    from oircal.coadd import coadd

    Nframes = 5
    rng = np.random.default_rng(0)
    rates = rng.normal(33.3, 0.5, size=(Nframes, Nrows, Ncols)).astype(np.float32)  # DN/s
    rate_errs = np.full(rates.shape, 0.5, dtype=np.float32)  # DN/s
    rate_dqs = np.zeros(rates.shape, dtype=np.uint32)

    result = coadd(
        input_data=rates,
        input_err=rate_errs,
        input_dq=rate_dqs,
        dq_flags=dq_flags,
        method="wmean",
        err_method="se",
        do_sigma_clip=True,
    )
    rate, rate_err = result["data"], result["err"]

The result is a dict of arrays of shape ``input_data.shape[1:]`` with keys ``data``, ``err``, ``var_poisson``, ``var_rnoise``, ``dq`` and ``mid_time``; a key is None if it cannot be computed from the inputs given.

Coadding spectra
----------------

:func:`.coadd_spectra` coadds spectra of shape ``(Nspectra, Npix)`` or ``(Nspectra, Norders, Npix)`` that are already sampled on a common wavelength grid (no resampling is done). It takes the same arguments as :func:`.coadd` and returns arrays of shape ``(Npix,)`` or ``(Norders, Npix)``. When sigma clipping, the spectra should be at comparable flux levels:

.. code-block:: python

    from oircal.coadd import coadd_spectra

    Nspectra, Norders, Npix = 4, 3, 100
    fluxes = rng.normal(100.0, 2.0, size=(Nspectra, Norders, Npix)).astype(np.float32)
    flux_errs = np.full(fluxes.shape, 2.0, dtype=np.float32)

    result = coadd_spectra(input_data=fluxes, input_err=flux_errs, method="wmean")
    flux, flux_err = result["data"], result["err"]

Combining other measurements
----------------------------

:func:`.combine` takes any stack and returns the combined values, their uncertainties, and the mask of values used (after sigma clipping), so other quantities can be combined over the same values:

.. code-block:: python

    from oircal.coadd import combine

    rvs = rng.normal(10.0, 2.0, size=20)  # m/s
    rv_errs = np.full(rvs.shape, 2.0)  # m/s

    rv, rv_err, used = combine(rvs, input_err=rv_errs, method="wmean", err_method="scatter")

Sigma clipping
--------------

Sigma clipping is optional (``do_sigma_clip=True``, with more than 3 values) and always uses the biweight location and midvariance: values beyond ``sigma_thresh_low`` or ``sigma_thresh_high`` standard deviations from the biweight location are rejected, iteratively, keeping at least ``min_keep`` values. :func:`.sigma_clip_stack` does this for any stack, along its first axis:

.. code-block:: python

    from oircal.utils.sigma_clipping import sigma_clip_stack

    good = sigma_clip_stack(rates, sigma_thresh_low=5.0, sigma_thresh_high=5.0)

:func:`.sigma_clip` does the same for a single 1D array, and can be called from Numba code.
