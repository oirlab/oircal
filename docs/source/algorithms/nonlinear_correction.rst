Non-linearity
=============

Two functions are provided: :func:`.fit_nonlin_coefs` fits non-linearity correction coefficients from a set of flat-field ramps, and :func:`.correct_nonlinearity` applies correction coefficients to ramps.

Fitting the coefficients
------------------------

:func:`.fit_nonlin_coefs` takes a 4D stack of ramps, ``(Nfiles, Nreads, Nrows, Ncols)``, that run up to saturation. It fits a polynomial (default order 5) that maps measured counts to linear counts. ``mode="per_pixel"`` fits each pixel; ``mode="full_detector"`` fits one polynomial to the whole detector.

The example below simulates bright flats with a quadratic non-linearity, flags saturation, and fits the correction:

.. code-block:: python

    from oircal.nonlinear_correction import fit_nonlin_coefs
    from oircal.saturation import check_saturation

    shape = (4, 4)
    Nfiles, n_flat_reads = 4, 60
    flat_gain = np.ones(shape, dtype=np.float32)
    flat_read_noise = np.full(shape, 0.1, dtype=np.float32)
    flat_bias = 1000.0
    sat_thresh = np.full(shape, 65535, dtype=np.uint16)

    flat_ramps, flat_dq_raw = [], []
    for _ in range(Nfiles):
        flat = create_ramp(
            source=np.full(shape, 1000.0),
            readtime=readtime,
            Nreads=n_flat_reads,
            gain=flat_gain,
            bias=flat_bias,
            read_noise=0.1,
            kTC_noise=0,
            nonlin_coeffs=[-1e-6, 1, 0],
            poisson_noise=False,
        )
        ramps_i = flat["data"]
        dq_raw_i = np.zeros(ramps_i.shape, dtype=np.uint8)
        check_saturation(
            input_data=ramps_i,
            input_dq_raw=dq_raw_i,
            input_dq=np.zeros(shape, np.uint32),
            sat_thresh=sat_thresh,
            sat_dq=np.zeros(shape, np.uint32),
            dq_flags=dq_flags,
        )
        flat_ramps.append(ramps_i)
        flat_dq_raw.append(dq_raw_i)
    flat_ramps = np.array(flat_ramps)
    flat_dq_raw = np.array(flat_dq_raw)

    nonlin_coeffs, nonlin_thresh, nonlin_max = fit_nonlin_coefs(
        input_data=flat_ramps,
        gain=flat_gain,
        rn=flat_read_noise,
        input_dq_raw=flat_dq_raw,
        dq_flags=dq_flags,
        satval=0.95 * (sat_thresh - flat_bias),
    )

``nonlin_coeffs`` has shape ``(order + 1, Nrows, Ncols)``, lowest degree first. ``nonlin_thresh`` is the level (DN) where counts deviate from linear by 1%, and ``nonlin_max`` is the level beyond which the correction is no longer reliable (10% by default; see ``criterion_for_nonlin_thresh`` and ``criterion_for_nonlin_max``).

Applying the correction
-----------------------

:func:`.correct_nonlinearity` subtracts the first read, applies the polynomial, and adds the first read back, so the corrected ramp keeps its bias. Reads above ``nonlin_max`` are flagged ``NON_LINEAR | DO_NOT_USE`` in ``dq_raw``, and the DQ map from the non-linearity calibration (``nonlin_dq``) is OR'ed into ``dq``. The ramps must be floating point. Only ``input_data`` and ``coeffs`` are required; the flagging and DQ propagation run only when their inputs are given.

.. code-block:: python

    from oircal.nonlinear_correction import correct_nonlinearity

    # Coefficients for the 64 x 64 science ramp: the identity polynomial P(x) = x
    coeffs = np.zeros((2, Nrows, Ncols))
    coeffs[1] = 1.0
    nonlin_max = np.full((Nrows, Ncols), 60000.0, dtype=np.float32)  # DN
    nonlin_dq = np.zeros((Nrows, Ncols), dtype=np.uint32)
    correct_nonlinearity(
        input_data=data,
        coeffs=coeffs,
        input_dq_raw=dq_raw,
        input_dq=dq,
        nonlin_max=nonlin_max,
        nonlin_dq=nonlin_dq,
        dq_flags=dq_flags,
    )
