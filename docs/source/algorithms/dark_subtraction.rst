Dark Subtraction
================

Subtracting the Dark
--------------------

:func:`.subtract_dark_rate` subtracts a dark current rate image (DN/s) from a rate image, in place.

.. code-block:: python

    from oircal.dark_subtraction import subtract_dark_rate

    dark_rate = np.full((Nrows, Ncols), 0.01, dtype=np.float32)  # DN/s
    dark_err = np.full((Nrows, Ncols), 0.001, dtype=np.float32)  # DN/s
    subtract_dark_rate(
        input_data=rate,
        dark_rate=dark_rate,
        input_err=rate_err,
        dark_err=dark_err,
    )

Errors are combined in quadrature only if both ``input_err`` and ``dark_err`` are given. Likewise, pass both ``input_dq`` and ``dark_dq`` to OR the dark DQ into the image DQ.

Make Dark
---------

:func:`.make_dark` makes the dark calibration from a dark rate map, e.g. several dark exposures combined with :func:`.coadd`. It returns copies of the rate and its error as ``dark_rate`` and ``dark_err``, and ``dark_dq`` with pixels above ``hot_pixel_threshold`` (in the units of the rate) flagged ``HOT``. The keys match the arguments of :func:`.subtract_dark_rate`:

.. code-block:: python

    from oircal.dark_subtraction import make_dark

    dark_rate_map = np.full((Nrows, Ncols), 0.01, dtype=np.float32)  # DN/s
    dark_rate_map[10, 20] = 50.0  # a hot pixel
    dark = make_dark(dark_rate_map, dq_flags=dq_flags, hot_pixel_threshold=5.0)
    subtract_dark_rate(input_data=rate, input_dq=dq, **dark)
