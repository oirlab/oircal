Reference Pixel Correction
==========================

:func:`.ref_filter` uses the vertical (left and right) reference pixel columns to remove correlated 1/f noise, which appears as horizontal stripes. It smooths the reference columns in the frequency domain (or with a Savitzky-Golay filter) and subtracts them from each row. It accepts a single frame ``(Nrows, Ncols)`` or a cube ``(Nreads, Nrows, Ncols)``.

.. code-block:: python

    from oircal.refpix import ref_filter

    # The 4 leftmost and 4 rightmost columns are reference pixels, read out over 4 channels
    ref_filter(input_data=data, nchans=4, nleft=4, nright=4)

By default the data are corrected in place. Pass ``in_place=False`` to return a corrected copy, and ``left_ref=False`` or ``right_ref=False`` to use only one side.
