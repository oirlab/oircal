Bias Subtraction
================

Subtracting the Bias
--------------------

:func:`.subtract_bias` subtracts a 2D bias image from every read of the ramp, in place.

.. code-block:: python

    from oircal.bias_subtraction import subtract_bias

    subtract_bias(input_data=data, bias_data=bias)

To propagate the bias calibration's data quality, pass both DQ arrays. The bias DQ is OR'ed into ``dq``:

.. code-block:: python

    bias_dq = np.zeros((Nrows, Ncols), dtype=np.uint32)
    subtract_bias(input_data=data, bias_data=bias, input_dq=dq, bias_dq=bias_dq)

Make Bias
---------

:func:`.make_bias` makes the bias from the first reads of a set of exposures. Each pixel's first reads are sigma clipped with :func:`.combine`. The bias is the median of the kept reads, and the kTC (reset) noise ``kTC`` is their standard deviation. Pixels whose bias or kTC noise is above ``bias_threshold`` or ``kTC_threshold`` are flagged ``UNRELIABLE_BIAS``.

The kTC noise is the scatter of a pixel's first read across exposures, and includes the read noise of that read. The example simulates 20 dark exposures and keeps the first read of each:

.. code-block:: python

    from oircal.bias_subtraction import make_bias

    first_reads = []
    for seed in range(20):
        dark_ramp = create_ramp(
            source=np.zeros((Nrows, Ncols)),  # dark: no signal
            readtime=readtime,
            Nreads=2,
            gain=gain,
            bias=bias * gain,
            read_noise=read_noise * gain,
            convert_to_uint16=False,
            seed=seed,
        )
        first_reads.append(dark_ramp["data"][0])
    first_reads = np.array(first_reads)  # (20, Nrows, Ncols)

    calib = make_bias(first_reads, dq_flags=dq_flags, bias_threshold=2000.0)  # DN
    bias_data, bias_dq, kTC = calib["bias_data"], calib["bias_dq"], calib["kTC"]
