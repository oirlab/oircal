Bias Subtraction
================

:func:`.subtract_bias` subtracts a 2D bias image from every read of the ramp, in place.

.. code-block:: python

    from oircal.bias_subtraction import subtract_bias

    subtract_bias(input_data=data, bias_data=bias)

To propagate the bias calibration's data quality, pass both DQ arrays. The bias DQ is OR'ed into ``dq``:

.. code-block:: python

    bias_dq = np.zeros((Nrows, Ncols), dtype=np.uint32)
    subtract_bias(input_data=data, bias_data=bias, input_dq=dq, bias_dq=bias_dq)
