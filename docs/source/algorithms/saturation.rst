Saturation
==========

:func:`.check_saturation` compares each read to a per-pixel saturation threshold (DN). The first read at or above the threshold, and every read after it, are flagged ``SATURATED | DO_NOT_USE`` in ``dq_raw``. The DQ map from the saturation calibration (``sat_dq``) is OR'ed into ``dq``.

.. code-block:: python

    from oircal.saturation import check_saturation

    sat_thresh = np.full((Nrows, Ncols), 65535, dtype=np.uint16)  # DN
    sat_dq = np.zeros((Nrows, Ncols), dtype=np.uint32)
    check_saturation(
        input_data=data,
        input_dq_raw=dq_raw,
        input_dq=dq,
        sat_thresh=sat_thresh,
        sat_dq=sat_dq,
        dq_flags=dq_flags,
    )

Without ``dq_flags``, reads are not flagged and only ``sat_dq`` is propagated.
