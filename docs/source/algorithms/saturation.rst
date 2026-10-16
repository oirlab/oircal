Saturation
==========

Flagging Saturation
-------------------

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

Make Saturation
---------------

:func:`.make_saturation` makes the threshold map from the last reads of ramps that reach saturation: the median of the last reads, minus ``offset`` (100 DN by default) to allow for read noise and other systematics.

The threshold is computed as the median over several exposures. The example simulates five exposures of a source bright enough to saturate every pixel. :func:`.create_ramp` clips the counts at 65535 DN, which stands in for the detector's saturation level, and the last read of each exposure is kept:

.. code-block:: python

    from oircal.saturation import make_saturation

    last_reads = []
    for seed in range(5):
        sat_ramp = create_ramp(
            source=np.full((Nrows, Ncols), 5e5),  # e-/s
            readtime=readtime,
            Nreads=Nreads,
            gain=gain,
            bias=bias * gain,
            read_noise=read_noise * gain,
            seed=seed,
        )
        last_reads.append(sat_ramp["data"][-1])
    last_reads = np.array(last_reads)  # (5, Nrows, Ncols)

    sat_thresh = make_saturation(last_reads)  # 65535 - 100 DN everywhere
