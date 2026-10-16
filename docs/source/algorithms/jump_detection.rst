Jump Detection
==============

Jump detection finds reads where the counts change abruptly, e.g. from a cosmic ray hit, and flags them as ``JUMP_DET`` in ``dq_raw``. The ramp fitting functions fit the segments on either side of a flagged jump separately.

Two implementations are available.

Read differences
----------------

:func:`.jump_detection_difference` compares each difference between successive reads to the pixel's median difference, and flags reads that differ by more than ``rejection_threshold`` sigma. Reads at the end of the ramp flagged ``SATURATED`` or ``DO_NOT_USE`` are left out.

.. code-block:: python

    from oircal.jump_detection import jump_detection_difference

    jump_detection_difference(
        input_data=data,
        input_dq_raw=dq_raw,
        input_dq=dq,
        dq_flags=dq_flags,
        rdnoise=read_noise,
        gain=gain,
        rejection_threshold=4.0,
    )

JWST (stcal)
------------

:func:`.jump_detection_jwst` runs the JWST two-point difference algorithm from `stcal <https://github.com/spacetelescope/stcal>`_.

.. code-block:: python

    from oircal.jump_detection import jump_detection_jwst

    jump_detection_jwst(
        input_data=data,
        input_dq_raw=dq_raw,
        input_dq=dq,
        dq_flags=dq_flags,
        gain=gain,
        rdnoise=read_noise,
        rejection_threshold=4.0,
    )

Jump detection can also run as part of ramp fitting with ``fit_ramps_jwst(..., jump_detection=True)`` (see :doc:`ramp_fitting`).
