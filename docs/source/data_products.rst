Data Products
=============

Methods in oircal work directly with NumPy arrays.

- Up-the-ramp (UTR) data are 3D arrays of shape ``(Nreads, Nrows, Ncols)``, with a matching per-read data quality array ``dq_raw`` (typically ``uint8``).

- Per-pixel images (rates, calibrations, data quality) are 2D arrays of shape ``(Nrows, Ncols)``. The per-pixel data quality array ``dq`` is typically ``uint32``.

- Ramps are in DN and are converted to e- (or e-/s) with a gain correction after ramp fitting. Gain is in e-/DN, so the correction is multiplicative.

- Read noise is the single-read standard deviation in DN.

Algorithms use or track the following arrays:

- UTR data: ``data`` and ``dq_raw``.
- Rate maps: ``data``, ``var_poisson``, ``var_rnoise`` and ``dq``.

Ramp fitting also returns ``mid_time``, the midpoint in time of each pixel's valid reads (MJD if the start time of the first read is given).

Data Quality Flags
------------------

There are two DQ arrays: ``dq_raw`` (``uint8``) and ``dq`` (``uint32``).

Functions that read or set data quality (DQ) flags take a ``dq_flags`` argument: a dict mapping each flag name to its bit value. Only the flags a function uses are required (see each function's docstring).

.. code-block:: python

    dq_flags = {"DO_NOT_USE": 2**0, "SATURATED": 2**1, "JUMP_DET": 2**2}

.. note::

    Functions convert the dict to a named tuple internally with :func:`.dq_flags_to_namedtuple`, since Numba handles named tuples more efficiently than its typed dict.

Flags used by ``oircal``
~~~~~~~~~~~~~~~~~~~~~~~~

The flags read by each function:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Flags
   * - :func:`.check_saturation`
     - ``SATURATED``, ``DO_NOT_USE``
   * - :func:`.correct_nonlinearity`
     - ``NON_LINEAR``, ``DO_NOT_USE``
   * - :func:`.fit_nonlin_coefs`
     - ``DO_NOT_USE``, ``SATURATED``, ``JUMP_DET``
   * - :func:`.jump_detection_difference`
     - ``DO_NOT_USE``, ``SATURATED``, ``JUMP_DET``
   * - :func:`.jump_detection_jwst`
     - ``GOOD``, ``DO_NOT_USE``, ``SATURATED``, ``JUMP_DET``, ``UNRELIABLE_FLAT``, ``REFERENCE_PIXEL``
   * - :func:`.fit_ramps_ols`, :func:`.fit_ramps_mcds`
     - ``DO_NOT_USE``, ``SATURATED``, ``JUMP_DET``
   * - :func:`.coadd`, :func:`.coadd_spectra`
     - ``DO_NOT_USE``
   * - :func:`.fit_ramps_jwst`
     - ``DO_NOT_USE``, ``SATURATED``, ``JUMP_DET``, ``UNRELIABLE_FLAT``, ``UNRELIABLE_SLOPE``, ``OUTLIER`` (plus ``GOOD`` and ``REFERENCE_PIXEL`` with ``jump_detection=True``)

A single dict with every flag can be used everywhere:

.. code-block:: python

    dq_flags = {
        "GOOD": 0,
        "DO_NOT_USE": 2**0,
        "SATURATED": 2**1,
        "JUMP_DET": 2**2,
        "NON_LINEAR": 2**3,
        "OUTLIER": 2**4,
        "UNRELIABLE_SLOPE": 2**5,
        "UNRELIABLE_FLAT": 2**6,
        "REFERENCE_PIXEL": 2**7,
    }
