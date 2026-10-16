Data Products
=============

Methods in oircal work directly with NumPy arrays. Since most algorithms are implemented in Numba, the arrays must use little-endian byte order. oircal does not convert inputs itself, so arrays in big-endian order, such as data read from FITS files, must first be converted with :func:`.convert_endianness`.


Up-the-ramp (UTR)
-----------------

- Up-the-ramp (UTR) data are 3D arrays of shape ``(Nreads, Nrows, Ncols)``. The data is of type ``float32``.

- Ramps are in DN and are converted to e- (or e-/s) with a gain correction (:func:`.correct_gain`) after ramp fitting. Gain is in e-/DN, so the correction is multiplicative.

- UTR data use a per-read data quality array ``dq_raw`` of the same shape, typically ``uint8``. See :ref:`data_quality_flags` below.


Rate Maps
---------

- After ramp fitting, the output is a 2D rate map of shape ``(Nrows, Ncols)`` of type ``float32`` in units of DN/s. If gain correction is then applied, the units are e-/s.

- The ramp fitting function computes three error arrays:

  - ``rate_err``: The total error in the fitted rate.
  - ``rate_var_poisson``: The variance contribution from Poisson noise.
  - ``rate_var_rnoise``: The variance contribution from read noise.

- The ramp fitting function also computes a 2D data quality array ``dq`` of shape ``(Nrows, Ncols)``, typically ``uint32``. See :ref:`data_quality_flags` below.

- ``mid_time`` is a 2D array of shape ``(Nrows, Ncols)`` of type ``float64`` with the midpoint in time of each pixel's valid reads, in MJD if the start time of each pixel's first read is given (see :func:`.get_first_read_mjd`). See :ref:`the mid time in ramp fitting <mid_time>`.


Calibration Data
----------------

Each calibration is a set of arrays, named here as in the arguments of the functions that make and use them. Per-pixel arrays have shape ``(Nrows, Ncols)``, and ``*_dq`` arrays are ``uint32`` DQ flags of the calibration.

- **Bias**

  - ``bias_data``: the bias in DN.
  - ``bias_dq``: DQ flags.
  - ``kTC``: the kTC (reset) noise in DN, which includes the read noise of one read, from :func:`.make_bias`.

- **Saturation**

  - ``sat_thresh``: the saturation level in DN.
  - ``sat_dq``: DQ flags.

- **Nonlinearity**

  - ``coeffs``: array of shape ``(Ncoeff, Nrows, Ncols)`` containing the polynomial coefficients for the nonlinearity correction. The first axis is the polynomial order, and coefficients are ordered from lowest to highest order.
  - ``nonlin_thresh``: the level in DN (first read subtracted) where counts deviate from linear by 1% (``criterion_for_nonlin_thresh``), from :func:`.fit_nonlin_coefs`.
  - ``nonlin_max``: the level in DN where counts deviate from linear by 10% (``criterion_for_nonlin_max``), from :func:`.fit_nonlin_coefs`. :func:`.correct_nonlinearity` flags reads above it.
  - ``nonlin_dq``: DQ flags.

- **Read noise**

  - ``rn``: the single-read standard deviation in DN (``rdnoise`` in the ramp fitting functions).

- **Gain**

  - ``gain``: the gain in e-/DN, so the correction is multiplicative.
  - ``cov``: array of shape ``(2, 2, Nrows, Ncols)``, the covariance of the gain and read noise from :func:`.fit_gain_and_readnoise`.

- **Dark current**

  - ``dark_rate``: the dark current in DN/s.
  - ``dark_err``: its uncertainty in DN/s.
  - ``dark_dq``: DQ flags.

- **Flat field**

  - ``detflat``: the flat normalized to a median of 1. Rates are divided by it.
  - ``detflat_err``: its uncertainty.
  - ``detflat_dq``: DQ flags.


.. _data_quality_flags:

Data Quality Flags
------------------

There are two data quality (DQ) arrays:

- ``dq_raw``: Data quality flags for UTR data of shape ``(Nreads, Nrows, Ncols)`` of type ``uint8``. This allows for 8 different flags to be uniquely set for each read of each pixel.

- ``dq``: 2D data quality flags for rate maps of shape ``(Nrows, Ncols)`` of type ``uint32``. This allows for 32 different flags to be uniquely set for each pixel. This can be used with rate maps, or to assign flags for all reads for UTR data.

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
   * - :func:`.make_bias`
     - ``UNRELIABLE_BIAS``
   * - :func:`.correct_nonlinearity`
     - ``NON_LINEAR``, ``DO_NOT_USE``
   * - :func:`.fit_nonlin_coefs`
     - ``DO_NOT_USE``, ``SATURATED``, ``JUMP_DET``
   * - :func:`.jump_detection_difference`
     - ``DO_NOT_USE``, ``SATURATED``, ``JUMP_DET``
   * - :func:`.jump_detection_jwst`
     - ``GOOD``, ``DO_NOT_USE``, ``SATURATED``, ``JUMP_DET``, ``UNRELIABLE_FLAT``, ``REFERENCE_PIXEL``
   * - :func:`.fit_ramps_ols`, :func:`.fit_ramps_mcds`, :func:`.fit_gain_and_readnoise`
     - ``DO_NOT_USE``, ``SATURATED``, ``JUMP_DET``
   * - :func:`.make_dark`
     - ``HOT``
   * - :func:`.make_detector_flat`
     - ``COLD``
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
        "UNRELIABLE_BIAS": 2**8,
        "HOT": 2**9,
        "COLD": 2**10,
    }
