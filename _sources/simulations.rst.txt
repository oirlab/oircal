Simulations
===========

:mod:`oircal.simulations` simulates up-the-ramp detector data, which is useful for testing algorithms.

Simulating a ramp
-----------------

:func:`.create_ramp` builds an up-the-ramp exposure from a 2D source image in e-/s.

.. code-block:: python

    import numpy as np

    from oircal.simulations import create_ramp

    Nrows, Ncols = 64, 64
    source = np.full((Nrows, Ncols), 50.0)  # e-/s
    ramp = create_ramp(
        source=source,
        readtime=1.7,  # s between reads
        Nreads=10,
        gain=1.5,  # e-/DN
        bias=1500.0,  # e-
        dark=0.01,  # e-/s
        read_noise=7.5,  # e-
        kTC_noise=50.0,  # e-
        seed=0,
    )
    data = ramp["data"]  # (Nreads, Nrows, Ncols), DN
    meta = ramp["meta"]  # clock rate, start MJD and readout channel layout

The output ramp data is in ``DN``.

By default the output is rounded and converted to ``uint16`` and clipped to ``[0, 65535]``. Pass ``convert_to_uint16=False`` to keep floating point values. Pass ``seed`` for reproducible noise.

Detector effects
----------------

- **Flat field**: ``flat`` scales the source before the dark current is added.
- **Non-linearity**: ``nonlin_coeffs`` applies ``np.polyval(nonlin_coeffs, data)`` to the ramps (highest degree first).
- **1/f noise**: ``std_1overf`` (DN) adds correlated 1/f noise shared between ``n_channels`` readout channels (``vertical`` sets their orientation).

.. code-block:: python

    ramp = create_ramp(
        source=source,
        readtime=1.7,
        Nreads=10,
        nonlin_coeffs=[-1e-6, 1, 0],  # data - 1e-6 * data**2
        std_1overf=5.0,
        n_channels=4,
        seed=0,
    )

Variable sources
----------------

``flux_scaling`` and ``flux_scaling_times`` (s) multiply the source by a time series centered on unity, conserving the total flux. :func:`.utr_variable_rate_cube` returns the per-read rates used for this.

.. code-block:: python

    t = np.linspace(0, 20, 200)  # s
    scaling = 1 + 0.2 * np.sin(2 * np.pi * t / 10)
    ramp = create_ramp(
        source=source,
        readtime=1.7,
        Nreads=10,
        flux_scaling=scaling,
        flux_scaling_times=t,
        seed=0,
    )

1/f noise
---------

:func:`.generate_1overf_noise` generates a 1D 1/f noise time series, and :func:`.generate_1overf_noise_map` generates a 3D ``(Nreads, Nrows, Ncols)`` noise cube with the same noise in each readout channel.

.. code-block:: python

    from oircal.simulations import generate_1overf_noise_map

    noise = generate_1overf_noise_map(
        shape_3d=(10, Nrows, Ncols),
        n_channels=4,
        std_1overf=5.0,
        seed=0,
    )
