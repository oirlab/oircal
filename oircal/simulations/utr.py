import ctypes
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray

import numpy as np
from progressbar import progressbar

from ..utils.parallelization_utils import shared_array_to_numpy


__all__ = [
    "create_ramp",
    "generate_1overf_noise",
    "generate_1overf_noise_map",
    "utr_variable_rate_cube",
]

# np.random.poisson(lam) fails when lam is too large; use normal approximation above this
_POISSON_LAM_MAX = 1e8


def _poisson_like_draw(lam: np.ndarray, size: tuple) -> np.ndarray:
    """Draw counts from Poisson where lam is small; use Normal(lam, sqrt(lam)) for large lam."""
    lam = np.asarray(lam, dtype=np.float64)
    out = np.empty(size, dtype=np.float64)
    small = lam <= _POISSON_LAM_MAX
    large = ~small
    if np.any(small):
        out[small] = np.random.poisson(lam=lam[small])
    if np.any(large):
        mu, sig = lam[large], np.sqrt(lam[large])
        out[large] = np.clip(np.round(np.random.normal(mu, sig)), 0.0, None)
    return out


# NOTE: Find a new home for this function


def _create_ramp(
    data: np.ndarray,
    dq_raw: np.ndarray,
    source: np.ndarray,  # e- / s / pixel including all sources
    readtime: float = 1.7,  # seconds between reads
    Nreads: int = 10,
    gain: np.ndarray | float = 1.0,  # e-/ADU
    flat: np.ndarray | float = 1.0,
    bias: np.ndarray | float = 1000,  # e-
    dark: np.ndarray | float = 0,  # e-/s/pixel
    read_noise: np.ndarray | float = 3,  # e- standard deviation
    kTC_noise: np.ndarray | float = 50,  # e- standard deviation, None to skip
    nonlin_coeffs: np.ndarray | None = None,
    poisson_noise: bool = True,
    mjd_start: float = 60577.0,  # MJD start time of first read of first pixel
    convert_to_uint16: bool = True,
    clip_ramps: bool = True,
    flux_scaling: np.ndarray | None = None,
    flux_scaling_times: np.ndarray | None = None,
) -> dict[str, np.ndarray]:

    if convert_to_uint16:
        _data = np.zeros(shape=(Nreads, source.shape[0], source.shape[1]), dtype=np.float32)
    else:
        _data = data

    if flux_scaling is not None and flux_scaling_times is not None:
        _source, s_mean = utr_variable_rate_cube(
            source,
            flux_scaling,
            flux_scaling_times,
            Nreads=Nreads,
            readtime=readtime,
            t0=0.0,
        )
    else:
        _source = np.tile(source[None, :, :], (Nreads - 1, 1, 1))  # shape (Nreads-1, Nrows, Ncols)

    # First, compute photon counts only (no read noise or bias)
    for i in np.arange(1, Nreads):
        if poisson_noise:
            lam_source = np.asarray(_source[i - 1] * readtime, dtype=np.float64)
            lam_source = np.nan_to_num(lam_source, nan=0.0, posinf=0.0, neginf=0.0)
            lam_source = np.maximum(lam_source, 0.0)
            lam_dark = np.asarray(dark * readtime, dtype=np.float64)
            lam_dark = np.nan_to_num(lam_dark, nan=0.0, posinf=0.0, neginf=0.0)
            lam_dark = np.maximum(lam_dark, 0.0)
            draw_source = _poisson_like_draw(lam_source, source.shape)
            draw_dark = _poisson_like_draw(lam_dark, source.shape)
            _data[i, :, :] = _data[i - 1, :, :] + (draw_source * flat + draw_dark) / gain
        else:
            _data[i, :, :] = (
                _data[i - 1, :, :] + (_source[i - 1] * readtime * flat + dark * readtime) / gain
            )

    # apply non-linear behavior
    if nonlin_coeffs is not None:
        _data[:] = np.polyval(nonlin_coeffs, _data)

    # Add read noise
    if read_noise is not None:
        _data += np.random.normal(loc=0, scale=read_noise / gain, size=_data.shape)

    # Add bias (and kTC noise if desired)
    _data += bias / gain
    if kTC_noise is not None:
        _data += np.random.normal(loc=0, scale=kTC_noise / gain, size=source.shape)[None, :, :]

    if clip_ramps:
        # Clip _data to maximum value of uint16
        _data[:] = np.clip(_data, 0, float(np.iinfo(np.uint16).max))

    if convert_to_uint16:
        # Check for negative values before returning
        if np.any(_data < 0):
            raise ValueError(
                "Negative value found in 'data' array before returning from create_ramp. "
                "Data needs to be all positive to be converted to np.uint16."
                "Suggestion: Raise bias value.",
            )

        # Round and convert to uint16
        data[:] = np.round(_data).astype(np.uint16)

    return


def _create_ramp_worker(args: tuple):
    """
    Worker function for multiprocessing. Runs create_ramp on a block of rows.
    Args:
        args (tuple): (source_block, readtime, Nreads, gain_block, flat_block, bias_block, dark_block, read_noise,
        kTC_noise, nonlin_coeffs, poisson_noise, mjd_start, convert_to_uint16, clip_ramps, types_tuple,myrandomseed,
        flux_scaling,flux_scaling_times)
    Returns:
        dict[str, np.ndarray]: Output from create_ramp for the block
    """
    (
        block_start,
        block_end,
        readtime,
        Nreads,
        nonlin_coeffs,
        poisson_noise,
        mjd_start,
        convert_to_uint16,
        clip_ramps,
        types_tuple,
        myrandomseed,
        flux_scaling,
        flux_scaling_times,
    ) = args
    mp_data_type, mp_float_type, mp_dq_raw_type = types_tuple

    np.random.seed(myrandomseed)

    data_np = shared_array_to_numpy(shared_data, shared_data_shape, dtype=mp_data_type)
    dq_raw_np = shared_array_to_numpy(shared_dq_raw, shared_dq_raw_shape, dtype=mp_dq_raw_type)
    source_np = shared_array_to_numpy(shared_source, shared_source_shape, dtype=mp_float_type)
    gain_np = shared_array_to_numpy(shared_gain, shared_gain_shape, dtype=mp_float_type)
    flat_np = shared_array_to_numpy(shared_flat, shared_flat_shape, dtype=mp_float_type)
    bias_np = shared_array_to_numpy(shared_bias, shared_bias_shape, dtype=mp_float_type)
    dark_np = shared_array_to_numpy(shared_dark, shared_dark_shape, dtype=mp_float_type)
    read_noise_np = shared_array_to_numpy(
        shared_read_noise,
        shared_read_noise_shape,
        dtype=mp_float_type,
    )
    kTC_noise_np = shared_array_to_numpy(
        shared_kTC_noise,
        shared_kTC_noise_shape,
        dtype=mp_float_type,
    )

    _create_ramp(
        data_np[:, block_start:block_end, :],
        dq_raw_np[:, block_start:block_end, :],
        source=source_np[block_start:block_end, :],
        readtime=readtime,
        Nreads=Nreads,
        gain=gain_np[block_start:block_end, :],
        flat=flat_np[block_start:block_end, :],
        bias=bias_np[block_start:block_end, :],
        dark=dark_np[block_start:block_end, :],
        read_noise=read_noise_np[block_start:block_end, :],
        kTC_noise=kTC_noise_np[block_start:block_end, :],
        nonlin_coeffs=nonlin_coeffs,
        poisson_noise=poisson_noise,
        mjd_start=mjd_start,
        convert_to_uint16=convert_to_uint16,
        clip_ramps=clip_ramps,
        flux_scaling=flux_scaling,
        flux_scaling_times=flux_scaling_times,
    )

    return (block_start, block_end)


def create_ramp(
    source: np.ndarray,
    readtime: float = 1.7,
    Nreads: int = 10,
    gain: np.ndarray | float | None = 1.0,
    flat: np.ndarray | float | None = None,
    bias: np.ndarray | float | None = 1000,
    dark: np.ndarray | float | None = 0,
    read_noise: np.ndarray | float | None = 3,
    kTC_noise: np.ndarray | float | None = 50,
    nonlin_coeffs: np.ndarray | None = None,
    poisson_noise: bool = True,
    mjd_start: float = 60577.0,
    convert_to_uint16: bool = True,
    clip_ramps: bool = True,
    max_cores: int = 1,
    block_size: int | None = None,
    std_1overf: float | None = None,  # e-
    n_channels: int | None = None,
    vertical: bool = True,
    flux_scaling: np.ndarray | None = None,
    flux_scaling_times: np.ndarray | None = None,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Simulate a detector up-the-ramp (UTR) readout sequence for a given source image.

    Parameters
    ----------
    source
        2D array of shape (Nrows, Ncols) of source signal (e-/s/pixel) including all sources.
    readtime
        Time between reads (seconds). Default is 1.7.
    Nreads
        Number of reads in the ramp sequence. Default is 10.
    gain
        Gain of the detector (e-/ADU). A scalar or a 2D array of shape (Nrows, Ncols). Default is
        1.0.
    flat
        Flat field. A scalar or a 2D array of shape (Nrows, Ncols).
        We assume that flats have been corrected for gain variations already. In this function the gain is applied after the flat, but in the data reduction pipeline the gain correction should be applied first.
        Default is None, i.e. 1.
    bias
        Bias level to add to the first read. A scalar or a 2D array of shape (Nrows, Ncols).
        Default is 1000 e-.
    dark
        Dark current level to add to each read (e-/s/pixel). A scalar or a 2D array of shape (Nrows, Ncols).
        The flat field is applied BEFORE adding the dark current. This is because dark subtraction is typically done before the flat fielding step.
        Default is 0.
    read_noise
        Standard deviation of read noise (e-) for each read. A scalar or a 2D array of shape (Nrows, Ncols).
        Default is 3 e-.
    kTC_noise
        kTC noise parameter (e-) to add to the first read. A scalar or a 2D array of shape (Nrows,
        Ncols).
        Default is 50 e-.
    nonlin_coeffs
        1D array of shape (n_coeffs,) of nonlinearity coefficients, highest degree first. Applies as data = np.polyval(nonlin_coeffs, data).
        Default is None.
    poisson_noise
        If True, apply Poisson noise to the signal. Default is True.
    mjd_start
        Modified Julian Date for the start time. Default is 60577.0.
    convert_to_uint16
        If True, round and convert the output data to np.uint16. Default is True.
    clip_ramps
        If True, clip_ramps the output data to the range of np.uint16 (0 to 65535). Default is True.
    max_cores
        Maximum number of CPU cores to use for parallel processing. Default is 1.
    block_size
        Number of rows to process in each block when using multiple cores. Default is None.
    std_1overf
        Standard deviation of 1/f noise in ADU (ie, DN). If None, 1/f noise is not added.
        Default is None.
    n_channels
        Number of readout channels. If None, assume 1 channel. Default is None.
    vertical
        If True, channels are vertical; if False, channels are horizontal. Default is True.
    flux_scaling
        1D array of shape (n_times,) of flux timeseries to model variability centered on unity.
        Default is None.
    flux_scaling_times
        1D array of shape (n_times,) of times corresponding to flux_scaling (seconds).
        Default is None.
    seed
        RNG seed for Poisson / read / kTC / 1/f noise. When set, repeated calls
        with the same inputs reproduce the same ramp (needed for a packaged WAVE
        CCF bake to transfer across notebook runs). Default is None (nondeterministic).

    Returns
    -------
    output : dict
        Dictionary with keys:

        - 'data': 3D array of shape (Nreads, Nrows, Ncols) of ramps (DN); uint16 if
          `convert_to_uint16`, else float32.
        - 'dq_raw': 3D uint8 array of shape (Nreads, Nrows, Ncols) of per-read DQ flags.
        - 'meta': dict with the readout 'clock_rate' (Hz), 'mjd_start', 'n_channels',
          'channels_are_vertical' and 'n_refpixs'.

    Raises
    ------
    ValueError
        If any value in the output 'data' array is negative.
    """
    Nrows, Ncols = source.shape
    rng_master = np.random.default_rng(seed)

    if gain is None:
        gain = 1.0
    if flat is None:
        flat = 1.0
    if bias is None:
        bias = 1000
    if dark is None:
        dark = 0
    if read_noise is None:
        read_noise = 0
    if kTC_noise is None:
        kTC_noise = 0

    # Run in parallel
    if max_cores > 1:
        mp_data_type = ctypes.c_ushort if convert_to_uint16 else ctypes.c_float
        mp_float_type = ctypes.c_float
        mp_dq_raw_type = ctypes.c_uint8

        types_tuple = (mp_data_type, mp_float_type, mp_dq_raw_type)

        data_mp = RawArray(mp_data_type, Ncols * Nrows * Nreads)
        data_shape = (Nreads, Nrows, Ncols)
        data_np = shared_array_to_numpy(data_mp, data_shape, dtype=mp_data_type)

        dq_raw_mp = RawArray(mp_dq_raw_type, Ncols * Nrows * Nreads)
        dq_raw_shape = (Nreads, Nrows, Ncols)
        dq_raw_np = shared_array_to_numpy(dq_raw_mp, dq_raw_shape, dtype=mp_dq_raw_type)

        source_mp = RawArray(mp_float_type, Ncols * Nrows)
        source_shape = (Nrows, Ncols)
        source_np = shared_array_to_numpy(source_mp, source_shape, dtype=mp_float_type)
        source_np[:] = source

        gain_mp = RawArray(mp_float_type, Ncols * Nrows)
        gain_shape = (Nrows, Ncols)
        gain_np = shared_array_to_numpy(gain_mp, gain_shape, dtype=mp_float_type)
        if not isinstance(gain, np.ndarray):
            gain_np[:] = np.full(source.shape, gain, dtype=np.float32)
        else:
            gain_np[:] = gain.astype(np.float32)

        flat_mp = RawArray(mp_float_type, Ncols * Nrows)
        flat_shape = (Nrows, Ncols)
        flat_np = shared_array_to_numpy(flat_mp, flat_shape, dtype=mp_float_type)
        if not isinstance(flat, np.ndarray):
            flat_np[:] = np.full(source.shape, flat, dtype=np.float32)
        else:
            flat_np[:] = flat

        bias_mp = RawArray(mp_float_type, Ncols * Nrows)
        bias_shape = (Nrows, Ncols)
        bias_np = shared_array_to_numpy(bias_mp, bias_shape, dtype=mp_float_type)
        if not isinstance(bias, np.ndarray):
            bias_np[:] = np.full(source.shape, bias, dtype=np.float32)
        else:
            bias_np[:] = bias

        dark_mp = RawArray(mp_float_type, Ncols * Nrows)
        dark_shape = (Nrows, Ncols)
        dark_np = shared_array_to_numpy(dark_mp, dark_shape, dtype=mp_float_type)
        if not isinstance(dark, np.ndarray):
            dark_np[:] = np.full(source.shape, dark, dtype=np.float32)
        else:
            dark_np[:] = dark

        read_noise_mp = RawArray(mp_float_type, Ncols * Nrows)
        read_noise_shape = (Nrows, Ncols)
        read_noise_np = shared_array_to_numpy(read_noise_mp, read_noise_shape, dtype=mp_float_type)
        if not isinstance(read_noise, np.ndarray):
            read_noise_np[:] = np.full(source.shape, read_noise, dtype=np.float32)
        else:
            read_noise_np[:] = read_noise

        kTC_noise_mp = RawArray(mp_float_type, Ncols * Nrows)
        kTC_noise_shape = (Nrows, Ncols)
        kTC_noise_np = shared_array_to_numpy(kTC_noise_mp, kTC_noise_shape, dtype=mp_float_type)
        if not isinstance(kTC_noise, np.ndarray):
            kTC_noise_np[:] = np.full(source.shape, kTC_noise, dtype=np.float32)
        else:
            kTC_noise_np[:] = kTC_noise

        if block_size is None:
            block_size = max(1, Nrows // (4 * max_cores)) if max_cores > 1 else Nrows

        # Prepare blocks
        args_list = []
        for block_start in range(0, Nrows, block_size):
            block_end = min(block_start + block_size, Nrows)

            # Distinct per-block seeds; derived from ``seed`` when provided so the
            # full ramp is reproducible across runs / process counts.
            myrandomseed = int(
                rng_master.integers(low=0, high=np.iinfo(np.uint32).max, dtype=np.uint32),
            )

            args = (
                block_start,
                block_end,
                readtime,
                Nreads,
                nonlin_coeffs,
                poisson_noise,
                mjd_start,
                convert_to_uint16,
                clip_ramps,
                types_tuple,
                myrandomseed,
                flux_scaling,
                flux_scaling_times,
            )
            args_list.append(args)

        _init_args = (
            data_mp,
            data_shape,
            dq_raw_mp,
            dq_raw_shape,
            source_mp,
            source_shape,
            gain_mp,
            gain_shape,
            flat_mp,
            flat_shape,
            bias_mp,
            bias_shape,
            dark_mp,
            dark_shape,
            read_noise_mp,
            read_noise_shape,
            kTC_noise_mp,
            kTC_noise_shape,
        )
        tpool = mp.Pool(
            processes=max_cores,
            initializer=_tpool_init,
            initargs=_init_args,
            maxtasksperchild=50,
        )
        tasks = [tpool.apply_async(_create_ramp_worker, args=(_args,)) for _args in args_list]

        for t in progressbar(tasks, desc="Creating Ramp", fast=False):
            t.wait()

        tpool.close()
        tpool.join()

    else:
        data_np = np.zeros(shape=(Nreads, source.shape[0], source.shape[1]), dtype=np.float32)
        dq_raw_np = np.zeros(shape=(Nreads, source.shape[0], source.shape[1]), dtype=np.uint8)
        # Seed the global Generator used by _create_ramp (np.random.*) for RN/kTC/Poisson.
        if seed is not None:
            np.random.seed(int(seed))
        _outputs = _create_ramp(
            data_np,
            dq_raw_np,
            source=source,
            readtime=readtime,
            Nreads=Nreads,
            gain=gain,
            flat=flat,
            bias=bias,
            dark=dark,
            read_noise=read_noise,
            kTC_noise=kTC_noise,
            nonlin_coeffs=nonlin_coeffs,
            poisson_noise=poisson_noise,
            mjd_start=mjd_start,
            convert_to_uint16=convert_to_uint16,
            clip_ramps=clip_ramps,
            flux_scaling=flux_scaling,
            flux_scaling_times=flux_scaling_times,
        )

    if n_channels is None:
        n_channels = 1

    clock_rate = 1 / (readtime / ((Ncols * Nrows) // n_channels))  # in Hz
    _meta = {
        "clock_rate": clock_rate,
        "mjd_start": mjd_start,
        "n_channels": n_channels,
        "channels_are_vertical": vertical,
        "n_refpixs": 4,
    }

    if std_1overf is not None and std_1overf > 0:
        # 1/f is continuous; add in float space even if the shared buffer is ushort.
        data_np = np.asarray(data_np, dtype=np.float32) + generate_1overf_noise_map(
            data_np.shape,
            n_channels,
            vertical,
            clock_rate,
            std_1overf,
        )

    if convert_to_uint16:
        # Serial path historically did data[:] = round(...).astype(uint16) into a
        # float32 buffer, which silently cast back to float32. Finalize as true
        # uint16 after optional 1/f so callers match flight ADU dtype.
        data_f = np.asarray(data_np, dtype=np.float64)
        if np.any(data_f < 0):
            raise ValueError(
                "Negative value found in 'data' array before uint16 conversion. "
                "Suggestion: Raise bias value.",
            )
        data_np = np.clip(
            np.round(data_f),
            0,
            float(np.iinfo(np.uint16).max),
        ).astype(np.uint16)

    _outputs = {"data": data_np, "dq_raw": dq_raw_np, "meta": _meta}

    return _outputs


def _tpool_init(
    data: ctypes.Array,
    data_shape: tuple,
    dq_raw: ctypes.Array,
    dq_raw_shape: tuple,
    source: ctypes.Array,
    source_shape: tuple,
    gain: ctypes.Array,
    gain_shape: tuple,
    flat: ctypes.Array,
    flat_shape: tuple,
    bias: ctypes.Array,
    bias_shape: tuple,
    dark: ctypes.Array,
    dark_shape: tuple,
    read_noise: ctypes.Array,
    read_noise_shape: tuple,
    kTC_noise: ctypes.Array,
    kTC_noise_shape: tuple,
):
    """
    Initializer function for the thread pool that initializes various shared variables. Main things to note that all
    except the shapes are shared arrays (mp.Array).
    """
    global \
        shared_data, \
        shared_data_shape, \
        shared_dq_raw, \
        shared_dq_raw_shape, \
        shared_source, \
        shared_source_shape, \
        shared_gain, \
        shared_gain_shape, \
        shared_flat, \
        shared_flat_shape, \
        shared_bias, \
        shared_bias_shape, \
        shared_dark, \
        shared_dark_shape, \
        shared_read_noise, \
        shared_read_noise_shape, \
        shared_kTC_noise, \
        shared_kTC_noise_shape

    shared_data = data
    shared_data_shape = data_shape

    shared_dq_raw = dq_raw
    shared_dq_raw_shape = dq_raw_shape

    shared_source = source
    shared_source_shape = source_shape

    shared_gain = gain
    shared_gain_shape = gain_shape

    shared_flat = flat
    shared_flat_shape = flat_shape

    shared_bias = bias
    shared_bias_shape = bias_shape

    shared_dark = dark
    shared_dark_shape = dark_shape

    shared_read_noise = read_noise
    shared_read_noise_shape = read_noise_shape

    shared_kTC_noise = kTC_noise
    shared_kTC_noise_shape = kTC_noise_shape


def generate_1overf_noise_map(
    shape_3d: tuple,
    n_channels: int = 1,
    vertical: bool = True,
    clock_rate: float = 1.0,
    std_1overf: float = 1.0,
    seed: int | None = None,
):
    """
    Generate a 3D array of 1/f (pink) noise assuming same noise in each readout channel.

    Parameters
    ----------
    shape_3d
        Shape of the output array (Nreads, Nrows, Ncols).
    n_channels
        Number of readout channels. Default is 1.
    vertical
        If True, channels are vertical; if False, channels are horizontal. Default is True.
    clock_rate
        Clock rate in Hz (samples per second). Default is 1.0.
    std_1overf
        Desired standard deviation of the output time series. Default is 1.0.
    seed
        Random seed for reproducibility. Default is None.

    """
    Nreads, Nrows, Ncols = shape_3d

    if vertical:
        if Ncols % n_channels != 0:
            raise ValueError(
                "Ncols is not divisible by n_channels. Detector size and number of channels are incompatible.",
            )
        nx_chan = Ncols // n_channels
        ny_chan = Nrows
    else:
        if Nrows % n_channels != 0:
            raise ValueError(
                "Nrows is not divisible by n_channels. Detector size and number of channels are incompatible.",
            )
        nx_chan = Nrows // n_channels
        ny_chan = Ncols

    noise_1overf_channel = generate_1overf_noise(
        n=(Nreads * nx_chan * ny_chan),
        dt=1 / clock_rate,
        std=std_1overf,
        seed=seed,
    )
    noise_1overf_channel = noise_1overf_channel.reshape(Nreads, ny_chan, nx_chan)

    noise_1overf = np.tile(noise_1overf_channel, (1, 1, n_channels))

    if vertical:
        return noise_1overf
    else:
        return noise_1overf.transpose(0, 2, 1)


def generate_1overf_noise(
    n: int,
    dt: float = 1.0,
    std: float = 1.0,
    seed: int | None = None,
    return_time: bool = False,
):
    """
    Generate a 1/f (pink) noise time series using frequency-domain synthesis.

    Notes
    -----
    Generated from ChatGPT.

    Parameters
    ----------
    n
        Number of samples in the time series.
    dt
        Time sampling interval. Default is 1.0.
    std
        Desired standard deviation of the output time series. Default is 1.0.
    seed
        Random seed for reproducibility. Default is None.
    return_time
        If True, also return the time sampling vector. Default is False.

    Returns
    -------
    noise : np.ndarray
        1D array of shape (n,); real-valued 1/f noise time series.
    t : np.ndarray, optional
        1D array of shape (n,); time sampling vector (returned if return_time=True).
    """
    rng = np.random.default_rng(seed)

    # Fourier frequencies (positive only, real FFT)
    freqs = np.fft.rfftfreq(n, d=dt)

    # Avoid division by zero at DC
    scale = np.zeros_like(freqs)
    nonzero = freqs > 0
    scale[nonzero] = 1.0 / np.sqrt(freqs[nonzero])

    # Random complex spectrum
    real = rng.normal(size=freqs.size)
    imag = rng.normal(size=freqs.size)
    spectrum = (real + 1j * imag) * scale

    # Enforce zero DC component
    spectrum[0] = 0.0

    # Inverse FFT to time domain
    noise = np.fft.irfft(spectrum, n=n)

    # Normalize to requested standard deviation
    noise -= np.mean(noise)
    noise *= std / np.std(noise)

    if return_time:
        t = np.arange(n) * dt
        return noise, t

    return noise


def utr_variable_rate_cube(
    source: np.ndarray,
    flux_scaling: np.ndarray,
    flux_scaling_times: np.ndarray,
    Nreads: int,
    readtime: float,
    t0: float = 0.0,
):
    """
    Build a (Nreads-1, Nrows, Ncols) cube of per-read *average* electron rates
    for a time-variable UTR ramp, conserving total flux.

    Parameters
    ----------
    source
        2D array of shape (Nrows, Ncols) of average electron rate (e-/s), averaged over the full
        exposure.
    flux_scaling
        1D array of shape (nt,) of dimensionless flux scaling vs time (mean ~ 1).
    flux_scaling_times
        1D array of shape (nt,) of time samples in seconds (assumed monotonic).
    Nreads
        Total number of reads (including the first zero-count read).
    readtime
        Time between reads in seconds.
    t0
        Start time of the ramp. Default is 0.0.

    Returns
    -------
    rate_cube : np.ndarray
        3D array of shape (Nreads-1, Nrows, Ncols); average e-/s rate map for each read interval.
    scaling_mean_per_read : np.ndarray
        1D array of shape (Nreads-1,); mean flux scaling applied during each interval.
    """

    if source.ndim != 2:
        raise ValueError("source must be 2D (Nrows, Ncols).")
    if flux_scaling.ndim != 1:
        raise ValueError("flux_scaling must be 1D.")
    if flux_scaling_times.ndim != 1:
        raise ValueError("flux_scaling_times must be 1D.")
    if flux_scaling.size != flux_scaling_times.size:
        raise ValueError("flux_scaling and flux_scaling_times must match in size.")
    if Nreads < 2:
        raise ValueError("Nreads must be >= 2.")

    Nrows, Ncols = source.shape
    T = (Nreads - 1) * readtime

    # Read boundaries
    read_edges = t0 + np.arange(Nreads) * readtime

    # Integration grid: union of read edges and provided samples
    t_grid = np.unique(
        np.concatenate([read_edges, flux_scaling_times]),
    )

    # Restrict to the modeled exposure window
    mask = (t_grid >= t0) & (t_grid <= t0 + T)
    t_grid = t_grid[mask]

    # Interpolate scaling onto integration grid
    scaling_grid = np.interp(
        t_grid,
        flux_scaling_times,
        flux_scaling,
        left=flux_scaling[0],
        right=flux_scaling[-1],
    )

    # Integral over each read interval
    interval_integrals = np.empty(Nreads - 1, dtype=float)

    for i in range(Nreads - 1):
        t_lo = read_edges[i]
        t_hi = read_edges[i + 1]

        sel = (t_grid >= t_lo) & (t_grid <= t_hi)
        interval_integrals[i] = np.trapezoid(
            scaling_grid[sel],
            t_grid[sel],
        )

    # Enforce flux conservation over the whole exposure
    total_integral = np.sum(interval_integrals)
    norm = T / total_integral

    # Mean scaling per read
    scaling_mean_per_read = norm * interval_integrals / readtime

    # Build rate cube
    rate_cube = source[None, :, :] * scaling_mean_per_read[:, None, None]

    return rate_cube, scaling_mean_per_read
