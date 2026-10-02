import numpy as np
import multiprocessing as mp
import ctypes
from progressbar import progressbar

from ..utils.parallelization_utils import shared_array_to_numpy
from multiprocessing.sharedctypes import RawArray

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
    data : np.ndarray, dq_raw : np.ndarray,
    source : np.ndarray, # e- / s / pixel including all sources
    readtime : float = 1.7, # seconds between reads
    n_reads : int = 10,
    gain : np.ndarray | float = 1.0, # e-/ADU
    flat : np.ndarray | float = 1.0,
    bias : np.ndarray | float = 1000, # e-
    dark : np.ndarray | float = 0, # e-/s/pixel
    read_noise : np.ndarray | float = 3, # e- standard deviation
    kTC_noise : np.ndarray | float = 50, # e- standard deviation, None to skip
    nonlin_coeffs : np.ndarray | None = None,
    poisson_noise : bool = True,
    mjd_start : float = 60577.0, # MJD start time of first read of first pixel
    convert_to_uint16 : bool = True,
    clip_ramps: bool = True,
    flux_scaling: np.ndarray | None = None,
    flux_scaling_times: np.ndarray | None = None
) -> dict[str, np.ndarray]:

    if convert_to_uint16:
        _data = np.zeros(shape=(n_reads, source.shape[0], source.shape[1]), dtype=np.float32)
    else:
        _data = data

    if flux_scaling is not None and flux_scaling_times is not None:
        _source, s_mean = utr_variable_rate_cube(
            source, flux_scaling, flux_scaling_times,
            n_reads=n_reads, readtime=readtime, t0=0.0
        )
    else:
        _source = np.tile(source[None, :, :], (n_reads-1, 1, 1))  # shape (n_reads-1, ny, nx)

    # First, compute photon counts only (no read noise or bias)
    for i in np.arange(1,n_reads):
        if poisson_noise:
            lam_source = np.asarray(_source[i - 1] * readtime, dtype=np.float64)
            lam_source = np.nan_to_num(lam_source, nan=0.0, posinf=0.0, neginf=0.0)
            lam_source = np.maximum(lam_source, 0.0)
            lam_dark = np.asarray(dark * readtime, dtype=np.float64)
            lam_dark = np.nan_to_num(lam_dark, nan=0.0, posinf=0.0, neginf=0.0)
            lam_dark = np.maximum(lam_dark, 0.0)
            draw_source = _poisson_like_draw(lam_source, source.shape)
            draw_dark = _poisson_like_draw(lam_dark, source.shape)
            _data[i, :, :] = _data[i-1, :, :] + (draw_source * flat + draw_dark) / gain
        else:
            _data[i, :, :] = _data[i-1, :, :] + (_source[i-1] * readtime *flat + dark * readtime) / gain

    # apply non-linear behavior
    if nonlin_coeffs is not None:
        _data[:] = np.polyval(nonlin_coeffs, _data)

    # Add read noise
    if read_noise is not None:
        _data += np.random.normal(loc=0, scale=read_noise/gain, size=_data.shape)

    # Add bias (and kTC noise if desired)
    _data += bias/gain
    if kTC_noise is not None:
        _data += np.random.normal(loc=0, scale=kTC_noise/gain, size=source.shape)[None, :, :]



    if clip_ramps:
        # Clip _data to maximum value of uint16
        _data[:] = np.clip(_data, 0, float(np.iinfo(np.uint16).max))

    if convert_to_uint16:
        # Check for negative values before returning
        if np.any(_data < 0):
            raise ValueError("Negative value found in 'data' array before returning from create_ramp. "
                             "Data needs to be all positive to be converted to np.uint16."
                             "Suggestion: Raise bias value.")

        # Round and convert to uint16
        data[:] = np.round(_data).astype(np.uint16)

    return


def _create_ramp_worker(args : tuple):
    """
    Worker function for multiprocessing. Runs create_ramp on a block of rows.
    Args:
        args (tuple): (source_block, readtime, n_reads, gain_block, flat_block, bias_block, dark_block, read_noise,
        kTC_noise, nonlin_coeffs, poisson_noise, mjd_start, convert_to_uint16, clip_ramps, types_tuple,myrandomseed,
        flux_scaling,flux_scaling_times)
    Returns:
        dict[str, np.ndarray]: Output from create_ramp for the block
    """
    (block_start, block_end, readtime, n_reads, nonlin_coeffs, poisson_noise, mjd_start, convert_to_uint16, clip_ramps,
     types_tuple,myrandomseed,flux_scaling,flux_scaling_times) = args
    mp_data_type, mp_float_type, mp_dq_raw_type = types_tuple

    np.random.seed(myrandomseed)

    data_np = shared_array_to_numpy(shared_data, shared_data_shape, dtype=mp_data_type)
    dq_raw_np = shared_array_to_numpy(shared_dq_raw, shared_dq_raw_shape, dtype=mp_dq_raw_type)
    source_np = shared_array_to_numpy(shared_source, shared_source_shape, dtype=mp_float_type)
    gain_np = shared_array_to_numpy(shared_gain, shared_gain_shape, dtype=mp_float_type)
    flat_np = shared_array_to_numpy(shared_flat, shared_flat_shape, dtype=mp_float_type)
    bias_np = shared_array_to_numpy(shared_bias, shared_bias_shape, dtype=mp_float_type)
    dark_np = shared_array_to_numpy(shared_dark, shared_dark_shape, dtype=mp_float_type)
    read_noise_np = shared_array_to_numpy(shared_read_noise, shared_read_noise_shape, dtype=mp_float_type)
    kTC_noise_np = shared_array_to_numpy(shared_kTC_noise, shared_kTC_noise_shape, dtype=mp_float_type)

    outputs = _create_ramp(
        data_np[:,block_start:block_end, :],
        dq_raw_np[:,block_start:block_end, :],
        source=source_np[block_start:block_end, :],
        readtime=readtime,
        n_reads=n_reads,
        gain=gain_np[block_start:block_end, :],
        flat=flat_np[block_start:block_end, :],
        bias = bias_np[block_start:block_end, :],
        dark = dark_np[block_start:block_end, :],
        read_noise = read_noise_np[block_start:block_end, :],
        kTC_noise = kTC_noise_np[block_start:block_end, :],
        nonlin_coeffs = nonlin_coeffs,
        poisson_noise = poisson_noise,
        mjd_start = mjd_start,
        convert_to_uint16 = convert_to_uint16,
        clip_ramps = clip_ramps,
        flux_scaling=flux_scaling,
        flux_scaling_times=flux_scaling_times
    )

    return (block_start, block_end)


def create_ramp(
    source : np.ndarray,
    readtime : float = 1.7,
    n_reads : int = 10,
    gain : np.ndarray | float | None = 1.0,
    flat : np.ndarray | float | None = None,
    bias : np.ndarray | float | None = 1000,
    dark : np.ndarray | float | None = 0,
    read_noise : np.ndarray | float | None = 3,
    kTC_noise : np.ndarray | float | None = 50,
    nonlin_coeffs : np.ndarray | None = None,
    poisson_noise : bool = True,
    mjd_start : float = 60577.0,
    convert_to_uint16 : bool = True,
    clip_ramps: bool = True,
    max_cores: int = 1,
    block_size: int | None = None,
    std_1overf: float | None = None, # e-
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
    source : np.ndarray
        2D array of shape (ny, nx) of source signal (e-/s/pixel) including all sources.
    readtime : float, optional
        Time between reads (seconds) (default: 1.7).
    n_reads : int, optional
        Number of reads in the ramp sequence (default: 10).
    gain : float or np.ndarray, optional
        Gain of the detector (e-/ADU). A scalar or a 2D array of shape (ny, nx) (default: 1.0).
    flat : float or np.ndarray, optional
        Flat field. A scalar or a 2D array of shape (ny, nx) (default: None, i.e. 1).
        We assume that flats have been corrected for gain variations already. In this function the gain is applied after the flat, but in the data reduction pipeline the gain correction should be applied first.
    bias : float or np.ndarray, optional
        Bias level to add to the first read. A scalar or a 2D array of shape (ny, nx) (default: 1000 e-).
    dark : float or np.ndarray, optional
        Dark current level to add to each read (e-/s/pixel). A scalar or a 2D array of shape (ny, nx) (default: 0)
        The flat field is applied BEFORE adding the dark current. This is because dark subtraction is typically done before the flat fielding step.
    read_noise : float or np.ndarray, optional
        Standard deviation of read noise (e-) for each read. A scalar or a 2D array of shape (ny, nx) (default: 3 e-).
    kTC_noise : float or np.ndarray, optional
        kTC noise parameter (e-) to add to the first read. A scalar or a 2D array of shape (ny, nx) (default: 50 e-).
    nonlin_coeffs : np.ndarray, optional
        1D array of shape (n_coeffs,) of nonlinearity coefficients, highest degree first (default: None). Applies as data = np.polyval(nonlin_coeffs, data).
    poisson_noise : bool, optional
        If True, apply Poisson noise to the signal (default: True).
    mjd_start : float, optional
        Modified Julian Date for the start time (default: 60577.0).
    convert_to_uint16 : bool, optional
        If True, round and convert the output data to np.uint16 (default: True).
    clip_ramps: bool, optional
        If True, clip_ramps the output data to the range of np.uint16 (0 to 65535) (default: True).
    max_cores : int, optional
        Maximum number of CPU cores to use for parallel processing (default: 1).
    block_size : int or None, optional
        Number of rows to process in each block when using multiple cores (default: None).
    std_1overf : float, optional
        Standard deviation of 1/f noise in ADU (ie, DN). If None, 1/f noise is not added (default: None).
    n_channels : int, optional
        Number of readout channels. If None, assume 1 channel (default: None).
    vertical : bool, optional
        If True, channels are vertical; if False, channels are horizontal (default: True).
    flux_scaling : np.ndarray, optional
        1D array of shape (n_times,) of flux timeseries to model variability centered on unity (default: None).
    flux_scaling_times : np.ndarray, optional
        1D array of shape (n_times,) of times corresponding to flux_scaling (seconds) (default: None).
    seed : int, optional
        RNG seed for Poisson / read / kTC / 1/f noise. When set, repeated calls
        with the same inputs reproduce the same ramp (needed for a packaged WAVE
        CCF bake to transfer across notebook runs). Default ``None`` (nondeterministic).

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing:
            'times': astropy.table.Table of read numbers, times, and exptimes
            'data': 3D array (n_reads, ny, nx) of ramp data
            'dq': 3D array (n_reads, ny, nx) of data quality flags

    Raises
    ------
    ValueError
        If any value in the output 'data' array is negative.
    """
    ny, nx = source.shape
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

        if convert_to_uint16:
            mp_data_type = ctypes.c_ushort
        else:
            mp_data_type = ctypes.c_float
        mp_float_type = ctypes.c_float
        mp_dq_raw_type = ctypes.c_uint8

        types_tuple = (mp_data_type, mp_float_type, mp_dq_raw_type)

        data_mp = RawArray(mp_data_type, nx * ny * n_reads)
        data_shape = (n_reads, ny, nx)
        data_np = shared_array_to_numpy(data_mp, data_shape, dtype=mp_data_type)

        dq_raw_mp = RawArray(mp_dq_raw_type, nx * ny * n_reads)
        dq_raw_shape = (n_reads, ny, nx)
        dq_raw_np = shared_array_to_numpy(dq_raw_mp, dq_raw_shape, dtype=mp_dq_raw_type)

        source_mp = RawArray(mp_float_type, nx * ny)
        source_shape = (ny, nx)
        source_np = shared_array_to_numpy(source_mp, source_shape, dtype=mp_float_type)
        source_np[:] = source

        gain_mp = RawArray(mp_float_type, nx * ny)
        gain_shape = (ny, nx)
        gain_np = shared_array_to_numpy(gain_mp, gain_shape, dtype=mp_float_type)
        if not isinstance(gain, np.ndarray):
            gain_np[:]  = np.full(source.shape, gain, dtype=np.float32)
        else:
            gain_np[:] = gain.astype(np.float32)

        flat_mp = RawArray(mp_float_type, nx * ny)
        flat_shape = (ny, nx)
        flat_np = shared_array_to_numpy(flat_mp, flat_shape, dtype=mp_float_type)
        if not isinstance(flat, np.ndarray):
            flat_np[:] = np.full(source.shape, flat, dtype=np.float32)
        else:
            flat_np[:] = flat


        bias_mp = RawArray(mp_float_type, nx * ny)
        bias_shape = (ny, nx)
        bias_np = shared_array_to_numpy(bias_mp, bias_shape, dtype=mp_float_type)
        if not isinstance(bias, np.ndarray):
            bias_np[:] = np.full(source.shape, bias, dtype=np.float32)
        else:
            bias_np[:] = bias

        dark_mp = RawArray(mp_float_type, nx * ny)
        dark_shape = (ny, nx)
        dark_np = shared_array_to_numpy(dark_mp, dark_shape, dtype=mp_float_type)
        if not isinstance(dark, np.ndarray):
            dark_np[:] = np.full(source.shape, dark, dtype=np.float32)
        else:
            dark_np[:] = dark

        read_noise_mp = RawArray(mp_float_type, nx * ny)
        read_noise_shape = (ny, nx)
        read_noise_np = shared_array_to_numpy(read_noise_mp, read_noise_shape, dtype=mp_float_type)
        if not isinstance(read_noise, np.ndarray):
            read_noise_np[:] = np.full(source.shape, read_noise, dtype=np.float32)
        else:
            read_noise_np[:] = read_noise


        kTC_noise_mp = RawArray(mp_float_type, nx * ny)
        kTC_noise_shape = (ny, nx)
        kTC_noise_np = shared_array_to_numpy(kTC_noise_mp, kTC_noise_shape, dtype=mp_float_type)
        if not isinstance(kTC_noise, np.ndarray):
            kTC_noise_np[:] = np.full(source.shape, kTC_noise, dtype=np.float32)
        else:
            kTC_noise_np[:] = kTC_noise

        if block_size is None:
            block_size = max(1, ny // (4*max_cores)) if max_cores > 1 else ny

        # Prepare blocks
        args_list = []
        for block_start in range(0, ny, block_size):
            block_end = min(block_start + block_size, ny)

            # Distinct per-block seeds; derived from ``seed`` when provided so the
            # full ramp is reproducible across runs / process counts.
            myrandomseed = int(
                rng_master.integers(low=0, high=np.iinfo(np.uint32).max, dtype=np.uint32)
            )

            args = (block_start, block_end,
                    readtime,
                    n_reads,
                    nonlin_coeffs,
                    poisson_noise,
                    mjd_start,
                    convert_to_uint16,
                    clip_ramps,
                    types_tuple,
                    myrandomseed,
                    flux_scaling,flux_scaling_times
                    )
            args_list.append(args)

        _init_args = (data_mp, data_shape,
                dq_raw_mp, dq_raw_shape,
                source_mp, source_shape,
                gain_mp, gain_shape,
                flat_mp, flat_shape,
                bias_mp, bias_shape,
                dark_mp, dark_shape,
                read_noise_mp, read_noise_shape,
                kTC_noise_mp, kTC_noise_shape
                      )
        tpool = mp.Pool(processes=max_cores, initializer=_tpool_init,
                        initargs=_init_args, maxtasksperchild=50)
        tasks = [tpool.apply_async(_create_ramp_worker, args=(_args,))
                 for _args in args_list]

        for t in progressbar(tasks, desc="Creating Ramp", fast=False):
            t.wait()

        tpool.close()
        tpool.join()

    else:
        data_np = np.zeros(shape=(n_reads, source.shape[0], source.shape[1]), dtype=np.float32)
        dq_raw_np = np.zeros(shape=(n_reads, source.shape[0], source.shape[1]), dtype=np.uint8)
        # Seed the global Generator used by _create_ramp (np.random.*) for RN/kTC/Poisson.
        if seed is not None:
            np.random.seed(int(seed))
        _outputs = _create_ramp(
            data_np, dq_raw_np,
            source=source,
            readtime=readtime,
            n_reads=n_reads,
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
            flux_scaling_times=flux_scaling_times
        )

    if n_channels is None:
        n_channels = 1

    clock_rate = 1/(readtime/((nx*ny)//n_channels)) # in Hz
    _meta = dict(clock_rate=clock_rate,
                 mjd_start=mjd_start,
                 n_channels=n_channels,
                 channels_are_vertical=vertical,
                 n_refpixs = 4)

    if std_1overf is not None  and std_1overf > 0:
        # 1/f is continuous; add in float space even if the shared buffer is ushort.
        data_np = np.asarray(data_np, dtype=np.float32) + generate_1overf_noise_map(
            data_np.shape, n_channels, vertical, clock_rate, std_1overf
        )

    if convert_to_uint16:
        # Serial path historically did data[:] = round(...).astype(uint16) into a
        # float32 buffer, which silently cast back to float32. Finalize as true
        # uint16 after optional 1/f so callers match flight ADU dtype.
        data_f = np.asarray(data_np, dtype=np.float64)
        if np.any(data_f < 0):
            raise ValueError(
                "Negative value found in 'data' array before uint16 conversion. "
                "Suggestion: Raise bias value."
            )
        data_np = np.clip(
            np.round(data_f), 0, float(np.iinfo(np.uint16).max)
        ).astype(np.uint16)

    _outputs = dict(data=data_np, dq_raw=dq_raw_np, meta=_meta)

    return _outputs


def _tpool_init(data : ctypes.Array, data_shape : tuple,
                dq_raw : ctypes.Array, dq_raw_shape : tuple,
                source : ctypes.Array, source_shape : tuple,
                gain : ctypes.Array, gain_shape : tuple,
                flat : ctypes.Array, flat_shape : tuple,
                bias : ctypes.Array, bias_shape : tuple,
                dark : ctypes.Array, dark_shape : tuple,
                read_noise : ctypes.Array, read_noise_shape : tuple,
                kTC_noise : ctypes.Array, kTC_noise_shape : tuple
                ):
    """
    Initializer function for the thread pool that initializes various shared variables. Main things to note that all
    except the shapes are shared arrays (mp.Array).
    """
    global shared_data, shared_data_shape, \
        shared_dq_raw, shared_dq_raw_shape, \
        shared_source, shared_source_shape, \
        shared_gain, shared_gain_shape, \
        shared_flat, shared_flat_shape, \
        shared_bias, shared_bias_shape, \
        shared_dark, shared_dark_shape, \
        shared_read_noise, shared_read_noise_shape, \
        shared_kTC_noise, shared_kTC_noise_shape

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
    shape_3d : tuple, n_channels : int = 1, vertical : bool = True, clock_rate : float = 1.0,
    std_1overf : float = 1.0, seed : int | None = None
):
    """
    Generate a 3D array of 1/f (pink) noise assuming same noise in each readout channel.

    Parameters
    ----------
    shape_3d : tuple of int
        Shape of the output array (n_reads, ny, nx).
    n_channels : int, optional
        Number of readout channels (default: 1).
    vertical : bool, optional
        If True (default), channels are vertical; if False, channels are horizontal.
    clock_rate : float, optional
        Clock rate in Hz (samples per second) (default: 1.0).
    std_1overf : float, optional
        Desired standard deviation of the output time series (default: 1.0).
    seed : int, optional
        Random seed for reproducibility.

    """
    n_reads, ny, nx = shape_3d

    if vertical:
        if nx % n_channels != 0:
            raise ValueError(
                "nx is not divisible by n_channels. Detector size and number of channels are incompatible.")
        nx_chan = nx // n_channels
        ny_chan = ny
    else:
        if ny % n_channels != 0:
            raise ValueError(
                "ny is not divisible by n_channels. Detector size and number of channels are incompatible.")
        nx_chan = ny // n_channels
        ny_chan = nx

    noise_1overf_channel = generate_1overf_noise(n=(n_reads * nx_chan * ny_chan),
                                                 dt=1/clock_rate,
                                                 std=std_1overf,
                                                 seed=seed)
    noise_1overf_channel = noise_1overf_channel.reshape(n_reads, ny_chan, nx_chan)

    noise_1overf = np.tile(noise_1overf_channel, (1, 1, n_channels))

    if vertical:
        return noise_1overf
    else:
        return noise_1overf.transpose(0, 2, 1)


def generate_1overf_noise(n : int, dt : float = 1.0, std : float = 1.0, seed : int | None = None, return_time : bool = False):
    """
    Generate a 1/f (pink) noise time series using frequency-domain synthesis.

    Notes
    -----
    Generated from ChatGPT.

    Parameters
    ----------
    n : int
        Number of samples in the time series.
    dt : float, optional
        Time sampling interval (default: 1.0).
    std : float, optional
        Desired standard deviation of the output time series (default: 1.0).
    seed : int, optional
        Random seed for reproducibility.
    return_time : bool, optional
        If True, also return the time sampling vector (default: False).

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
    n_reads: int,
    readtime: float,
    t0: float = 0.0,
):
    """
    Build a (n_reads-1, ny, nx) cube of per-read *average* electron rates
    for a time-variable UTR ramp, conserving total flux.

    Parameters
    ----------
    source : np.ndarray
        2D array of shape (ny, nx) of average electron rate (e-/s), averaged over the full exposure.
    flux_scaling : np.ndarray
        1D array of shape (nt,) of dimensionless flux scaling vs time (mean ~ 1).
    flux_scaling_times : np.ndarray
        1D array of shape (nt,) of time samples in seconds (assumed monotonic).
    n_reads : int
        Total number of reads (including the first zero-count read).
    readtime : float
        Time between reads in seconds.
    t0 : float, optional
        Start time of the ramp. Default is 0.0.

    Returns
    -------
    rate_cube : np.ndarray
        3D array of shape (n_reads-1, ny, nx); average e-/s rate map for each read interval.
    scaling_mean_per_read : np.ndarray
        1D array of shape (n_reads-1,); mean flux scaling applied during each interval.
    """

    if source.ndim != 2:
        raise ValueError("source must be 2D (ny, nx).")
    if flux_scaling.ndim != 1:
        raise ValueError("flux_scaling must be 1D.")
    if flux_scaling_times.ndim != 1:
        raise ValueError("flux_scaling_times must be 1D.")
    if flux_scaling.size != flux_scaling_times.size:
        raise ValueError("flux_scaling and flux_scaling_times must match in size.")
    if n_reads < 2:
        raise ValueError("n_reads must be >= 2.")

    ny, nx = source.shape
    T = (n_reads - 1) * readtime

    # Read boundaries
    read_edges = t0 + np.arange(n_reads) * readtime

    # Integration grid: union of read edges and provided samples
    t_grid = np.unique(
        np.concatenate([read_edges, flux_scaling_times])
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
    interval_integrals = np.empty(n_reads - 1, dtype=float)

    for i in range(n_reads - 1):
        t_lo = read_edges[i]
        t_hi = read_edges[i + 1]

        sel = (t_grid >= t_lo) & (t_grid <= t_hi)
        interval_integrals[i] = np.trapezoid(
            scaling_grid[sel],
            t_grid[sel]
        )

    # Enforce flux conservation over the whole exposure
    total_integral = np.sum(interval_integrals)
    norm = T / total_integral

    # Mean scaling per read
    scaling_mean_per_read = norm * interval_integrals / readtime

    # Build rate cube
    rate_cube = source[None, :, :] * scaling_mean_per_read[:, None, None]

    return rate_cube, scaling_mean_per_read