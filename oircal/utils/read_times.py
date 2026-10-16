from typing import Literal

import numpy as np


__all__ = [
    "get_delta_read_times",
    "get_first_read_mjd",
]

# Whether each channel is read in reverse, repeating across channels
_REVERSED_CHANNELS = {
    "forward": (False,),
    "reverse": (True,),
    "forward_reverse": (False, True),
    "reverse_forward": (True, False),
}


def _channel_shape(Nrows: int, Ncols: int, Nchannels: int, vertical: bool) -> tuple[int, int]:
    """
    Return the (rows, cols) shape of one readout channel in readout order.

    Within each channel, pixels are read row by row along the width of the channel. For
    horizontal channels, the detector is treated as transposed, so "rows" are the detector
    columns.
    """
    if vertical:
        Nrows_read, Ncols_read = Nrows, Ncols
    else:
        Nrows_read, Ncols_read = Ncols, Nrows
    if Ncols_read % Nchannels != 0:
        raise ValueError(
            f"Detector size {Ncols_read} and number of channels {Nchannels} are incompatible.",
        )
    return Nrows_read, Ncols_read // Nchannels


def get_delta_read_times(
    Nreads: int,
    Nrows: int,
    Ncols: int,
    clock_rate: float,
    Nchannels: int = 1,
    vertical: bool = True,
) -> np.ndarray:
    """
    Get the time of each read in the ramp in seconds relative to the first read.

    The time of the first read of each pixel is given by `get_first_read_mjd`.

    Notes
    -----
    This implementation does not model any specific instrument and is meant for testing only.

    Parameters
    ----------
    Nreads
        Number of reads in the ramp.
    Nrows
        Number of detector rows.
    Ncols
        Number of detector columns.
    clock_rate
        Pixel clock rate of each readout channel in Hz.
    Nchannels
        Number of channels the detector is read out with.
    vertical
        If True, the channels are vertical (each spans all rows); if False, they are
        horizontal (each spans all columns). Within each channel, pixels are read row by row
        for vertical channels, meaning channels are read in the width direction.

    Returns
    -------
    read_times : np.ndarray
        1D float32 array of shape (Nreads,) of the time of each read in seconds relative to
        the first read.

    Raises
    ------
    ValueError
        If the detector width along the channel direction is not divisible by `Nchannels`.
    """
    Nrows_chan, Ncols_chan = _channel_shape(Nrows, Ncols, Nchannels, vertical)
    time_single_read = (Nrows_chan * Ncols_chan) / clock_rate
    read_times = np.arange(Nreads, dtype=np.float32) * time_single_read
    return read_times


def get_first_read_mjd(
    mjd_start: float,
    Nrows: int,
    Ncols: int,
    clock_rate: float,
    Nchannels: int = 1,
    vertical: bool = True,
    readout_direction: Literal[
        "forward",
        "reverse",
        "forward_reverse",
        "reverse_forward",
    ] = "forward_reverse",
) -> np.ndarray:
    """
    Get the MJD time of the first read of each pixel.

    Pixels in each channel are read one per clock tick starting at `mjd_start`, and all
    channels are read in parallel. The time of each later read relative to the first is given
    by `get_delta_read_times`.

    Notes
    -----
    This implementation does not model any specific instrument and is meant for testing only.

    Parameters
    ----------
    mjd_start
        MJD time the readout of the first read starts.
    Nrows
        Number of detector rows.
    Ncols
        Number of detector columns.
    clock_rate
        Pixel clock rate of each readout channel in Hz.
    Nchannels
        Number of channels the detector is read out with.
    vertical
        If True, the channels are vertical (each spans all rows); if False, they are
        horizontal (each spans all columns). Within each channel, pixels are read row by row
        for vertical channels, meaning channels are read in the width direction.
    readout_direction
        Direction each channel is read along its width. Forward is toward increasing column
        for vertical channels and increasing row for horizontal channels.

        - ``"forward"``: every channel forward.
        - ``"reverse"``: every channel in reverse.
        - ``"forward_reverse"``: the first channel forward, the second in reverse, and so on.
        - ``"reverse_forward"``: the first channel in reverse, the second forward, and so on.

    Returns
    -------
    first_read_mjd : np.ndarray
        2D float64 array of shape (Nrows, Ncols) of the MJD time of the first read of each
        pixel.

    Raises
    ------
    ValueError
        If the detector width along the channel direction is not divisible by `Nchannels`, or
        `readout_direction` is not one of the options above.
    """
    if readout_direction not in _REVERSED_CHANNELS:
        raise ValueError(
            f"readout_direction must be one of {list(_REVERSED_CHANNELS)}, "
            f"got {readout_direction!r}.",
        )
    Nrows_chan, Ncols_chan = _channel_shape(Nrows, Ncols, Nchannels, vertical)

    first_read_mjd_channel = (
        mjd_start
        + np.arange(Nrows_chan * Ncols_chan, dtype=np.float64).reshape(Nrows_chan, Ncols_chan)
        / clock_rate
        / 86400.0
    )

    first_read_mjd = np.tile(first_read_mjd_channel, (1, Nchannels))
    reversed_channels = _REVERSED_CHANNELS[readout_direction]
    flipped = first_read_mjd_channel[:, ::-1]
    for k in range(Nchannels):
        if reversed_channels[k % len(reversed_channels)]:
            first_read_mjd[:, k * Ncols_chan : (k + 1) * Ncols_chan] = flipped

    if vertical:
        return first_read_mjd
    else:
        return first_read_mjd.T
