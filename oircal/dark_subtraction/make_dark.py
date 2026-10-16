from numba import njit, prange
import numpy as np

from ..utils.dq_utils import dq_flags_to_namedtuple
from ..utils.parallelization_utils import numba_thread_scope


__all__ = ["make_dark"]


def make_dark(
    input_data: np.ndarray,
    input_err: np.ndarray | None = None,
    input_dq: np.ndarray | None = None,
    dq_flags: tuple | None = None,
    hot_pixel_threshold: float = 5.0,
    max_cores: int = 1,
) -> dict[str, np.ndarray | None]:
    """
    Make the dark calibration from a combined dark rate map.

    The dark rate is the input rate map, e.g. several dark exposures combined with
    :func:`.coadd`. Pixels whose dark rate is above `hot_pixel_threshold` are flagged HOT.

    Parameters
    ----------
    input_data
        2D array of shape (Nrows, Ncols) of the combined dark rate (DN/s). Not modified.
    input_err
        2D array of shape (Nrows, Ncols) of the dark rate uncertainty (DN/s). Not modified.
        Default is None.
    input_dq
        2D array of shape (Nrows, Ncols) of per-pixel DQ flags of the dark rate. Not modified.
        If None, the dark DQ starts at zero. Default is None.
    dq_flags
        Dict mapping DQ flag names to bit values, with key HOT. If None, pixels are not
        flagged. Default is None.
    hot_pixel_threshold
        Dark rate above which a pixel is flagged, in the units of `input_data`. Default is 5.
    max_cores
        Maximum number of CPU cores to use. Default is 1.

    Returns
    -------
    output : dict
        - ``dark_rate``: copy of `input_data`.
        - ``dark_err``: copy of `input_err`, or None.
        - ``dark_dq``: 2D uint32 array of shape (Nrows, Ncols) of the dark DQ flags.
    """
    dq_flags = dq_flags_to_namedtuple(dq_flags)
    dark_dq = np.zeros(input_data.shape, dtype=np.uint32) if input_dq is None else input_dq.copy()
    if dq_flags is not None:
        with numba_thread_scope(max_cores):
            _flag_above(input_data, dark_dq, hot_pixel_threshold, dq_flags.HOT)
    return {
        "dark_rate": input_data.copy(),
        "dark_err": None if input_err is None else input_err.copy(),
        "dark_dq": dark_dq,
    }


@njit(nogil=True, parallel=True)
def _flag_above(image, dq, threshold, flag):
    Nrows, Ncols = image.shape
    for j in prange(Nrows):
        for i in range(Ncols):
            if image[j, i] > threshold:
                dq[j, i] |= flag
