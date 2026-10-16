from numba import njit, prange
import numpy as np

from ..utils.dq_utils import dq_flags_to_namedtuple
from ..utils.parallelization_utils import numba_thread_scope


__all__ = ["make_detector_flat"]


def make_detector_flat(
    input_data: np.ndarray,
    input_err: np.ndarray | None = None,
    input_dq: np.ndarray | None = None,
    dq_flags: tuple | None = None,
    cold_pixel_threshold: float = 0.9,
    max_cores: int = 1,
) -> dict[str, np.ndarray | float | None]:
    """
    Make the detector flat from a combined flat-field rate map.

    The flat is the input rate map, e.g. several flat-field exposures combined with
    :func:`.coadd`, divided by its median. Pixels whose normalized flat is below
    `cold_pixel_threshold` are flagged COLD.

    Parameters
    ----------
    input_data
        2D array of shape (Nrows, Ncols) of the combined flat-field rate (DN/s). Not modified.
    input_err
        2D array of shape (Nrows, Ncols) of the rate uncertainty (DN/s). Not modified. Default
        is None.
    input_dq
        2D array of shape (Nrows, Ncols) of per-pixel DQ flags of the rate. Not modified. If
        None, the flat DQ starts at zero. Default is None.
    dq_flags
        Dict mapping DQ flag names to bit values, with key COLD. If None, pixels are not
        flagged. Default is None.
    cold_pixel_threshold
        Normalized flat value below which a pixel is flagged. Default is 0.9.
    max_cores
        Maximum number of CPU cores to use. Default is 1.

    Returns
    -------
    output : dict
        - ``detflat``: 2D array of shape (Nrows, Ncols), `input_data` divided by its median.
        - ``detflat_err``: `input_err` divided by the same median, or None.
        - ``detflat_dq``: 2D uint32 array of shape (Nrows, Ncols) of the flat DQ flags.
        - ``median_rate``: the median of `input_data` (DN/s), for monitoring the lamp.
    """
    dq_flags = dq_flags_to_namedtuple(dq_flags)
    median_rate = float(np.nanmedian(input_data))
    detflat = input_data / median_rate
    detflat_err = None if input_err is None else input_err / median_rate
    if input_dq is None:
        detflat_dq = np.zeros(input_data.shape, dtype=np.uint32)
    else:
        detflat_dq = input_dq.copy()
    if dq_flags is not None:
        with numba_thread_scope(max_cores):
            _flag_below(detflat, detflat_dq, cold_pixel_threshold, dq_flags.COLD)
    return {
        "detflat": detflat,
        "detflat_err": detflat_err,
        "detflat_dq": detflat_dq,
        "median_rate": median_rate,
    }


@njit(nogil=True, parallel=True)
def _flag_below(image, dq, threshold, flag):
    Nrows, Ncols = image.shape
    for j in prange(Nrows):
        for i in range(Ncols):
            if image[j, i] < threshold:
                dq[j, i] |= flag
