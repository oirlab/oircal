import numpy as np

from ..jump_detection.jump_detection_difference import median_axis0_3d
from ..utils.parallelization_utils import numba_thread_scope


__all__ = ["make_saturation"]


def make_saturation(
    input_data: np.ndarray,
    offset: float = 100.0,
    max_cores: int = 1,
) -> np.ndarray:
    """
    Make the saturation threshold map from the last reads of a set of saturated ramps.

    The threshold is the median of the last reads minus `offset`, which keeps reads that sit
    just below the saturated level (within the read noise) from being missed.

    Parameters
    ----------
    input_data
        3D array of shape (Nfiles, Nrows, Ncols) of the last read of each ramp (DN), e.g.
        ``ramps[:, -1]`` for a stack of ramps of shape (Nfiles, Nreads, Nrows, Ncols). The ramps
        must reach saturation.
    offset
        Amount (DN) subtracted from the median last read. Default is 100.
    max_cores
        Maximum number of CPU cores to use. Default is 1.

    Returns
    -------
    sat_thresh : np.ndarray
        2D float64 array of shape (Nrows, Ncols) of saturation thresholds (DN).
    """
    with numba_thread_scope(max_cores):
        sat_thresh = median_axis0_3d(input_data)
    sat_thresh -= offset
    return sat_thresh
