from numba import njit, prange
import numpy as np

from ..utils.dq_utils import dq_flags_to_namedtuple
from ..utils.parallelization_utils import numba_thread_scope


__all__ = ["jump_detection_difference"]


@njit(nogil=True)
def get_diffs(ramp_row: np.ndarray):
    """
    Successive-read differences as float64 (avoid uint16 wraparound).

    Parameters
    ----------
    ramp_row
        2D array of shape (Nreads, Ncols) containing the ramps.
    """
    Nreads, Ncols = ramp_row.shape
    diffs = np.empty((Nreads - 1, Ncols), dtype=np.float64)
    for i in range(Nreads - 1):
        for j in range(Ncols):
            diffs[i, j] = np.float64(ramp_row[i + 1, j]) - np.float64(ramp_row[i, j])
    return diffs


@njit(nogil=True)
def median_axis0_3d(ramp: np.ndarray):
    """
    Median along axis=0 for a 3D array shaped (Nreads, Nrows, Ncols).
    Returns NumPy's median on integer inputs.

    Parameters
    ----------
    ramp
        3D array of shape (Nreads, Nrows, Ncols).
    """
    n, Nrows, Ncols = ramp.shape
    out = np.empty((Nrows, Ncols), dtype=np.float64)
    for y in range(Nrows):
        for x in range(Ncols):
            vec = np.empty(n, dtype=np.float64)
            for i in range(n):
                vec[i] = ramp[i, y, x]
            vec.sort()
            mid = n // 2
            if n % 2 == 0:
                out[y, x] = 0.5 * (vec[mid - 1] + vec[mid])
            else:
                out[y, x] = vec[mid]
    return out


@njit(nogil=True, parallel=True)
def _jump_detection_difference_numba(
    ramp_data: np.ndarray,
    dq_raw: np.ndarray,
    dq: np.ndarray,
    dq_flags: tuple,
    rdnoise: np.ndarray,
    gain: np.ndarray,
    rejection_threshold: float,
):
    Nreads, Nrows, Ncols = ramp_data.shape

    for i in prange(Nrows):
        for j in range(Ncols):
            # Find first saturated index; exclude it and everything after.
            n_use = 1
            for k in range(Nreads):
                # checking backward for saturation to speed up processing since saturation will be at the end of the ramp
                if (dq_raw[Nreads - k - 1, i, j] & (dq_flags.SATURATED | dq_flags.DO_NOT_USE)) == 0:
                    n_use = Nreads - k
                    break

            # Calculate differences between successive reads (float64 DN)
            diffs = get_diffs(ramp_data[0:n_use, i, j : j + 1])

            # Typical difference and noise in DN (rdnoise is DN/read)
            g = gain[i, j] if gain[i, j] > 0.0 else 1.0
            median_diff_dn = median_axis0_3d(diffs[:, None, :])[0, 0]
            photon_var_dn = np.abs(median_diff_dn) / g  # Poisson var of Δ in DN
            total_dn_var = (rdnoise[i, j] ** 2) * 2.0 + photon_var_dn
            sigma_diff = np.sqrt(total_dn_var) if total_dn_var > 0.0 else 0.0

            # Identify jumps
            thr = rejection_threshold * sigma_diff
            for t in range(diffs.shape[0]):
                if sigma_diff > 0.0 and np.abs(diffs[t, 0] - median_diff_dn) > thr:
                    dq_raw[t + 1, i, j] |= dq_flags.JUMP_DET

    return


def jump_detection_difference(
    input_data: np.ndarray,
    input_dq_raw: np.ndarray,
    input_dq: np.ndarray,
    dq_flags: tuple,
    rdnoise: np.ndarray | None = None,
    gain: np.ndarray | None = None,
    rejection_threshold: float = 4.0,
    max_cores: int = 1,
) -> np.ndarray:
    """
    Simple jump detection for a 3D array of up the ramp data.
    Modifies the input dq array (in-place) to flag where jumps are detected.

    Parameters
    ----------
    input_data
        3D array of the UTR data with shape (Nreads, Nrows, Ncols).
    input_dq_raw
        Input data quality flags for the UTR data with shape (Nreads, Nrows, Ncols).
    input_dq
        Input data quality flags with shape (Nrows, Ncols).
    dq_flags
        Dict mapping DQ flag names to bit values, with keys DO_NOT_USE, SATURATED and JUMP_DET.
    rdnoise
        Input read noise array with shape (Nrows, Ncols). If None, 0 is used. Default is None.
    gain
        Input gain array with shape (Nrows, Ncols). If None, 1 is used. Default is None.
    rejection_threshold
        Sigma threshold for jump detection. Default is 4.0.
    max_cores
        Maximum number of CPU cores to use for parallel processing. Default is 1.
    """
    dq_flags = dq_flags_to_namedtuple(dq_flags)

    Nreads, Nrows, Ncols = input_data.shape

    # TODO - determine how to use the input DQ flags to ignore bad data

    if rdnoise is None:
        # TODO - optimize read noise when array is not passed
        rdnoise = np.zeros((Nrows, Ncols), dtype=np.float32)
    if gain is None:
        # TODO - optimize gain when array is not passed
        gain = np.ones((Nrows, Ncols), dtype=np.float32)

    assert Nreads >= 2, "Need at least 2 reads to do jump detection."

    with numba_thread_scope(max_cores):
        _jump_detection_difference_numba(
            input_data,
            input_dq_raw,
            input_dq,
            dq_flags,
            rdnoise,
            gain,
            rejection_threshold,
        )
