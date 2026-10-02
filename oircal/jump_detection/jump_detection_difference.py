from ..utils.parallelization_utils import numba_thread_scope

import numpy as np
from numba import njit, prange

__all__ = ['jump_detection_difference']

@njit(nogil=True, cache=True, inline='always')
def get_diffs(ramp_row: np.ndarray):
    """
    Successive-read differences as float64 (avoid uint16 wraparound).

    Parameters
    ----------
    ramp_row : np.ndarray
        2D array of shape (n_reads, nx) containing the ramps.
    """
    n_reads, nx = ramp_row.shape
    diffs = np.empty((n_reads - 1, nx), dtype=np.float64)
    for i in range(n_reads - 1):
        for j in range(nx):
            diffs[i, j] = np.float64(ramp_row[i + 1, j]) - np.float64(ramp_row[i, j])
    return diffs


@njit(nogil=True, cache=True, inline='always')
def median_axis0_3d(ramp: np.ndarray):
    """
    Median along axis=0 for a 3D array shaped (n_reads, ny, nx).
    Returns NumPy's median on integer inputs.

    Parameters
    ----------
    ramp : np.ndarray
        3D array of shape (n_reads, ny, nx).
    """
    n, ny, nx = ramp.shape
    out = np.empty((ny, nx), dtype=np.float64)
    for y in range(ny):
        for x in range(nx):
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

@njit(nogil=True, parallel=True, cache=True)
def _jump_detection_difference_numba(
    ramp_data: np.ndarray,
    dq_raw: np.ndarray,
    dq: np.ndarray,
    dq_flags: tuple,
    rdnoise: np.ndarray,
    gain: np.ndarray,
    rejection_threshold: float,
):
    n_reads, ny, nx = ramp_data.shape

    for i in prange(ny):
        for j in range(nx):
            # Find first saturated index; exclude it and everything after.
            n_use = 1
            for k in range(n_reads):
                # checking backward for saturation to speed up processing since saturation will be at the end of the ramp
                if (dq_raw[n_reads-k-1,i,j] & (dq_flags.SATURATED | dq_flags.DO_NOT_USE)) == 0:
                    n_use = n_reads-k
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
    ramp_data: np.ndarray,
    dq_raw: np.ndarray,
    dq: np.ndarray,
    dq_flags: tuple,
    rdnoise : np.ndarray | None = None,
    gain: np.ndarray | None = None,
    rejection_threshold: float = 4.0,
    max_cores: int = 1,
) -> np.ndarray:
    """
    Simple jump detection for a 3D array of up the ramp data.
    Modifies the input dq array (in-place) to flag where jumps are detected.

    Parameters
    ----------
    ramp_data : np.ndarray
        3D array of the UTR data with shape (Nreads, Ny, Nx).
    dq_raw : np.ndarray
        Input data quality flags for the UTR data with shape (Nreads, Ny, Nx).
    dq : np.ndarray
        Input data quality flags with shape (Ny, Nx).
    dq_flags : namedtuple
        Named tuple of DQ bit values with integer fields DO_NOT_USE, SATURATED and JUMP_DET.
    rdnoise : np.ndarray, optional
        Input read noise array with shape (Ny, Nx). Defaults to 0.
    gain : np.ndarray, optional
        Input gain array with shape (Ny, Nx). Defaults to 1.
    rejection_threshold : float, optional
        Sigma threshold for jump detection. Default is 4.0.
    max_cores : int, optional
        Maximum number of CPU cores to use for parallel processing (default: 1).
    """

    n_reads, ny, nx = ramp_data.shape

    # TODO - determine how to use the input DQ flags to ignore bad data

    if rdnoise is None:
        # TODO - optimize read noise when array is not passed
        rdnoise = np.zeros((ny, nx), dtype=np.float32)
    if gain is None:
        # TODO - optimize gain when array is not passed
        gain = np.ones((ny, nx), dtype=np.float32)

    assert n_reads >= 2, "Need at least 2 reads to do jump detection."

    with numba_thread_scope(max_cores):
        _jump_detection_difference_numba(ramp_data, dq_raw, dq, dq_flags, rdnoise, gain, rejection_threshold)
