from numba import njit, prange
import numpy as np

from .utils import mean_numba_if_not_zero


@njit(nogil=True)
def get_ramp_chunks(dq_raw: np.ndarray, dq_flags: tuple) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (starts, ends) arrays of indices of dtype int64.
    Drops everything at and after the first SATURATED index.
    Keeps the JUMP_DET read but splits before it (i.e., previous chunk ends at i-1,
    next chunk starts at i).
    Excludes DO_NOT_USE reads entirely (may split).

    Notes
    -----
    Does NOT filter out single-read chunks; wrapper does that.

    Parameters
    ----------
    dq_raw
        1D uint8 array of shape (Nreads,) containing the data quality flags.
    dq_flags
        Named tuple of DQ bit values with integer fields DO_NOT_USE, SATURATED and JUMP_DET.

    Returns
    -------
    starts : np.ndarray
        1D int32 array of shape (n_chunks,) of starting indices of valid ramp chunks.
    ends : np.ndarray
        1D int32 array of shape (n_chunks,) of ending indices (exclusive) of valid ramp chunks.
    """
    n = dq_raw.size

    # Find first saturated index; exclude it and everything after.
    n_use = 1
    for i in range(n):
        # checking backward for saturation to speed up processing since saturation will be at the end of the ramp
        if (dq_raw[n - i - 1] & (dq_flags.SATURATED | dq_flags.DO_NOT_USE)) == 0:
            n_use = n - i
            break
    # If all bad, n_use will be 1, and we return empty arrays below
    # If no trailing bad reads, then n_use == n
    if n_use == 1:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    # Preallocate worst-case space (each valid run -> separate chunk)
    starts = np.empty(n_use // 2 + 1, dtype=np.int32)
    ends = np.empty(n_use // 2 + 1, dtype=np.int32)
    count = 0

    in_chunk = False
    start = 0

    for i in range(n_use):
        dqv = dq_raw[i]

        # If bad, close chunk and move to next read
        if (dqv & dq_flags.DO_NOT_USE) != 0:
            if in_chunk:  # then close chunk
                end = i
                if end - start >= 2:
                    starts[count] = start
                    ends[count] = end
                    count += 1
                in_chunk = False
            continue

        # If jump, close chunk and start new one
        if (dqv & dq_flags.JUMP_DET) != 0 and in_chunk:  # then close chunk
            end = i
            if end - start >= 2:
                starts[count] = start
                ends[count] = end
                count += 1
            in_chunk = False

        # Open chunk if needed
        if not in_chunk:
            start = i
            in_chunk = True

    # Flush trailing chunk
    if in_chunk:
        end = n_use
        if end - start >= 2:
            starts[count] = start
            ends[count] = end
            count += 1

    return starts[:count], ends[:count]


@njit(nogil=True, parallel=True)
def calculate_midpoint_time_deltas(
    times: np.ndarray,
    dq_raw: np.ndarray,
    dq_flags: tuple,
) -> np.ndarray:
    """
    Parallelized wrapper for calculate_midpoint_time_delta to apply on a 3D DQ array.
    Loops over the 2nd and 3rd dimensions (y, x) and applies calculate_midpoint_time_delta to each pixel.

    Parameters
    ----------
    times
        1D array of shape (Nreads,) containing the read times.
    dq_raw
        3D uint8 array of shape (Nreads, Nrows, Ncols) containing the DQ flags.
    dq_flags
        Named tuple of DQ bit values with integer fields DO_NOT_USE, SATURATED and JUMP_DET.

    Returns
    -------
    mid_time_map : np.ndarray
        2D array of shape (Nrows, Ncols) of mid times for each pixel.
    """
    _, Nrows, Ncols = dq_raw.shape
    mid_time_map = np.empty((Nrows, Ncols), dtype=np.float32)
    for j in prange(Nrows):
        for k in range(Ncols):
            mid_time_map[j, k] = calculate_midpoint_time_delta(times, dq_raw[:, j, k], dq_flags)
    return mid_time_map


@njit(nogil=True)
def calculate_midpoint_time_delta(
    times: np.ndarray,
    dq_raw: np.ndarray,
    dq_flags: tuple,
) -> float:
    """
    Compute the mid time of the exposure accounting for the DQ flags: DO_NOT_USE, SATURATED, or JUMP_DET.

    Parameters
    ----------
    times
        1D array of shape (Nreads,) containing the read times.
    dq_raw
        1D uint8 array of shape (Nreads,) containing the data quality flags.
    dq_flags
        Named tuple of DQ bit values with integer fields DO_NOT_USE, SATURATED and JUMP_DET.

    Returns
    -------
    mid_delta_time : float
        Mid-exposure offset from the first read, in **seconds** (not MJD days).
    """
    Nreads = len(times)

    # The mid point of each read
    avg_bin_times = (times[0 : Nreads - 1] + times[1::]) / 2

    # Mask out the bad reads (DO_NOT_USE, SATURATED, JUMP_DET) and their neighbors
    mask = ((dq_raw & dq_flags.DO_NOT_USE) | (dq_raw & dq_flags.SATURATED)).astype(np.bool)
    mask2 = (dq_raw & dq_flags.JUMP_DET).astype(np.bool)

    # Set the mid times of the bins that are adjacent to bad reads to zero, so they don't contribute to the mean.
    avg_bin_times = avg_bin_times * ~(mask[1:Nreads] | mask[0 : Nreads - 1] | mask2[1:Nreads])

    # Compute the unweighted mean of the remaining reads
    mid_delta_time = mean_numba_if_not_zero(avg_bin_times)

    return mid_delta_time
