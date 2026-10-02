import numpy as np

from stcal.jump.jump import detect_jumps_data
from stcal.jump.jump_class import JumpData

import multiprocessing as mp

__all__ = ['jump_detection_jwst']

def _jump_detection_jwst_row(indata : tuple) -> tuple[int, np.ndarray]:
    """
    Finds jumps for row of a ramp using stcal.jump.jump.detect_jumps_data

    Parameters
    ----------
    indata : tuple
        Tuple containing the following elements for a single row:
            ramp row index (int)
            rejection threshold (float)
            ramp data (np.ndarray of shape (1, n_reads, 1, nx))
            readnoise (np.ndarray of shape (1, nx))
            gain (np.ndarray of shape (1, nx))
            raw data quality (np.ndarray of shape (1, n_reads, 1, nx))
            data quality (np.ndarray of shape (1, nx))
            DQ flags (namedtuple)

    Returns
    -------
    i : int
        index of input row from the ramps, same as input.
    dq_out : np.ndarray
        output data quality image with flags from algorithm (1, nx).
    """

    i, rejection_threshold, ramp_data, rdnoise, gain, dq_raw, dq, dq_flags = indata

    # We are implicitly detecting by row, so columns is only other axis and nrows=1
    nints, nframes, ny, nx = ramp_data.shape

    # Initialize ramp data model
    jump_data = JumpData()

    # Set data arrays and parameters
    jump_data.nframes = 1 # This is not the same as number of reads but number of frames per read
    jump_data.data = ramp_data
    jump_data.gain_2d = gain
    jump_data.rnoise_2d = rdnoise

    # Spoof unused data structure for the groupdq - can be refactored if want to use data quality flags at the group/frame level
    # TODO - update these from input dq
    jump_data.gdq = dq_raw
    jump_data.pdq = dq

    # Tunable parameters - update to be able to tweak these:
    # https://github.com/spacetelescope/stcal/blob/main/src/stcal/jump/jump_class.py
    jump_data.rejection_thresh = rejection_threshold
    jump_data.three_grp_thresh = rejection_threshold
    jump_data.four_grp_thresh = rejection_threshold

    # If set to True (default is True), it will cause the four perpendicular
    # neighbors of all detected jumps to also be flagged as a jump.
    jump_data.flag_4_neighbors = False

    # Snowball information for near-IR
    #  Turns on Snowball detector for NIR detectors
    jump_data.expand_large_events = False

    # The minimum number of groups required to use sigma clipping to find outliers.
    jump_data.minimum_sigclip_groups = np.inf

    # The minimum area of saturated pixels at the center of a snowball. Only
    # contours with area above the minimum will create snowballs.
    jump_data.min_sat_area = 0

    # The factor that is used to increase the size of the enclosing
    # circle/ellipse jump flagged pixels.
    jump_data.expand_factor = 0.0

    # If true there must be a saturation circle within the radius of the jump
    # circle to trigger the creation of a snowball. All true snowballs appear
    # to have at least one saturated pixel.
    jump_data.sat_required_snowball = False

    # The minimum radius of the saturated core of a snowball for the core to be extended
    jump_data.min_sat_radius_extend = 0.0

    # The number of pixels to expand the saturated core of detected snowballs
    jump_data.sat_expand = 0

    # The distance from the edge of the detector where saturated cores are not
    # required for snowball detection
    jump_data.edge_size = 0

    # Turns on the flagging of the faint extended emission of MIRI showers
    jump_data.find_showers = False

    # The minimum number of groups to switch to flagging all outliers in a single pass.
    jump_data.min_diffs_single_pass = 0

    # The maximum width for any extension of saturation or jump
    jump_data.max_extended_width = np.inf

    # Sigma clipping
    # The minimum number of groups for jump detection
    jump_data.minimum_groups = 3

    # In sigma clipping, if True only differences between integrations are compared.
    # If False, then all differences are processed at once.
    jump_data.only_use_ints = False

    # Internal state
    # Number of groups after detected extended emission to flag as a jump for MIRI showers
    jump_data.grps_masked_after_shower = 0

    # The flag to turn on the extension of the flagging of the saturated cores of snowballs.
    jump_data.mask_persist_grps_next_int = False

    # How many groups to be flagged when the saturated cores are extended into
    # subsequent integrations.
    jump_data.persist_grps_flagged = 0

    # We are parallelizing at row level, so set max_cores=1
    jump_data.max_cores = "1"

    # Explicitly set flags required by JWST pipeline
    jump_data.fl_good = dq_flags.GOOD
    jump_data.fl_sat = dq_flags.SATURATED
    jump_data.fl_jump = dq_flags.JUMP_DET
    jump_data.fl_ngv = dq_flags.UNRELIABLE_FLAT # Check if this is best/optimal flag
    jump_data.fl_dnu = dq_flags.DO_NOT_USE
    jump_data.fl_ref = dq_flags.REFERENCE_PIXEL

    gdq, pdq, total_primary_crs, number_extended_events = detect_jumps_data(jump_data)

    # Ignore first axis since we have nints=1
    gdq = gdq[0, :].astype(np.uint32)

    return (i, gdq)

def jump_detection_jwst(
    ramp_data: np.ndarray,
    dq_raw: np.ndarray,
    dq: np.ndarray,
    dq_flags: tuple,
    gain : np.ndarray | None = None,
    rdnoise : np.ndarray | None = None,
    max_cores: int = 1,
    rejection_threshold: float = 4.0
):
    """
    Top-level function for running jump detection with stcal.jump.jump.detect_jumps_data

    Parameters
    ----------
    ramp_data : np.ndarray
        3D array of the UTR data with shape (n_reads, ny, nx).
    dq_raw : np.ndarray
        Input data quality flags for the UTR data with shape (n_reads, ny, nx).
    dq : np.ndarray
        Input data quality flags with shape (ny, nx).
    dq_flags : namedtuple
        Named tuple of DQ bit values with integer fields GOOD, DO_NOT_USE, SATURATED,
        JUMP_DET, UNRELIABLE_FLAT and REFERENCE_PIXEL. Must be defined at module level
        so it can be pickled to worker processes.
    gain : np.ndarray, optional
        Input gain array with shape (ny, nx). Defaults to 1.
    rdnoise : np.ndarray, optional
        Input read noise array with shape (ny, nx). Defaults to 0.
    max_cores : int, optional
        Number of worker processes. Default is 1.
    rejection_threshold : float, optional
        Sigma threshold for jump detection. Default is 4.0.
    """

    n_reads, ny, nx = ramp_data.shape

    if rdnoise is None:
        # TODO - optimize read noise when array is not passed
        rdnoise = np.zeros((ny, nx), dtype=np.float32)
    if gain is None:
        # TODO - optimize gain when array is not passed
        gain = np.ones((ny, nx), dtype=np.float32)

    assert n_reads >= 2, "Need at least 2 reads to do jump detection."

    with mp.Pool(max_cores) as pool:
        # TODO - parallelize with memory sharing instead of this way
        data = []
        for row in range(ny):
            ramps_3d = ramp_data[:, [row], :]
            dq_raw_3d = dq_raw[:, [row], :]
            # Reshape to add the nints axis that JWST pipeline requires
            ramps_3d = ramps_3d[np.newaxis, :]
            dq_raw_3d = dq_raw_3d[np.newaxis, :]

            data.append((
                row,
                rejection_threshold,
                ramps_3d,
                # Make sure to add a new axis to each array to maintain number of dimensions
                rdnoise[[row], :],
                gain[[row], :],
                dq_raw_3d,
                dq[[row], :],
                dq_flags
            ))

        results = pool.map(_jump_detection_jwst_row, data)

    for row_element in results:
        _row, _dq = row_element
        dq_raw[:, [_row], :] = _dq
