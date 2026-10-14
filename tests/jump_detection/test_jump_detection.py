"""Tests for jump detection."""

import numpy as np

from oircal.jump_detection import jump_detection_difference, jump_detection_jwst
from oircal.simulations import create_ramp


def test_jump_detection(dq_flags):
    np.random.seed(12345)

    slope = 10
    Nrows, Ncols = 10, 10
    source = np.full((Nrows, Ncols), slope, dtype=np.float32)

    gain_val = 1.0
    read_noise_val = 10.0

    gain_data = np.ones((Nrows, Ncols), dtype=np.float32) * gain_val
    read_noise_data = np.ones((Nrows, Ncols), dtype=np.float32) * read_noise_val

    ramp_data = create_ramp(
        source,
        readtime=1.0,
        Nreads=16,
        gain=gain_val,
        flat=1,
        dark=0,
        bias=1000,
        kTC_noise=50,
        read_noise=read_noise_val,
        poisson_noise=True,
        convert_to_uint16=True,
        nonlin_coeffs=None,
        clip_ramps=False,
    )
    data = ramp_data["data"]
    dq_raw = ramp_data["dq_raw"]

    # Modify rows to insert a jump of 100 on specified reads in pixel at specified column
    # Also, create a comparison array with JUMP_DET flag set
    jump_val = 100
    reads = [2, 8, 2, 5]
    rows = [5, 5, 2, 4]
    cols = [5, 5, 6, 7]
    dq_comp = np.zeros(dq_raw.shape, dtype=np.uint8)
    for read, row, col in zip(reads, rows, cols):
        data[read:, row, col] += jump_val
        dq_comp[read, row, col] = np.uint8(dq_flags["JUMP_DET"])

    dq_pixel = np.zeros((Nrows, Ncols), dtype=np.uint32)

    # At this threshold, both methods should work
    thresh = 6.0

    jump_detection_jwst(
        data,
        dq_raw,
        dq_pixel,
        dq_flags,
        gain=gain_data,
        rdnoise=read_noise_data,
        rejection_threshold=thresh,
    )

    np.testing.assert_array_equal(dq_comp, dq_raw)

    # Reinitialize the dq_raw array
    dq_raw = np.zeros(data.shape, dtype=np.uint8)

    thresh = 5.0

    jump_detection_difference(
        data,
        dq_raw,
        dq_pixel,
        dq_flags,
        gain=gain_data,
        rdnoise=read_noise_data,
        rejection_threshold=thresh,
    )

    np.testing.assert_array_equal(dq_comp, dq_raw)

    # At this threshold, no jumps should be identified
    thresh = 100.0
    dq_comp = np.zeros(dq_raw.shape, dtype=np.uint8)
    dq_raw = np.zeros(data.shape, dtype=np.uint8)

    jump_detection_jwst(
        data,
        dq_raw,
        dq_pixel,
        dq_flags,
        gain=gain_data,
        rdnoise=read_noise_data,
        rejection_threshold=thresh,
    )

    np.testing.assert_array_equal(dq_comp, dq_raw)

    # Reinitialize the dq_raw array
    dq_raw = np.zeros(data.shape, dtype=np.uint8)

    jump_detection_difference(
        data,
        dq_raw,
        dq_pixel,
        dq_flags,
        gain=gain_data,
        rdnoise=read_noise_data,
        rejection_threshold=thresh,
    )

    np.testing.assert_array_equal(dq_comp, dq_raw)


def test_jump_detection_difference_noiseless(dq_flags):
    """Noise-free ramp: one large step is flagged, a step below the read noise is not.

    Ported from the jump detection section of hispecdrp's ``test_stage0_units``.
    """
    Nreads, Nrows, Ncols = 10, 1, 1
    ramp = np.arange(Nreads, dtype=np.float32).reshape(Nreads, 1, 1)

    # Inject a single positive jump between read 4 and 5
    # i.e., diffs[4] is large -> dq_raw[5] should be flagged
    ramp[5:, 0, 0] += 20.0
    ramp[2:, 0, 0] += 3 * 0.1  # checking that a small jump below readnoise is not detectable

    dq_raw = np.zeros((Nreads, Nrows, Ncols), dtype=np.uint8)
    dq_2d = np.zeros((Nrows, Ncols), dtype=np.uint8)
    rdnoise = np.full((Nrows, Ncols), 0.1, dtype=np.float32)  # small read noise
    gain = np.ones((Nrows, Ncols), dtype=np.float32)  # unit gain

    # Run jump detection (in-place modification of dq_raw is expected)
    jump_detection_difference(
        input_data=ramp,
        input_dq_raw=dq_raw,
        input_dq=dq_2d,
        dq_flags=dq_flags,
        rdnoise=rdnoise,
        gain=gain,
        rejection_threshold=5.0,
    )

    # Only the read AFTER the large diff (index 5) should be flagged
    flagged = (dq_raw[:, 0, 0] & dq_flags["JUMP_DET"]) != 0
    assert flagged.sum() == 1, f"Expected exactly one jump flag, got {flagged}"
    assert flagged[5]
