"""Tests for saturation flagging.

Ported from the saturation section of hispecdrp's ``test_stage0_units``.
"""

import numpy as np

from oircal.saturation import check_saturation


def test_check_saturation(dq_flags):
    saturation_threshold = np.iinfo(np.uint16).max  # DN
    dq_raw = np.zeros((3, 2, 2), dtype=np.uint8)
    dq = np.zeros((2, 2), dtype=np.uint32)
    data_raw = np.zeros((3, 2, 2), dtype=np.uint16)
    data_raw[2::, :, :] = saturation_threshold
    data_raw[1::, 1, 1] = saturation_threshold
    saturation_map = np.full((2, 2), saturation_threshold, dtype=np.uint16)
    sat_dq = np.zeros((2, 2), dtype=np.uint32)
    check_saturation(
        data_raw,
        dq_raw,
        dq,
        saturation_map,
        sat_dq,
        dq_flags=dq_flags,
    )
    np.testing.assert_array_equal(
        np.sum(dq_raw & dq_flags["SATURATED"]),
        5 * dq_flags["SATURATED"],
        err_msg="Error in saturation check",
    )
