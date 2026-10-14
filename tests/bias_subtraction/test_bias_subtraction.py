"""Tests for bias subtraction.

Ported from the bias section of hispecdrp's ``test_stage0_units``.
"""

import numpy as np

from oircal.bias_subtraction import subtract_bias


def _make_inputs():
    # Fake ramp data: shape (Nreads=3, Nrows=2, Ncols=2)
    data_raw = np.array(
        [
            [[10, 20], [30, 40]],
            [[15, 25], [35, 45]],
            [[20, 30], [40, 50]],
        ],
        dtype=np.float32,
    )
    dq = np.zeros((2, 2), dtype=np.uint32)
    bias_map = np.array([[1, 2], [3, 4]], dtype=np.float32)
    bias_dq = np.array([[0, 1], [2, 4]], dtype=np.uint32)
    return data_raw, dq, bias_map, bias_dq


def test_subtract_bias():
    data_raw, dq, bias_map, bias_dq = _make_inputs()
    expected = data_raw - bias_map

    data_out, dq_out = subtract_bias(data_raw, bias_map, dq, bias_dq)

    # Bias is subtracted from every read, in place
    np.testing.assert_array_equal(data_out, expected, err_msg="Bias subtraction failed")
    assert data_out is data_raw

    # DQ is OR-ed with the bias DQ, in place
    np.testing.assert_array_equal(dq_out, bias_dq, err_msg="Bias DQ propagation failed")
    assert dq_out is dq


def test_subtract_bias_without_dq():
    data_raw, dq, bias_map, bias_dq = _make_inputs()
    expected = data_raw - bias_map

    # No DQ at all: data still corrected, DQ returned as None
    data_out, dq_out = subtract_bias(data_raw.copy(), bias_map)
    np.testing.assert_array_equal(data_out, expected)
    assert dq_out is None

    # Only one of the two DQ arrays: DQ left unchanged
    _, dq_out = subtract_bias(data_raw.copy(), bias_map, dq, None)
    np.testing.assert_array_equal(dq_out, np.zeros((2, 2), dtype=np.uint32))
    _, dq_out = subtract_bias(data_raw.copy(), bias_map, None, bias_dq)
    assert dq_out is None
