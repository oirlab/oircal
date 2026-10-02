"""Tests for ramp mid-times and ramp chunks.

Ported from the ``get_mid_delta_times_ramp`` and ``get_ramp_chunks`` sections of
hispecdrp's ``test_stage0_units``.
"""
from oircal.ramp_fitting.midpoint import calculate_midpoint_time_delta, get_ramp_chunks
import numpy as np


def test_calculate_midpoint_time_delta(dq_flags):
    n_reads = 4
    times = np.arange(n_reads)
    dq_raw = np.zeros((n_reads), dtype=np.uint32)
    mid_time = calculate_midpoint_time_delta(times, dq_raw, dq_flags)
    np.testing.assert_array_equal(mid_time, 1.5, err_msg='Error in mid_time calculation with no DQ flags')

    n_reads = 6
    times = np.arange(n_reads)
    dq_raw = np.zeros((n_reads), dtype=np.uint32)
    dq_raw[[0, 3]] |= dq_flags.DO_NOT_USE
    dq_raw[[5]] |= dq_flags.SATURATED
    mid_time = calculate_midpoint_time_delta(times, dq_raw, dq_flags)
    np.testing.assert_array_equal(mid_time, 1.5, err_msg='Error in mid_time calculation with DO_NOT_USE flags')

    n_reads = 5
    times = np.arange(n_reads)
    dq_raw = np.zeros((n_reads), dtype=np.uint32)
    dq_raw[[2]] |= dq_flags.JUMP_DET
    mid_time = calculate_midpoint_time_delta(times, dq_raw, dq_flags)
    np.testing.assert_array_equal(mid_time, (0.5 + 2.5 + 3.5) / 3.0,
                                  err_msg='Error in mid_time calculation with JUMP_DET flag')


def test_get_ramp_chunks(dq_flags):
    # test all bad return empty
    n_reads = 4
    dq_raw = np.ones((n_reads), dtype=np.uint8)
    chunks = get_ramp_chunks(dq_raw, dq_flags)
    np.testing.assert_array_equal(len(chunks[0]), 0, err_msg='Error: chunks should be all empty when all bad')
    np.testing.assert_array_equal(len(chunks[1]), 0, err_msg='Error: chunks should be all empty when all bad')

    # test all good
    n_reads = 4
    dq_raw = np.zeros((n_reads), dtype=np.uint8)
    chunks = get_ramp_chunks(dq_raw, dq_flags)
    np.testing.assert_array_equal(chunks[0], [0], err_msg='Error when all reads are good')
    np.testing.assert_array_equal(chunks[1], [4], err_msg='Error when all reads are good')

    # test saturation at the end
    n_reads = 4
    dq_raw = np.zeros((n_reads), dtype=np.uint8)
    dq_raw[[3]] |= dq_flags.SATURATED
    chunks = get_ramp_chunks(dq_raw, dq_flags)
    np.testing.assert_array_equal(chunks[0], [0], err_msg='Error when saturation')
    np.testing.assert_array_equal(chunks[1], [3], err_msg='Error when saturation')

    # test maximum number of jumps
    n_reads = 6
    dq_raw = np.zeros((n_reads), dtype=np.uint8)
    dq_raw[[2, 4]] |= dq_flags.JUMP_DET
    chunks = get_ramp_chunks(dq_raw, dq_flags)
    np.testing.assert_array_equal(chunks[0], [0, 2, 4], err_msg='Error when maximum number of jumps')
    np.testing.assert_array_equal(chunks[1], [2, 4, 6], err_msg='Error when maximum number of jumps')
    dq_raw[[3]] |= dq_flags.DO_NOT_USE
    chunks = get_ramp_chunks(dq_raw, dq_flags)
    np.testing.assert_array_equal(chunks[0], [0, 4], err_msg='Error when maximum number of jumps')
    np.testing.assert_array_equal(chunks[1], [2, 6], err_msg='Error when maximum number of jumps')
