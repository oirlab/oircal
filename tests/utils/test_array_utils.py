"""Tests for array helpers."""

import numpy as np
import pytest

from oircal.utils.array_utils import convert_endianness, polyval1d


@pytest.mark.parametrize("dtype", ["f4", "f8", "i2", "u4"])
def test_convert_endianness_big_to_little(dtype):
    values = np.arange(12).reshape(3, 4).astype(dtype)
    big = values.astype(np.dtype(dtype).newbyteorder(">"))

    little = convert_endianness(big)
    assert little.dtype.byteorder in ("<", "=")
    np.testing.assert_array_equal(little, values)


def test_convert_endianness_little_to_big():
    values = np.linspace(0, 1, 5, dtype="<f4")
    big = convert_endianness(values, "big")
    assert big.dtype.byteorder == ">"
    np.testing.assert_array_equal(big, values)


def test_convert_endianness_no_op():
    # Already little-endian (explicit or native) and byte-order-free dtypes come back unchanged
    little = np.zeros(4, dtype="<f4")
    assert convert_endianness(little) is little
    native = np.zeros(4, dtype=np.float32)
    if native.dtype.byteorder == "=" and np.little_endian:
        assert convert_endianness(native) is native
    flags = np.zeros(4, dtype=np.uint8)
    assert convert_endianness(flags) is flags
    assert convert_endianness(flags, "big") is flags


def test_converted_array_works_with_numba():
    coeffs = np.array([1.0, 2.0])  # 1 + 2 x
    x = np.arange(4, dtype=">f8")
    np.testing.assert_allclose(polyval1d(coeffs, convert_endianness(x)), 1 + 2 * np.arange(4))
