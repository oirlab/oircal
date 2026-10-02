import numpy as np
import pytest

from oircal.utils.dq_utils import dq_flags_to_namedtuple

# DQ bit values used by the tests.
_DQ_FLAGS = {
    'GOOD':             np.uint32(0),
    'DO_NOT_USE':       np.uint32(2**0),
    'SATURATED':        np.uint32(2**1),
    'JUMP_DET':         np.uint32(2**2),
    'OUTLIER':          np.uint32(2**4),
    'UNRELIABLE_SLOPE': np.uint32(2**24),
    'UNRELIABLE_FLAT':  np.uint32(2**25),
    'REFERENCE_PIXEL':  np.uint32(2**31),
}


@pytest.fixture
def dq_flags():
    return dq_flags_to_namedtuple(_DQ_FLAGS)
