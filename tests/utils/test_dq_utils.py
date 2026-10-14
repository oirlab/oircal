"""Tests for DQ flag helpers."""

import pickle

import numpy as np
import pytest

from oircal.utils import dq_utils
from oircal.utils.dq_utils import dq_flags_to_namedtuple


def test_dq_flags_to_namedtuple():
    flags = {"DO_NOT_USE": np.uint32(1), "SATURATED": np.uint32(2), "JUMP_DET": np.uint32(4)}
    dq_flags = dq_flags_to_namedtuple(flags)

    assert dq_flags._fields == ("DO_NOT_USE", "SATURATED", "JUMP_DET")
    assert dq_flags._asdict() == flags

    # Same field names give the same class, so Numba compiles once per set of flags
    assert type(dq_flags_to_namedtuple(dict(flags))) is type(dq_flags)

    # Already converted and None pass through unchanged
    assert dq_flags_to_namedtuple(dq_flags) is dq_flags
    assert dq_flags_to_namedtuple(None) is None


def test_dq_flags_to_namedtuple_pickle():
    flags = {"DO_NOT_USE": 1, "SAT_": 2, "JUMP_DET": 4}
    data = pickle.dumps(dq_flags_to_namedtuple(flags))

    # Unpickling rebuilds the named tuple from its fields, even in a process (simulated here
    # by clearing the class cache) that has not created the class yet
    dq_utils._CLASSES.clear()
    restored = pickle.loads(data)
    assert restored._asdict() == flags
    assert type(restored) is type(dq_flags_to_namedtuple(flags))


@pytest.mark.parametrize("name", ["_PRIVATE", "1BAD", "has space", "class"])
def test_dq_flags_to_namedtuple_invalid_names(name):
    with pytest.raises(ValueError):
        dq_flags_to_namedtuple({name: 1})
