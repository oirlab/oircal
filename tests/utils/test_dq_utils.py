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

    # Same field names give the same class
    assert type(dq_flags_to_namedtuple(flags)) is type(dq_flags)

    # Picklable, and the class can be looked up by name in its module
    assert pickle.loads(pickle.dumps(dq_flags)) == dq_flags
    assert getattr(dq_utils, type(dq_flags).__name__) is type(dq_flags)


def test_dq_flags_to_namedtuple_class_recreated_from_name():
    # A class name not created yet in this process resolves on attribute access
    cls = getattr(dq_utils, "DQFlags__ALPHA__BETA")
    assert cls._fields == ("ALPHA", "BETA")


@pytest.mark.parametrize("name", ["_PRIVATE", "DOUBLE__UNDERSCORE", "1BAD", "has space"])
def test_dq_flags_to_namedtuple_invalid_names(name):
    with pytest.raises(ValueError):
        dq_flags_to_namedtuple({name: 1})
