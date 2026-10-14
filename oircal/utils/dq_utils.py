"""
Helpers for data quality (DQ) flags.
"""

from collections import namedtuple


__all__ = [
    "dq_flags_to_namedtuple",
]

# One named tuple class per set of field names, so Numba compiles once per set of flags.
_CLASSES: dict[tuple[str, ...], type] = {}


def dq_flags_to_namedtuple(dq_flags: dict | tuple | None) -> tuple | None:
    """
    Convert a dict of DQ flag names and bit values to a named tuple for Numba.

    Functions that take `dq_flags` accept a dict and call this on it, since Numba handles
    named tuples efficiently. The same field names (in the same order) always give the same
    class. A named tuple from this function is returned unchanged, and None is returned as
    None. The named tuple can be pickled (e.g. sent to worker processes); it is rebuilt from
    its fields.

    Parameters
    ----------
    dq_flags
        Mapping of flag name to bit value, e.g. ``{"DO_NOT_USE": 1, "SATURATED": 2}``. Names
        must be valid Python identifiers that do not start with an underscore.

    Returns
    -------
    dq_flags : namedtuple or None
        Named tuple with one field per flag, in the order of `dq_flags`.
    """
    if dq_flags is None:
        return None
    if isinstance(dq_flags, tuple) and type(dq_flags) in _CLASSES.values():
        return dq_flags
    fields = tuple(dq_flags)
    cls = _CLASSES.get(fields)
    if cls is None:
        cls = namedtuple("DQFlags", fields)  # validates the names
        cls.__reduce__ = _reduce_dq_flags
        _CLASSES[fields] = cls
    return cls(**dq_flags)


def _reduce_dq_flags(dq_flags: tuple) -> tuple:
    """Pickle a DQ flags named tuple as a call that rebuilds it from its fields."""
    return dq_flags_to_namedtuple, (dq_flags._asdict(),)
