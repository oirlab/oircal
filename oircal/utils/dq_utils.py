"""
Helpers for data quality (DQ) flags.
"""
from collections import namedtuple

__all__ = [
    "dq_flags_to_namedtuple",
]

_PREFIX = "DQFlags"
_SEP = "__"


def dq_flags_to_namedtuple(dq_flags : dict) -> tuple:
    """
    Convert a dict of DQ flag names and bit values to the named tuple expected by `dq_flags` arguments.

    The returned named tuple can be pickled (e.g. sent to worker processes) and used with
    Numba's on-disk cache from any process, since its class can be looked up by name in this
    module.

    Parameters
    ----------
    dq_flags : dict
        Mapping of flag name to bit value, e.g. ``{"DO_NOT_USE": 1, "SATURATED": 2}``.
        Names must be valid Python identifiers that do not start with an underscore or
        contain a double underscore.

    Returns
    -------
    dq_flags : namedtuple
        Named tuple with one field per flag, in the order of `dq_flags`.
    """
    for name in dq_flags:
        if not isinstance(name, str) or not name.isidentifier() or name.startswith("_") or _SEP in name:
            raise ValueError(f"Invalid DQ flag name {name!r}")
    cls = _get_dq_flags_class(tuple(dq_flags))
    return cls(**dq_flags)


def _get_dq_flags_class(fields : tuple) -> type:
    """Return the named tuple class for these field names, creating and registering it if needed."""
    name = _PREFIX + _SEP + _SEP.join(fields)
    cls = globals().get(name)
    if cls is None:
        cls = namedtuple(name, fields, module=__name__)
        globals()[name] = cls
    return cls


def __getattr__(name : str):
    # Recreate classes on demand from their names, so pickled instances and Numba cache
    # entries resolve in processes that have not created them yet.
    if name.startswith(_PREFIX + _SEP):
        return _get_dq_flags_class(tuple(name[len(_PREFIX + _SEP):].split(_SEP)))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
