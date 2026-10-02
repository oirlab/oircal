from .fit_ramp_numba import fit_ramps_ols, fit_ramps_mcds
from .fit_ramp_jwst import fit_ramps_jwst

__all__ = [
    'fit_ramps_ols',
    'fit_ramps_mcds',
    'fit_ramps_jwst'
]