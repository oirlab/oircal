"""Ramp mid-time must stay in MJD (days), not seconds added to MJD."""

import numpy as np

from oircal.ramp_fitting.utils import seconds_to_mjd_offset


def test_seconds_to_mjd_offset():
    assert seconds_to_mjd_offset(86400.0) == 1.0
    assert seconds_to_mjd_offset(0.0) == 0.0

