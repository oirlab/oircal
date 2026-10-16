import numpy as np

from ..coadd.combine import MEDIAN_VAR_FACTOR, combine
from ..utils.dq_utils import dq_flags_to_namedtuple


__all__ = ["make_bias"]


def make_bias(
    input_data: np.ndarray,
    input_dq: np.ndarray | None = None,
    dq_flags: tuple | None = None,
    kTC_threshold: float = 100.0,
    bias_threshold: float = 1000.0,
    sigma_thresh: float = 5.0,
    min_keep: int | float = 10,
    max_cores: int = 1,
) -> dict[str, np.ndarray]:
    """
    Make the bias and kTC noise maps from the first reads of a set of ramps.

    Each pixel's first reads are sigma clipped with :func:`.combine`. The bias is the median
    of the kept first reads, and the kTC (reset) noise is their standard deviation. Pixels
    whose bias or kTC noise is above its threshold are flagged UNRELIABLE_BIAS.

    Parameters
    ----------
    input_data
        3D array of shape (Nfiles, Nrows, Ncols) of the first read of each ramp (DN), e.g.
        ``ramps[:, 0]`` for a stack of ramps of shape (Nfiles, Nreads, Nrows, Ncols).
    input_dq
        2D array of shape (Nrows, Ncols) of per-pixel DQ flags to start the bias DQ from. Not
        modified. If None, the bias DQ starts at zero. Default is None.
    dq_flags
        Dict mapping DQ flag names to bit values, with key UNRELIABLE_BIAS. If None, pixels are
        not flagged. Default is None.
    kTC_threshold
        kTC noise (DN) above which a pixel is flagged. Default is 100.
    bias_threshold
        Bias (DN) above which a pixel is flagged. Default is 1000.
    sigma_thresh
        Sigma clipping threshold, in standard deviations, on both sides. Default is 5.
    min_keep
        Minimum number of first reads to keep per pixel when clipping. A value below 1 is a
        fraction of Nfiles. Default is 10.
    max_cores
        Maximum number of CPU cores to use. Default is 1.

    Returns
    -------
    output : dict
        - ``bias_data``: 2D array of shape (Nrows, Ncols) of the bias (DN).
        - ``bias_dq``: 2D uint32 array of shape (Nrows, Ncols) of the bias DQ flags.
        - ``kTC``: 2D array of shape (Nrows, Ncols) of the kTC noise (DN), the sample standard
          deviation (ddof=1) of the kept first reads.
    """
    dq_flags = dq_flags_to_namedtuple(dq_flags)

    bias, median_err, good = combine(
        input_data,
        method="median",
        err_method="scatter",
        do_sigma_clip=True,
        sigma_thresh_low=sigma_thresh,
        sigma_thresh_high=sigma_thresh,
        min_keep=min_keep,
        max_cores=max_cores,
    )
    # The scatter error of the median is the standard deviation times sqrt(pi / 2) over sqrt(N)
    kTC = median_err * np.sqrt(good.sum(axis=0) / MEDIAN_VAR_FACTOR)

    bias_dq = (
        np.zeros(input_data.shape[1:], dtype=np.uint32) if input_dq is None else input_dq.copy()
    )
    if dq_flags is not None:
        bias_dq[(kTC > kTC_threshold) | (bias > bias_threshold)] |= dq_flags.UNRELIABLE_BIAS
    return {"bias_data": bias, "bias_dq": bias_dq, "kTC": kTC}
