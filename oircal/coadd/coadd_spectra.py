import numpy as np

from .coadd import coadd


__all__ = ["coadd_spectra"]


def coadd_spectra(
    input_data: np.ndarray,
    input_err: np.ndarray | None = None,
    input_dq: np.ndarray | None = None,
    input_var_poisson: np.ndarray | None = None,
    input_var_rnoise: np.ndarray | None = None,
    input_mid_time: np.ndarray | None = None,
    dq_flags: dict | None = None,
    method: str = "wmean",
    err_method: str = "se",
    do_sigma_clip: bool = False,
    sigma_thresh_low: float | None = 5.0,
    sigma_thresh_high: float | None = 5.0,
    min_keep: int | float = 3,
    max_cores: int = 1,
) -> dict[str, np.ndarray | None]:
    """
    Coadd spectra sampled on a common wavelength grid.

    Each pixel (of each order) is coadded independently over the spectra with :func:`.coadd`,
    so the spectra must already be on the same wavelength grid; no resampling is done. See
    :func:`.coadd` for how values are selected and combined.

    Parameters
    ----------
    input_data
        Array of shape (Nspectra, Npix) or (Nspectra, Norders, Npix) of spectra.
    input_err
        Array of the same shape as `input_data` of uncertainties. Required for `method`
        'wmean' and for `err_method` 'se'. Default is None.
    input_dq
        Array of the same shape as `input_data` of DQ flags. Nonzero values exclude the value.
        If None, all values are good. Default is None.
    input_var_poisson
        Array of the same shape as `input_data` of Poisson variances. Default is None.
    input_var_rnoise
        Array of the same shape as `input_data` of read noise variances. Default is None.
    input_mid_time
        Array of the same shape as `input_data` of mid times. Default is None.
    dq_flags
        Dict mapping DQ flag names to bit values, with key DO_NOT_USE. If None, pixels with no
        good values are not flagged. Default is None.
    method
        How to combine the good values: 'wmean' (inverse-variance weighted mean), 'mean' or
        'median'. Default is 'wmean'.
    err_method
        How to compute the uncertainty: 'se' (standard error propagated from `input_err`) or
        'scatter' (measured from the spread of the good values). Default is 'se'.
    do_sigma_clip
        Sigma clip each pixel's good values before coadding, using the biweight location and
        midvariance (see :func:`.sigma_clip_stack`). Only done with more than 3 spectra. The
        spectra should be at comparable flux levels. Default is False.
    sigma_thresh_low
        Low-side sigma clipping threshold in standard deviations. If None, no low-side
        clipping. Default is 5.0.
    sigma_thresh_high
        High-side sigma clipping threshold in standard deviations. If None, no high-side
        clipping. Default is 5.0.
    min_keep
        Minimum number of good values to keep per pixel when sigma clipping. A value below 1 is
        a fraction of Nspectra. Default is 3.
    max_cores
        Maximum number of CPU cores to use. Default is 1.

    Returns
    -------
    output : dict[str, np.ndarray or None]
        Arrays of shape (Npix,) or (Norders, Npix) with keys data, err, var_poisson,
        var_rnoise, dq and mid_time. A key is None if it cannot be computed from the inputs
        given (dq is always returned).
    """
    if input_data.ndim not in (2, 3):
        raise ValueError(
            "input_data must have shape (Nspectra, Npix) or (Nspectra, Norders, Npix), "
            f"got {input_data.shape}",
        )
    return coadd(
        input_data,
        input_err=input_err,
        input_dq=input_dq,
        input_var_poisson=input_var_poisson,
        input_var_rnoise=input_var_rnoise,
        input_mid_time=input_mid_time,
        dq_flags=dq_flags,
        method=method,
        err_method=err_method,
        do_sigma_clip=do_sigma_clip,
        sigma_thresh_low=sigma_thresh_low,
        sigma_thresh_high=sigma_thresh_high,
        min_keep=min_keep,
        max_cores=max_cores,
    )
