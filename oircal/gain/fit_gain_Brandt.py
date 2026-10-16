# Author: Tim Brandt
# Shared with HISPEC team in Aug 2025
# Ported into oircal by Bryson Cale October 2026 (gen_likelihood_map and fitquad only)
#
# Reference:
# Brandt, T. D., "Computing the Electronic Gain for Detectors Read Out Up-the-ramp", PASP, 137, 125006 (2025)
# https://ui.adsabs.harvard.edu/abs/2025PASP..137l5006B/abstract

import numpy as np

from ..ramp_fitting import fitramp_cython_Brandt as fitramp_cython


def gen_likelihood_map(
    diffs_lin,
    diffs_quad,
    covar,
    sigs,
    gains,
    alphas,
    countratesidentical=False,
    asig=0,
    cguess=None,
    d2use=None,
    readtime=None,
):
    """
    Generate a 3D array of the log likelihood of the ramp fit
    marginalized over the count rates ramp-by-ramp (or, if the
    countratesidentical variable is set to True, assume that all ramps
    have the same count rate).  Returns a 3D array of the size of
    len(sigs) x len(gains) x len(alphas), where sig is the read noise,
    gain is the gain in electrons/DN, and alpha is the nonlinearity
    coefficient.  Chi squared is equal to this returned array times
    -2.

    Parameters
    ----------
    diffs_lin : 2D array of floats
        Size nreads-1 x nramps. Differences between consecutive reads.
    diffs_quad : 2D array of floats
        Size nreads-1 x nramps. Differences between the squares of the values (from a designated pedestal) in consecutive reads
    covar : fitramp.Covar class
        Should be appropriate for diffs_lin
    sigs : list of floats
        Trial read noise values
    gains : list of floats
        Trial gain values
    alphas : list of floats
        Trial nonlinearity coefficient values
    asig : float, optional
        Standard deviation of the read noise from other constraints. If the best-fit read noise from this other constraint is sigs[1], asig will incorporate this constraint to the likelihood.
    countratesidentical : boolean, optional
        Assume that the count rate is the same for all ramps? Default False.
    cguess : 1D array of floats or None, optional
        Count rates for the purposes of computing a covariance matrix. Default None. If None, compute cguess from the nonlinearity-corrected ramps.
    d2use : 2D array of bools or None, optional
        Boolean mask of differences between reads to use for computing a likelihood. If None, use all read differences.

    Returns
    -------
    logL : 3D array of floats
        log likelihood, shape (len(sigs), len(gains), len(alphas))

    """

    nsig = len(sigs)
    ngain = len(gains)
    nalph = len(alphas)

    logL = np.zeros((nsig, ngain, nalph))

    # Normally, the lines below will not be used; these arrays
    # should be supplied as input to the routine.

    n0, n1 = diffs_lin.shape
    diffs2use = np.ones((n0, n1), dtype=np.uint8) if d2use is None else d2use

    # Estimate the covariance matrix inside this routine?  This will
    # be done using the linearity-corrected ramps.

    dynamic_cguess = cguess is None

    for k in range(nalph):
        diffs_nonlinscaled = diffs_lin + alphas[k] * diffs_quad

        if dynamic_cguess:
            cguess = np.sum(diffs_nonlinscaled * diffs2use, axis=0)
            cguess /= np.sum(diffs2use, axis=0)
            cguess *= cguess > 0

        for j in range(ngain):
            for i in range(nsig):
                r = fitramp_cython.fit_ramps(
                    diffs_nonlinscaled,
                    diffs2use,
                    covar.alpha_phnoise / gains[j],
                    covar.beta_phnoise / gains[j],
                    covar.alpha_readnoise,
                    covar.beta_readnoise,
                    sigs[i],
                    cguess,
                    n1,
                    n0,
                )

                logdetC, A, B, C, countrate, chisq, uncert = r

                # Can only use the lines below if we are *sure* that the count
                # rates are all the same.

                if countratesidentical:
                    c = np.sum(countrate * C) / np.sum(C)
                    chisqval = A - 2 * c * B + c**2 * C
                    logsigval = -0.5 * np.log(np.sum(C))
                else:
                    chisqval = chisq
                    logsigval = fitramp_cython.logsum(uncert, n1)

                logL[i, j, k] = np.sum(-chisqval / 2 - 0.5 * logdetC) + logsigval

                # Incorporate other information about the read noise.
                if asig != 0 and (i == 0 or i == 2):
                    logL[i, j, k] += asig

    return logL


def fitquad(
    diffs_lin,
    diffs_quad,
    covar,
    sigs,
    gains,
    alphas=(0,),
    asig=0,
    countratesidentical=False,
    cguess=None,
    d2use=None,
    log=True,
):
    """
    Fit a quadratic form to the probability distributions from
    marginalizing the ramps over the count rates.  This function
    will return the vertex of the fit, the covariance matrix
    implied by the coefficients, and the coefficients themselves.

    Parameters
    ----------
    diffs_lin : 2D array of float
        Size (nreads-1, nramps). Differences between consecutive reads.
    diffs_quad : 2D array of float
        Size (nreads-1, nramps). Differences between the squares of the values
        (from a designated pedestal) in consecutive reads.
    covar : fitramp.Covar
        Should be appropriate for diffs_lin.
    sigs : list of float
        Trial read noise values.
    gains : list of float
        Trial gain values.
    alphas : list of float
        Trial nonlinearity coefficient values.
    asig : float, optional
        Standard deviation of the read noise from other constraints. If the best-fit read noise
        from this other constraint is sigs[1], asig will incorporate this constraint into the likelihood.
    countratesidentical : bool, optional
        Assume that the count rate is the same for all ramps. Default is False.
    cguess : 1D array of float or None, optional
        Count rates for computing a covariance matrix. Default is None. If None, compute cguess
        from the nonlinearity-corrected ramps.
    d2use : 2D array of bool or None, optional
        Boolean mask of differences between reads to use for computing a likelihood. If None,
        use all read differences.
    log : bool, optional
        Use logarithmic coordinates for the quadratic form. Default is True.

    Returns
    -------
    center : tuple of float
        Center of the quadratic form in gain, read noise, nonlinearity. The coordinates
        are given in log(gain) and log(readnoise) if log is True.
    cov : 2D array of float
        Covariance matrix of the log likelihood as estimated by the quadratic form fit.
    c_2d : list of float
        Coefficients describing the quadratic form.


    """

    logL = gen_likelihood_map(
        diffs_lin,
        diffs_quad,
        covar,
        sigs,
        gains,
        alphas,
        asig=asig,
        countratesidentical=countratesidentical,
        cguess=cguess,
        d2use=d2use,
    )

    arr = np.ones((len(sigs) * len(gains) * len(alphas), 10))
    if log:
        x, y, z = np.meshgrid(np.log(gains), np.log(sigs), alphas)
    else:
        x, y, z = np.meshgrid(gains, sigs, alphas)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    arr[:, 1] = x
    arr[:, 2] = y
    arr[:, 3] = z
    arr[:, 4] = x * y
    arr[:, 5] = x * z
    arr[:, 6] = y * z
    arr[:, 7] = x**2
    arr[:, 8] = y**2
    arr[:, 9] = z**2

    c = np.linalg.lstsq(arr, logL.flatten(), rcond=None)[0]

    # the covariance matrix interpreting the quadratic form as a log likelihood

    Cinv = -np.array([[2 * c[7], c[4], c[5]], [c[4], 2 * c[8], c[6]], [c[5], c[6], 2 * c[9]]])

    # We might be fitting a quadratic form in 3d; we might be fitting one
    # in 2d.

    if len(alphas) > 1:
        cov = np.linalg.inv(Cinv)

        # the vertex of the quadratic form
        xc, yc, zc = np.linalg.solve(Cinv, np.array([c[1], c[2], c[3]]))

        return (xc, yc, zc), cov, c

    else:
        cov = np.linalg.inv(Cinv[:2, :2])

        # the vertex of the quadratic form
        xc, yc = np.linalg.solve(Cinv[:2, :2], np.array([c[1], c[2]]))
        c_2d = np.array([c[0], c[1], c[2], c[4], c[7], c[8]])

        return (xc, yc), cov, c_2d
