# -*- coding: utf-8 -*-


import logging

import numba
import numpy as np

try:
    import scipy.signal
    from scipy import optimize as opt
except ImportError as e:
    opt_import_err = e
else:
    opt_import_err = None

from africanus.util.requirements import requires_optional


@numba.jit(nopython=True, nogil=True, cache=True)
def twod_gaussian(coords, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x = coords[0]
    y = coords[1]
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = (offset + amplitude *
         np.exp(- (a*((x-xo)**2) +
                   2*b*(x-xo)*(y-yo) +
                   c*((y-yo)**2))))
    return g.flatten()


@requires_optional('scipy', opt_import_err)
def fit_2d_gaussian(psf):
    """
    Fit an elliptical Gaussian to the primary lobe of the psf
    """

    # Get the full width at half maximum height of the psf

    # numba doesn't have argwhere, but it can jit argwhere's
    # implementation
    # I = np.stack((psf>=0.5*psf.max()).nonzero()).transpose()

    loc = np.argwhere(psf >= 0.5*psf.max())
    # Create an array with these values at the same indices and zeros otherwise
    lk, mk = psf.shape
    psf_fit = np.zeros_like(psf)
    psf_fit[loc[:, 0], loc[:, 1]] = psf[loc[:, 0], loc[:, 1]]
    # Create x and y indices
    x = np.linspace(0, psf.shape[0]-1, psf.shape[0])
    y = np.linspace(0, psf.shape[1]-1, psf.shape[1])
    x, y = np.meshgrid(x, y)
    # Set starting point of optimiser
    initial_guess = (0.5, lk/2, mk/2, 1.75, 1.4, -4.0, 0)
    # Flatten the data
    data = psf_fit.ravel()
    # Fit the function (Gaussian for now)
    popt, pcov = opt.curve_fit(twod_gaussian, (x, y), data, p0=initial_guess)
    # Get function with fitted params
    data_fitted = twod_gaussian((x, y), *popt)
    # Normalise the psf to have a max value of one
    data_fitted = data_fitted/data_fitted.max()
    return data_fitted.reshape(lk, mk)


@numba.jit(nopython=True, nogil=True, cache=True)
def find_peak(residuals):
    abs_residuals = residuals
    min_peak = abs_residuals.min()
    max_peak = abs_residuals.max()

    nx, ny = abs_residuals.shape

    minx, miny = -1, -1
    maxx, maxy = -1, -1
    peak_intensity = -1

    for x in range(nx):
        for y in range(ny):
            intensity = abs_residuals[x, y]

            if intensity == min_peak:
                minx = x
                miny = y

            if intensity == max_peak:
                maxx = x
                maxy = y
                peak_intensity = intensity

    if minx == -1 or miny == -1:
        raise ValueError("Minimum peak not found")

    if maxx == -1 or maxy == -1:
        raise ValueError("Maximum peak not found")

    return maxx, maxy, minx, miny, peak_intensity


@numba.jit(nopython=True, nogil=True, cache=True)
def build_cleanmap(clean, intensity, gamma, p, q):
    clean[p, q] += intensity*gamma


@numba.jit(nopython=True, nogil=True, cache=True)
def update_residual(residual, intensity, gamma, p, q, npix, psf):
    npix = residual.shape[0]  # Assuming square image
    residual -= gamma*intensity*psf[npix - 1 - p:2*npix - 1 - p,
                                    npix - 1 - q:2*npix - 1 - q]


def hogbom_clean(dirty, psf,
                 gamma=0.1,
                 threshold="default",
                 niter="default"):
    """
    Performs Hogbom Clean on the  ``dirty`` image given the ``psf``.

    Parameters
    ----------
    dirty : np.ndarray
        float64 dirty image of shape (ny, nx)
    psf : np.ndarray
        float64 Point Spread Function of shape (2*ny, 2*nx)
    gamma (optional) float
        the gain factor (must be less than one)
    threshold (optional) : float or str
        the threshold to clean to
    niter (optional : integer
        the maximum number of iterations allowed

    Returns
    -------
    np.ndarray
        float64 clean image of shape (ny, nx)
    np.ndarray
        float64 residual image of shape (ny, nx)
    """
    # deep copy dirties to first residuals,
    # want to keep the original dirty maps
    residuals = dirty.copy()

    # Check that psf is twice the size of residuals
    if (psf.shape[0] != 2*residuals.shape[0] or
            psf.shape[1] != 2*residuals.shape[1]):
        raise ValueError("Warning psf not right size")

    # Initialise array to store cleaned image
    clean = np.zeros_like(residuals)

    assert clean.shape[0] == clean.shape[1]
    npix = clean.shape[0]

    if niter == "default":
        niter = 3*npix

    p, q, pmin, qmin, intensity = find_peak(residuals)

    if threshold == "default":
        # Imin + 0.001*(intensity - Imin)
        threshold = 0.2*np.abs(intensity)
        logging.info("Threshold set at %s", threshold)
    else:
        # Imin + 0.001*(intensity - Imin)
        threshold = threshold*np.abs(intensity)
        logging.info("Assuming user set threshold at %s", threshold)

    # CLEAN the image
    i = 0

    while np.abs(intensity) > threshold and i <= niter:
        logging.info("min %f max %f peak %f threshold %f" %
                     (residuals.min(), residuals.max(), intensity, threshold))

        # First we set the
        build_cleanmap(clean, intensity, gamma, p, q)
        # Subtract out pixel
        update_residual(residuals, intensity, gamma, p, q, npix, psf)
        # Get new indices where residuals is max
        p, q, _, _, intensity = find_peak(residuals)
        # Increment counter
        i += 1
        # Warn if niter exceeded
        if i > niter:
            logging.warn("Number of iterations exceeded")
            logging.warn("Minimum residuals = %s", residuals.max())

    logging.info("Done cleaning after %d iterations.", i)

    return clean, residuals


@requires_optional("scipy", opt_import_err)
def restore(clean, psf, residuals):
    """
    Parameters
    ----------
    clean : np.ndarray
        float64 clean image of shape (ny, nx)
    psf : np.ndarray
        float64 Point Spread Function of shape (2*ny, 2*nx)
    residuals : np.ndarray
        float64 residual image of shape (ny, nx)

    Returns
    -------
    np.ndarray
        float64 Restored image of shape (ny, nx)
    np.ndarray
        float64 Convolved model of shape (ny, nx)
    """

    logging.info("Fitting 2D Gaussian")

    # get the ideal beam (fit 2D Gaussian to HWFH of psf)
    clean_beam = fit_2d_gaussian(psf)

    logging.info("Convolving")

    # cval=0.0) #Fast using fft
    iconv_model = scipy.signal.fftconvolve(clean, clean_beam, mode='same')

    logging.info("Convolving done")

    # Finally we add the residuals back to the image
    restored = iconv_model + residuals

    return (restored, iconv_model)


if __name__ == "__main__":
    pass
