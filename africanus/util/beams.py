# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import string
import re

import numpy as np

from africanus.compatibility import range


class FitsAxes(object):
    """
    FitsAxes object, inspired by Tigger's FITSAxes
    """

    def __init__(self, header=None):
        # Create an zero-dimensional object if no header supplied
        self._ndims = ndims = 0 if header is None else header['NAXIS']

        # Extract header information for each dimension
        axr = list(range(1, ndims+1))
        self._naxis = [header.get('NAXIS%d' % n) for n in axr]
        self._ctype = [header.get('CTYPE%d' % n, n).strip() for n in axr]
        self._crval = [header.get('CRVAL%d' % n, 0) for n in axr]
        # Convert right pixel from FORTRAN to C indexing
        self._crpix = [header['CRPIX%d' % n]-1 for n in axr]
        self._cdelt = [header.get('CDELT%d' % n, 1) for n in axr]
        self._cunit = [header.get('CUNIT%d' % n, '').strip().upper()
                       for n in axr]


class BeamAxes(FitsAxes):
    """
    Describes the FITS axes of a BEAM cube.

    In general, FORTRAN ordered indices are
    converted to C ordering
    (CRPIX and the individual axis indices)

    Any degree axes are converted to radians and
    grids for each axis are created.

    Inversions of the L, M, X and Y grids are supported
    if a minus sign is detected before the CUNIT in
    the FITS header.
    """

    def __init__(self, header=None):
        super(BeamAxes, self).__init__(header)

        # Check for custom irregular grid format.
        # Currently only implemented for FREQ dimension.
        irregular_grid = np.asarray([
            [header.get('G%s%d' % (self._ctype[i], j), None)
             for j in range(1, self._naxis[i]+1)]
            for i in range(self._ndims)])

        # Irregular grids are only valid if values exist for all grid points
        self._irreg = [all(x is not None for x in irregular_grid[i])
                       for i in range(self._ndims)]

        def _regular_grid(i):
            """ Construct a regular grid from a FitsAxes object and index """
            R = np.arange(0.0, float(self._naxis[i]))
            return (R - self._crpix[i])*self._cdelt[i] + self._crval[i]

        # Set up the grid
        self._grid = [_regular_grid(i) if not self._irreg[i]
                      else np.asarray(irregular_grid[i])
                      for i in range(self._ndims)]

        self._sign = [1.0]*self._ndims

        for i in range(self._ndims):
            # Convert any degree axes to radians
            if self._cunit[i] == 'DEG':
                self._cunit[i] = 'RAD'
                self._crval[i] = np.deg2rad(self._crval[i])
                self._cdelt[i] = np.deg2rad(self._cdelt[i])
                self._grid[i] = np.deg2rad(self._grid[i])

            # Flip the sign and correct the ctype if necessary
            if self._ctype[i].startswith('-'):
                self._ctype[i] = self._ctype[i][1]
                self._sign[i] = -1.0

    @property
    def ndims(self):
        return self._ndims

    @property
    def crpix(self):
        return self._crpix

    @property
    def naxis(self):
        return self._naxis

    @property
    def crval(self):
        return self._crval

    @property
    def cdelt(self):
        return self._cdelt

    @property
    def cunit(self):
        return self._cunit

    @property
    def ctype(self):
        return self._ctype

    @property
    def grid(self):
        return self._grid

    @property
    def sign(self):
        return self._sign


def beam_grids(header):
    """
    Extracts the FITS indices and grids for the beam dimensions
    in the supplied FITS ``header``.
    Specifically the axes specified by

    1. ``L`` or ``X`` CTYPE
    2. ``M`` or ``Y`` CTYPE
    3. ``FREQ`` CTYPE

    If the first two axes have a negative sign, such as ``-L``, the grid
    will be inverted.

    Any grids corresponding to axes with a CUNIT type of ``DEG``
    will be converted to radians.

    Parameters
    ----------
    header : :class:`~astropy.io.fits.Header` or dict
        FITS header object.

    Returns
    -------
    tuple
        Returns
        ((l_axis, l_grid), (m_axis, m_grid), (freq_axis, freq_grid))
        where the axis is the FORTRAN indexed FITS axis (1-indexed)
        and grid contains the values at each pixel along the axis.
    """
    beam_axes = BeamAxes(header)

    l = m = freq = None  # noqa

    # Find the relevant axes
    for i in range(beam_axes.ndims):
        if beam_axes.ctype[i] in ('L', 'X'):
            l = i  # noqa
        elif beam_axes.ctype[i] in ('M', 'Y'):
            m = i
        elif beam_axes.ctype[i] == "FREQ":
            freq = i

    # Complain if not found
    if l is None:
        raise ValueError("No L/X axis present in FITS header")

    if m is None:
        raise ValueError("No M/Y axis present in FITS header")

    if freq is None:
        raise ValueError("No FREQ axis present in FITS header")

    # Sign of L/M axes?
    l_sign = beam_axes.sign[l]
    m_sign = beam_axes.sign[m]

    # Obtain axes grids
    l_grid = beam_axes.grid[l]
    m_grid = beam_axes.grid[m]
    freq_grid = beam_axes.grid[freq]

    # flip the grid around if signs are different
    l_grid = np.flipud(l_grid) if l_sign == -1.0 else l_grid
    m_grid = np.flipud(m_grid) if m_sign == -1.0 else m_grid

    return ((l+1, l_grid), (m+1, m_grid), (freq+1, freq_grid))


class FitsFilenameTemplate(string.Template):
    """
    Overrides the ${identifer} braced pattern in the string Template
    with a $(identifier) braced pattern expected by FITS beam filename
    schema
    """
    pattern = r"""
    %(delim)s(?:
      (?P<escaped>%(delim)s)   |   # Escape sequence of two delimiters
      (?P<named>%(id)s)        |   # delimiter and a Python identifier
      \((?P<braced>%(id)s)\)   |   # delimiter and a braced identifier
      (?P<invalid>)                # Other ill-formed delimiter exprs
    )
    """ % {'delim': re.escape(string.Template.delimiter),
           'id': string.Template.idpattern}


CIRCULAR_CORRELATIONS = ('rr', 'rl', 'lr', 'll')
LINEAR_CORRELATIONS = ('xx', 'xy', 'yx', 'yy')
REIM = ('re', 'im')


def beam_filenames(filename_schema, polarisation_type):
    """
    Returns a dictionary of beam filename pairs,
    keyed on correlation,from the cartesian product
    of correlations and real, imaginary pairs

    Given ``beam_$(corr)_$(reim).fits`` returns:

    .. code-block:: python

        {
          'xx' : ('beam_xx_re.fits', 'beam_xx_im.fits'),
          'xy' : ('beam_xy_re.fits', 'beam_xy_im.fits'),
          ...
          'yy' : ('beam_yy_re.fits', 'beam_yy_im.fits'),
        }

    Given ``beam_$(CORR)_$(REIM).fits`` returns:

    .. code-block:: python

        {
          'xx' : ('beam_XX_RE.fits', 'beam_XX_IM.fits'),
          'xy' : ('beam_XY_RE.fits', 'beam_XY_IM.fits'),
          ...
          'yy' : ('beam_YY_RE.fits', 'beam_YY_IM.fits'),
        }

    Parameters
    ----------
    filename_schema : str
        String containing the filename schema.
    polarisation_type : {'linear', 'circular'}
        String defining the type of polarisation.

    Returns
    -------
    dict
        Dictionary of schema ``{correlation : (refile, imfile)}``
        mapping correlations to real and imaginary filename pairs

    """
    template = FitsFilenameTemplate(filename_schema)

    def _re_im_filenames(corr, template):
        try:
            return tuple(template.substitute(
                corr=corr.lower(), CORR=corr.upper(),
                reim=ri.lower(), REIM=ri.upper())
                for ri in REIM)
        except KeyError:
            raise ValueError("Invalid filename schema '%s'. "
                             "FITS Beam filename schemas "
                             "must follow forms such as "
                             "'beam_$(corr)_$(reim).fits' or "
                             "'beam_$(CORR)_$(REIM).fits." % filename_schema)

    if polarisation_type == 'linear':
        CORRELATIONS = LINEAR_CORRELATIONS
    elif polarisation_type == 'circular':
        CORRELATIONS = CIRCULAR_CORRELATIONS
    else:
        raise ValueError("Invalid polarisation_type '{}'. "
                         "Should be 'linear' or 'circular'")

    return OrderedDict((c, _re_im_filenames(c, template))
                       for c in CORRELATIONS)
