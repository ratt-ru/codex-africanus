# -*- coding: utf-8 -*-


from collections import OrderedDict
import string
import re

import numpy as np

from africanus.util.casa_types import STOKES_ID_MAP


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


def axis_and_sign(ax_str, default=None):
    """ Extract axis and sign from given axis string """
    if not ax_str:
        if default:
            return default, 1.0

        raise ValueError("Need default if ax_str is None")

    if not isinstance(ax_str, str):
        raise TypeError("ax_str must be a string")

    return (ax_str[1:], -1.0) if ax_str[0] == '-' else (ax_str, 1.0)


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

    Parameters
    ----------
    header : dict, optional
        FITS file header
    """

    def __init__(self, header=None):
        super(BeamAxes, self).__init__(header)
        # Check for custom irregular grid format.
        # Currently only implemented for FREQ dimension.
        irregular_grid = [np.asarray(
                    [header.get('G%s%d' % (self._ctype[i], j), None)
                     for j in range(1, self._naxis[i]+1)])
                for i in range(self._ndims)]

        # Irregular grids are only valid if values exist for all grid points
        self._irreg = [all(x is not None for x in irregular_grid[i])
                       for i in range(self._ndims)]

        def _regular_grid(i):
            """ Construct a regular grid from a FitsAxes object and index """
            R = np.arange(0.0, float(self._naxis[i]))
            return (R - self._crpix[i])*self._cdelt[i] + self._crval[i]

        self._grid = [None]*self._ndims

        for i in range(self._ndims):
            # Convert any degree axes to radians
            if self._cunit[i] == 'DEG':
                self._cunit[i] = 'RAD'
                self._crval[i] = np.deg2rad(self._crval[i])
                self._cdelt[i] = np.deg2rad(self._cdelt[i])

            # Set up the grid
            self._grid[i] = (_regular_grid(i) if not self._irreg[i]
                             else irregular_grid[i])

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


def beam_grids(header, l_axis=None, m_axis=None):
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
    l_axis : str
        FITS axis interpreted as the L axis. `L` and `X` are
        sensible values here. `-L` will invert the coordinate
        system on that axis.
    m_axis : str
        FITS axis interpreted as the M axis. `M` and `Y` are
        sensible values here. `-M` will invert the coordinate
        system on that axis.


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
        if beam_axes.ctype[i].upper() in ('L', 'X', 'PX'):
            l = i  # noqa
        elif beam_axes.ctype[i].upper() in ('M', 'Y', 'PY'):
            m = i
        elif beam_axes.ctype[i] == "FREQ":
            freq = i

    # Complain if not found
    if l is None:
        raise ValueError("No L/X/PX axis present in FITS header")

    if m is None:
        raise ValueError("No M/Y/PY axis present in FITS header")

    if freq is None:
        raise ValueError("No FREQ axis present in FITS header")

    # Sign of L/M axes?
    l_sign = axis_and_sign(l_axis, "L")[1]
    m_sign = axis_and_sign(m_axis, "M")[1]

    # Obtain axes grids
    l_grid = beam_axes.grid[l] * l_sign
    m_grid = beam_axes.grid[m] * m_sign
    freq_grid = beam_axes.grid[freq]

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


def _re_im_filenames(corr, template):
    filenames = []

    for ri in REIM:
        try:
            filename = template.substitute(corr=corr.lower(),
                                           CORR=corr.upper(),
                                           reim=ri.lower(),
                                           REIM=ri.upper())
        except KeyError:
            raise ValueError("Invalid filename schema '%s'. "
                             "FITS Beam filename schemas "
                             "must follow forms such as "
                             "'beam_$(corr)_$(reim).fits' or "
                             "'beam_$(CORR)_$(REIM).fits."
                             % template.template)
        else:
            filenames.append(filename)

    return filenames


def beam_filenames(filename_schema, corr_types):
    """
    Returns a dictionary of beam filename pairs,
    keyed on correlation,from the cartesian product
    of correlations and real, imaginary pairs

    Given ``beam_$(corr)_$(reim).fits`` returns:

    .. code-block:: python

        {
          'xx' : ['beam_xx_re.fits', 'beam_xx_im.fits'],
          'xy' : ['beam_xy_re.fits', 'beam_xy_im.fits'],
          ...
          'yy' : ['beam_yy_re.fits', 'beam_yy_im.fits'],
        }

    Given ``beam_$(CORR)_$(REIM).fits`` returns:

    .. code-block:: python

        {
          'xx' : ['beam_XX_RE.fits', 'beam_XX_IM.fits'],
          'xy' : ['beam_XY_RE.fits', 'beam_XY_IM.fits'],
          ...
          'yy' : ['beam_YY_RE.fits', 'beam_YY_IM.fits']),
        }

    Parameters
    ----------
    filename_schema : str
        String containing the filename schema.
    corr_types : list of integers
        list of integers defining the correlation type.

    Returns
    -------
    dict
        Dictionary of schema ``{correlation : (refile, imfile)}``
        mapping correlations to real and imaginary filename pairs

    """
    template = FitsFilenameTemplate(filename_schema)

    corr_names = []

    for corr_type in corr_types:
        try:
            corr_name = STOKES_ID_MAP[corr_type]
        except KeyError:
            raise ValueError("Unknown Stokes ID %d" % corr_type)
        else:
            corr_names.append(corr_name.lower())

    return OrderedDict((c, _re_im_filenames(c, template))
                       for c in corr_names)
