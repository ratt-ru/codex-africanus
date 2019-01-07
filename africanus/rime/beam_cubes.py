#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from itertools import product

import numpy as np

try:
    from scipy import interpolate
    from scipy.ndimage import interpolation
except ImportError as e:
    scipy_import_error = e
else:
    scipy_import_error = None

from africanus.util.requirements import requires_optional


@requires_optional("scipy", scipy_import_error)
def beam_cube_dde(beam, coords, l_grid, m_grid, freq_grid,
                  spline_order=1, mode='nearest'):
    """
    Computes Direction Dependent Effects (E) by sampling
    complex values in ``beam`` at the coordinates ``coords``.


    Both real and imaginary beam values are sampled at
    the given coordinates and normalised to form a
    `mean of circular quantities
    <https://en.wikipedia.org/wiki/Mean_of_circular_quantities>`_.

    ``l_grid``, ``m_grid`` and ``freq_grid`` can be obtained from
    :func:`~africanus.util.beams.beam_grids`.

    Parameters
    ----------
    beam : :class:`numpy.ndarray`
        complex beam cube of shape
        :code:`(beam_lw, beam_mh, beam_nud, corr_1, corr_2)` where
        ``beam_lw`` is the grid width of the l dimension,
        ``beam_mh`` is the grid height of the m dimension and
        ``beam_nud`` is the grid depth of the frequency dimension.
        Either ``corr_1`` or both ``corr_1`` and ``corr_2`` may be
        present, representing 1, 2 or 2x2 correlations respectively.
    coords : :class:`numpy.ndarray`
        beam cube coordinates of shape :code:`(coords, dim_1, ..., dim_n)`
        where ``coord`` always has size 3 and refers to `(l,m,frequency)`.
    l_grid : :class:`numpy.ndarray`
        Monotonically *increasing* or *decreasing* grid values for
        the l axis, with shape :code:`(beam_lw,)`.
        If decreasing, the
    m_grid : :class:`numpy.ndarray`
        Monotonically *increasing* or *decreasing* grid values for
        the m axis, with shape :code:`(beam_mh,)`
    freq_grid : :class:`numpy.ndarray`
        Monotonically increasing grid values for the frequency axis,
        with shape :code:`(beam_nud,)`
    spline_order : int
        Spline order to use in
        :func:`scipy.ndimage.interpolation.map_coordinates`.
        Defaults to 1 ('linear')
    mode : str
        Border mode to use in
        :func:`scipy.ndimage.interpolation.map_coordinates`
        Defaults to 'nearest'

    Returns
    -------
    ddes : :class:`numpy.ndarray`
        Sampled complex beam values at the specified coordinates with
        shape :code:`(dim_1, ..., dim_n, corr_1, corr_2)`
    """
    if not np.iscomplexobj(beam):
        raise ValueError("beam is not complex")

    l_diff = np.diff(l_grid)
    l_inc = np.all(l_diff > 0)
    l_dec = np.all(l_diff < 0)

    if not (l_inc or l_dec):
        raise ValueError("l_grid is not monotonically increasing/decreasing")

    m_diff = np.diff(m_grid)
    m_inc = np.all(m_diff > 0)
    m_dec = np.all(m_diff < 0)

    if not (m_inc or m_dec):
        raise ValueError("m_grid is not monotonically increasing/decreasing")

    freq_diff = np.diff(freq_grid)
    freq_inc = np.all(freq_diff > 0)

    if not freq_inc:
        raise ValueError("freq_grid is not monotonically increasing")

    # interp1d works on monotically increasing/decreasing values
    #
    # .. code-block:: python
    #
    #    values = np.asarray([1.0, 0.7, 0.2, 0.0, -0.4, -1.0])
    #    values = np.flipud(values)
    #    grid = np.arange(values.size)
    #
    #    initial = np.stack((values, grid))
    #    interp = interp1d(values, grid, bounds_error=False,
    #                                    fill_value='extrapolate')
    #    assert np.all(initial == np.stack((values,interp(values))))

    l_interp = interpolate.interp1d(l_grid, np.arange(l_grid.size),
                                    'linear', bounds_error=False,
                                    fill_value='extrapolate',
                                    assume_sorted=l_inc)

    m_interp = interpolate.interp1d(m_grid, np.arange(m_grid.size),
                                    'linear', bounds_error=False,
                                    fill_value='extrapolate',
                                    assume_sorted=m_inc)

    freq_interp = interpolate.interp1d(freq_grid, np.arange(freq_grid.size),
                                       'linear', bounds_error=False,
                                       fill_value='extrapolate',
                                       assume_sorted=True)

    head, tail = coords.shape[0], coords.shape[1:]

    if not head == 3:
        raise ValueError("coord axis must have size 3 "
                         "representing l, m and frequency")

    # Flatten coordinates
    coords = coords.reshape(head, -1)

    # TODO(sjperkins)
    # This scaling code might actually be more suited
    # to transform_sources.

    # LM coordinates must be scaled if
    # they lie outside the beam cube.
    # Check for frequency coordinates (index 2)
    # that lie below or above
    below = coords[2, :] < freq_grid[0]
    above = coords[2, :] > freq_grid[-1]

    # Scaling factors for frequencies above and below
    below_scale = coords[2, below] / freq_grid[0]
    above_scale = coords[2, above] / freq_grid[-1]

    # Now scale L and M (0 and 1) by frequency scaling
    coords[0:2, below] *= below_scale
    coords[0:2, above] *= above_scale

    # Convert to grid coordinates
    grid_coords = np.empty_like(coords)
    grid_coords[0, :] = l_interp(coords[0, :])
    grid_coords[1, :] = m_interp(coords[1, :])
    grid_coords[2, :] = freq_interp(coords[2, :])

    # Create beam and result indices
    corr_dims = beam.shape[3:]
    all_ = slice(None)
    result_all = tuple(all_ for _ in range(len(tail)))

    corr_indices = list(product(*(range(d) for d in corr_dims)))
    if len(corr_indices) > 0:
        beam_indices = tuple((all_,) + cp for cp in corr_indices)
        result_indices = tuple(result_all + cp for cp in corr_indices)
    else:
        beam_indices = ((all_,),)
        result_indices = ((result_all,),)

    # Allocate output array
    result = np.empty(tail + corr_dims, dtype=beam.dtype)

    prefilter = spline_order == 1

    # For each correlation
    for bi, ri in zip(beam_indices, result_indices):
        re = result[ri].real
        im = result[ri].imag

        # Interpolate real and imaginary beams
        re.flat[:] = interpolation.map_coordinates(beam[bi].real, grid_coords,
                                                   order=spline_order,
                                                   prefilter=prefilter,
                                                   mode=mode)
        im.flat[:] = interpolation.map_coordinates(beam[bi].imag, grid_coords,
                                                   order=spline_order,
                                                   prefilter=prefilter,
                                                   mode=mode)

        # This computes a mean of circular quantities
        # and the following should hold
        #
        # .. code-block:: python
        #
        #   phase = np.arctan2(re, im)
        #   re == np.cos(phase)
        #   im == np.sin(phase)

        # Compute the amplitude
        amplitude = np.sqrt(re**2 + im**2)
        # Handle divide by zero when normalising
        amplitude[amplitude == 0.0] = 1.0

        # Normalise real and imaginary components
        re /= amplitude
        im /= amplitude

    return result
