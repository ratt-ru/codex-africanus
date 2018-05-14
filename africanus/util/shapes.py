# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def corr_shape(ncorr, corr_shape):
    """
    Returns the shape of the correlations, given
    ``ncorr`` and the type of correlation shape requested

    Parameters
    ----------
    ncorr : integer
        Number of correlations
    corr_shape : {'flat', 'matrix'}
        Shape of output correlations


    Returns
    -------
    tuple
        Shape tuple describing the correlation dimensions

        * If ``flat`` returns :code:`(ncorr,)`
        * If ``matrix`` returns

            * :code:`(1,)` if :code:`ncorr == 1`
            * :code:`(2,)` if :code:`ncorr == 2`
            * :code:`(2,2)` if :code:`ncorr == 4`


    """
    if corr_shape == "flat":
        return (ncorr,)
    elif corr_shape == "matrix":
        if ncorr == 1:
            return (1,)
        elif ncorr == 2:
            return (2,)
        elif ncorr == 4:
            return (2, 2)
        else:
            raise ValueError("ncorr not in (1, 2, 4)")
    else:
        raise ValueError("corr_shape must be 'flat' or 'matrix'")
