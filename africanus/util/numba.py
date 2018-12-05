# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from numba import types


def is_numba_type_none(arg):
    """
    Returns True if the numba type represents None


    Parameters
    ----------
    arg : :class:`numba.Type`
        The numba type

    Returns
    -------
    boolean
        True if the type represents None
    """
    return (isinstance(arg, types.misc.NoneType) or
            (isinstance(arg, types.misc.Omitted) and arg.value is None))
