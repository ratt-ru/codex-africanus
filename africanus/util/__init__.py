


def corr_shape(ncorr, corr_shape):
    """
    Returns the shape of the correlations, given
    ``ncorr`` and the type of correlation shape requested

    Parameters
    ----------
    ncorr : integer
        Number of polarisations
    corr_shape : {'flat', 'matrix'}
        If 'flat' returns a single dimension.
        If 'matrix' returns a square matrix of dimensions

    Returns
    -------
    tuple
        Shape tuple describing the correlation dimensions
    """
    if corr_shape == "flat":
        return (ncorr,)
    elif corr_shape == "matrix":
        if ncorr == 1:
            return (1,)
        elif ncorr == 2:
            return (2, 1)
        elif ncorr == 4:
            return (2, 2)
        else:
            raise ValueError("ncorr not in (1, 2, 4)")
    else:
        raise ValueError("corr_shape must be 'flat' or 'matrix'")
