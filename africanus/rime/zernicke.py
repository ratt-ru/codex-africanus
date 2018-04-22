import numba

@numba.jit
def zernicke_dde(coords, m, n, p, phi):
    """
    Parameters:
    -----------
    coords: source coordinates of shape (3, src, time, ant, chan)
    m and n: the two variables describing the order of the Zernicke function
    p: radial distance
    phi: azimuthal angle

    Returns:
    --------
    Zernicke polynomial with shape (src, time, ant, chan
    """
    if len(coords) != 3:
        raise ValueError("coords must be of shape (3, src, time, ant, chan.")
    _, nsrc, ntime, na, nchan = coords.shape

    return  #return shape: (src, time, ant, chan)