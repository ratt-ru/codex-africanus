# -*- coding: utf-8 -*-


def aggregate_chunks(chunks, max_chunks):
    """
    Aggregate dask ``chunks`` together into chunks no larger than
    ``max_chunks``.

    .. code-block:: python

        chunks, max_c = ((3,4,6,3,6,7),(1,1,1,1,1,1)), (10,3)
        expected = ((7,9,6,7), (2,2,1,1))
        assert aggregate_chunks(chunks, max_c) == expected


    Parameters
    ----------
    chunks : sequence of tuples or tuple
    max_chunks : sequence of ints or int

    Returns
    -------
    sequence of tuples or tuple

    """

    if isinstance(max_chunks, int):
        chunks = (chunks,)
        max_chunks = (max_chunks,)

    singleton = True if len(max_chunks) == 1 else False

    if len(chunks) != len(max_chunks):
        raise ValueError("len(chunks) != len(max_chunks)")

    if not all(len(chunks[0]) == len(c) for c in chunks):
        raise ValueError("Number of chunks do not match")

    agg_chunks = [[] for _ in max_chunks]
    agg_chunk_counts = [0] * len(max_chunks)
    chunk_scratch = [0] * len(max_chunks)
    ndim = len(chunks[0])

    # For each chunk dimension
    for di in range(ndim):
        # For each chunk
        aggregate = False

        for ci, chunk in enumerate(chunks):
            chunk_scratch[ci] = agg_chunk_counts[ci] + chunk[di]
            if chunk_scratch[ci] > max_chunks[ci]:
                aggregate = True

        if aggregate:
            for ci, chunk in enumerate(chunks):
                agg_chunks[ci].append(agg_chunk_counts[ci])
                agg_chunk_counts[ci] = chunk[di]
        else:
            for ci, chunk in enumerate(chunks):
                agg_chunk_counts[ci] = chunk_scratch[ci]

    # Do the final aggregation
    for ci, chunk in enumerate(chunks):
        agg_chunks[ci].append(agg_chunk_counts[ci])
        agg_chunk_counts[ci] = chunk[di]

    agg_chunks = tuple(tuple(ac) for ac in agg_chunks)

    return agg_chunks[0] if singleton else agg_chunks


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
