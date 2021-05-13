# -*- coding: utf-8 -*-


from operator import getitem

from africanus.averaging.bda_mapping import (
                bda_mapper as np_bda_mapper)
from africanus.averaging.bda_avg import (
                BDA_DOCS,
                row_average as np_bda_row_avg,
                row_chan_average as np_bda_row_chan_avg,
                AverageOutput as BDAAverageOutput,
                RowAverageOutput as BDARowAverageOutput,
                RowChanAverageOutput as BDARowChanAverageOutput)
from africanus.averaging.time_and_channel_mapping import (
                row_mapper as np_tc_row_mapper,
                channel_mapper as np_tc_channel_mapper)
from africanus.averaging.time_and_channel_avg import (
                row_average as np_tc_row_average,
                row_chan_average as np_tc_row_chan_average,
                chan_average as np_tc_chan_average,
                merge_flags as np_merge_flags,
                AVERAGING_DOCS as TC_AVERAGING_DOCS,
                AverageOutput as TcAverageOutput,
                ChannelAverageOutput as TcChannelAverageOutput,
                RowAverageOutput as TcRowAverageOutput,
                RowChanAverageOutput as TcRowChanAverageOutput)

from africanus.util.requirements import requires_optional

import numpy as np

try:
    from dask.base import tokenize
    import dask.array as da
    import dask.blockwise as db
    from dask.highlevelgraph import HighLevelGraph
except ImportError as e:
    dask_import_error = e
else:
    dask_import_error = None


def tc_chan_metadata(row_chan_arrays, chan_arrays, chan_bin_size):
    """ Create dask array with channel metadata for each chunk channel """
    chan_chunks = None

    for array in row_chan_arrays:
        if isinstance(array, tuple):
            for a in array:
                chan_chunks = a.chunks[1]
                break
        elif array is not None:
            chan_chunks = array.chunks[1]

        if chan_chunks is not None:
            break

    if chan_chunks is None:
        for array in chan_arrays:
            if array is not None:
                chan_chunks = array.chunks[0]
                break

    if chan_chunks is None:
        return None

    # Create a dask channel mapping structure
    name = "channel-mapper-" + tokenize(chan_chunks, chan_bin_size)
    layers = {(name, i): (np_tc_channel_mapper, c, chan_bin_size)
              for i, c in enumerate(chan_chunks)}
    graph = HighLevelGraph.from_collections(name, layers, ())
    chunks = (chan_chunks,)
    chan_mapper = da.Array(graph, name, chunks, dtype=object)

    return chan_mapper


def tc_row_mapper(time, interval, antenna1, antenna2,
                  flag_row=None, time_bin_secs=1.0):
    """ Create a dask row mapping structure for each row chunk """
    return da.blockwise(np_tc_row_mapper, ("row",),
                        time, ("row",),
                        interval, ("row",),
                        antenna1, ("row",),
                        antenna2, ("row",),
                        flag_row, None if flag_row is None else ("row",),
                        adjust_chunks={"row": lambda x: np.nan},
                        time_bin_secs=time_bin_secs,
                        meta=np.empty((0,), dtype=object),
                        dtype=object)


def _getitem_row(avg, idx, array, dims):
    """ Extract row-like arrays from a dask array of tuples """
    assert dims[0] == "row"

    name = ("row-average-getitem-%d-" % idx) + tokenize(avg, idx)
    layers = db.blockwise(getitem, name, dims,
                          avg.name, ("row",),
                          idx, None,
                          new_axes=dict(zip(dims[1:], array.shape[1:])),
                          numblocks={avg.name: avg.numblocks})
    graph = HighLevelGraph.from_collections(name, layers, (avg,))
    chunks = avg.chunks + tuple((s,) for s in array.shape[1:])

    return da.Array(graph, name, chunks,
                    meta=np.empty((0,)*len(dims), dtype=array.dtype),
                    dtype=array.dtype)


def _tc_row_average_wrapper(row_meta, ant1, ant2, flag_row,
                            time_centroid, exposure, uvw,
                            weight, sigma):
    return np_tc_row_average(row_meta, ant1, ant2, flag_row,
                             time_centroid, exposure,
                             uvw[0] if uvw is not None else None,
                             weight[0] if weight is not None else None,
                             sigma[0] if sigma is not None else None)


def tc_row_average(row_meta, ant1, ant2, flag_row=None,
                   time_centroid=None, exposure=None, uvw=None,
                   weight=None, sigma=None):
    """ Average row-based dask arrays """

    rd = ("row",)
    rcd = ("row", "corr")

    # (output, array, dims)
    args = [(False, row_meta, rd),
            (True, ant1, rd),
            (True, ant2, rd),
            (False, flag_row, None if flag_row is None else rd),
            (True, time_centroid, None if time_centroid is None else rd),
            (True, exposure, None if exposure is None else rd),
            (True, uvw, None if uvw is None else ("row", "uvw")),
            (True, weight, None if weight is None else rcd),
            (True, sigma, None if sigma is None else rcd)]

    avg = da.blockwise(_tc_row_average_wrapper, rd,
                       *(v for pair in args for v in pair[1:]),
                       align_arrays=False,
                       adjust_chunks={"row": lambda x: np.nan},
                       meta=np.empty((0,)*len(rd), dtype=object),
                       dtype=object)

    # ant1, ant2, time_centroid, exposure, uvw, weight, sigma
    out_args = [(a, dims) for out, a, dims in args if out is True]

    tuple_gets = [None if a is None else _getitem_row(avg, i, a, dims)
                  for i, (a, dims) in enumerate(out_args)]

    return TcRowAverageOutput(*tuple_gets)


def _getitem_row_chan(avg, idx, dtype):
    """ Extract (row,chan,corr) arrays from dask array of tuples """
    name = ("row-chan-average-getitem-%d-" % idx) + tokenize(avg, idx)
    dim = ("row", "chan", "corr")

    layers = db.blockwise(getitem, name, dim,
                          avg.name, dim,
                          idx, None,
                          numblocks={avg.name: avg.numblocks})

    graph = HighLevelGraph.from_collections(name, layers, (avg,))
    return da.Array(graph, name, avg.chunks,
                    meta=np.empty((0,)*len(dim), dtype=object),
                    dtype=dtype)


_row_chan_avg_dims = ("row", "chan", "corr")


def tc_row_chan_average(row_meta, chan_meta, flag_row=None, weight=None,
                        visibilities=None, flag=None,
                        weight_spectrum=None, sigma_spectrum=None,
                        chan_bin_size=1):
    """ Average (row,chan,corr)-based dask arrays """

    if chan_meta is None:
        return TcRowChanAverageOutput(None, None, None, None)

    # We don't know how many rows are in each row chunk,
    # but we can simply divide each channel chunk size by the bin size
    adjust_chunks = {
        "row": lambda r: np.nan,
        "chan": lambda c: (c + chan_bin_size - 1) // chan_bin_size
    }

    flag_row_dims = None if flag_row is None else ("row",)
    weight_dims = None if weight is None else ("row", "corr")
    vis_dims = None if visibilities is None else _row_chan_avg_dims
    flag_dims = None if flag is None else _row_chan_avg_dims
    ws_dims = None if weight_spectrum is None else _row_chan_avg_dims
    ss_dims = None if sigma_spectrum is None else _row_chan_avg_dims

    have_vis_tuple = False
    nvis_elements = 0

    # If we received a tuple of visibility arrays
    # convert them into an array of tuples of visibilities
    if isinstance(visibilities, (tuple, list)):
        if not all(isinstance(a, da.Array) for a in visibilities):
            raise ValueError("Visibility tuple must exclusively "
                             "contain dask arrays")

        have_vis_tuple = True
        nvis_elements = len(visibilities)
        meta = np.empty((0, 0, 0), visibilities[0].dtype)

        visibilities = da.blockwise(lambda *a: a, _row_chan_avg_dims,
                                    *[elem for a in visibilities
                                      for elem in (a, _row_chan_avg_dims)],
                                    meta=meta)

    avg = da.blockwise(np_tc_row_chan_average, _row_chan_avg_dims,
                       row_meta, ("row",),
                       chan_meta, ("chan",),
                       flag_row, flag_row_dims,
                       weight, weight_dims,
                       visibilities, vis_dims,
                       flag, flag_dims,
                       weight_spectrum, ws_dims,
                       sigma_spectrum, ss_dims,
                       align_arrays=False,
                       adjust_chunks=adjust_chunks,
                       meta=np.empty((0,)*len(_row_chan_avg_dims),
                                     dtype=object),
                       dtype=object)

    tuple_gets = (None if a is None else _getitem_row_chan(avg, i, a.dtype)
                  for i, a in enumerate([visibilities, flag,
                                         weight_spectrum,
                                         sigma_spectrum]))

    # If we received an array of tuples of visibilities
    # convert them into a tuple of visibility arrays
    if have_vis_tuple:
        tuple_gets = tuple(tuple_gets)
        vis_tuple = tuple_gets[0]
        tuple_vis = []

        for v in range(nvis_elements):
            v = da.blockwise(getitem, _row_chan_avg_dims,
                             vis_tuple, _row_chan_avg_dims,
                             v, None,
                             dtype=vis_tuple.dtype)
            tuple_vis.append(v)

        tuple_gets = (tuple(tuple_vis),) + tuple_gets[1:]

    return TcRowChanAverageOutput(*tuple_gets)


def _getitem_chan(avg, idx, dtype):
    """ Extract row-like arrays from a dask array of tuples """
    name = ("chan-average-getitem-%d-" % idx) + tokenize(avg, idx)
    layers = db.blockwise(getitem, name, ("chan",),
                          avg.name, ("chan",),
                          idx, None,
                          numblocks={avg.name: avg.numblocks})
    graph = HighLevelGraph.from_collections(name, layers, (avg,))
    return da.Array(graph, name, avg.chunks,
                    meta=np.empty((0,), dtype=dtype),
                    dtype=dtype)


def tc_chan_average(chan_meta, chan_freq=None, chan_width=None,
                    effective_bw=None, resolution=None, chan_bin_size=1):

    if chan_meta is None:
        return TcChannelAverageOutput(None, None)

    adjust_chunks = {
        "chan": lambda c: (c + chan_bin_size - 1) // chan_bin_size
    }

    cdim = ("chan",)

    avg = da.blockwise(np_tc_chan_average, cdim,
                       chan_meta, cdim,
                       chan_freq, None if chan_freq is None else cdim,
                       chan_width, None if chan_width is None else cdim,
                       effective_bw, None if effective_bw is None else cdim,
                       resolution, None if resolution is None else cdim,
                       adjust_chunks=adjust_chunks,
                       meta=np.empty((0,), dtype=object),
                       dtype=object)

    tuple_gets = (None if a is None else _getitem_chan(avg, i, a.dtype)
                  for i, a in enumerate([chan_freq, chan_width,
                                         effective_bw, resolution]))

    return TcChannelAverageOutput(*tuple_gets)


def merge_flags(flag_row, flag):
    """ Perform flag merging on dask arrays """
    if flag_row is None and flag is not None:
        return da.blockwise(np_merge_flags, "r",
                            flag_row, None,
                            flag, "rfc",
                            concatenate=True,
                            dtype=flag.dtype)
    elif flag_row is not None and flag is None:
        return da.blockwise(np_merge_flags, "r",
                            flag_row, "r",
                            None, None,
                            dtype=flag_row.dtype)
    elif flag_row is not None and flag is not None:
        return da.blockwise(np_merge_flags, "r",
                            flag_row, "r",
                            flag, "rfc",
                            concatenate=True,
                            dtype=flag_row.dtype)
    else:
        return None


@requires_optional("dask.array", dask_import_error)
def time_and_channel(time, interval, antenna1, antenna2,
                     time_centroid=None, exposure=None, flag_row=None,
                     uvw=None, weight=None, sigma=None,
                     chan_freq=None, chan_width=None,
                     effective_bw=None, resolution=None,
                     visibilities=None, flag=None,
                     weight_spectrum=None, sigma_spectrum=None,
                     time_bin_secs=1.0, chan_bin_size=1):

    row_chan_arrays = (visibilities, flag, weight_spectrum, sigma_spectrum)
    chan_arrays = (chan_freq, chan_width, effective_bw, resolution)

    # The flow of this function should match that of the numba
    # time_and_channel implementation

    # Merge flag_row and flag arrays
    flag_row = merge_flags(flag_row, flag)

    # Generate row mapping metadata
    row_meta = tc_row_mapper(time, interval,
                             antenna1, antenna2,
                             flag_row=flag_row,
                             time_bin_secs=time_bin_secs)

    # Generate channel mapping metadata
    chan_meta = tc_chan_metadata(row_chan_arrays, chan_arrays, chan_bin_size)

    # Average row data
    row_data = tc_row_average(row_meta, antenna1, antenna2,
                              flag_row=flag_row,
                              time_centroid=time_centroid,
                              exposure=exposure, uvw=uvw,
                              weight=weight, sigma=sigma)

    # Average channel data
    row_chan_data = tc_row_chan_average(row_meta, chan_meta,
                                        flag_row=flag_row, weight=weight,
                                        visibilities=visibilities, flag=flag,
                                        weight_spectrum=weight_spectrum,
                                        sigma_spectrum=sigma_spectrum,
                                        chan_bin_size=chan_bin_size)

    chan_data = tc_chan_average(chan_meta,
                                chan_freq=chan_freq,
                                chan_width=chan_width,
                                effective_bw=effective_bw,
                                resolution=resolution)

    # Merge output tuples
    return TcAverageOutput(_getitem_row(row_meta, 1, time, ("row",)),
                           _getitem_row(row_meta, 2, interval, ("row",)),
                           (_getitem_row(row_meta, 3, flag_row, ("row",))
                            if flag_row is not None else None),
                           row_data.antenna1,
                           row_data.antenna2,
                           row_data.time_centroid,
                           row_data.exposure,
                           row_data.uvw,
                           row_data.weight,
                           row_data.sigma,
                           chan_data.chan_freq,
                           chan_data.chan_width,
                           chan_data.effective_bw,
                           chan_data.resolution,
                           row_chan_data.visibilities,
                           row_chan_data.flag,
                           row_chan_data.weight_spectrum,
                           row_chan_data.sigma_spectrum)


def _bda_mapper_wrapper(time, interval, ant1, ant2,
                        uvw, chan_width, chan_freq,
                        max_uvw_dist, flag_row,
                        max_fov=None,
                        decorrelation=None,
                        time_bin_secs=None,
                        min_nchan=None):
    return np_bda_mapper(time, interval, ant1, ant2,
                         None if uvw is None else uvw[0],
                         chan_width[0], chan_freq[0],
                         max_uvw_dist=max_uvw_dist,
                         flag_row=flag_row,
                         max_fov=max_fov,
                         decorrelation=decorrelation,
                         time_bin_secs=time_bin_secs,
                         min_nchan=min_nchan)


def bda_mapper(time, interval, antenna1, antenna2, uvw,
               chan_width, chan_freq,
               max_uvw_dist,
               flag_row=None, max_fov=None,
               decorrelation=None,
               time_bin_secs=None,
               min_nchan=None):
    """ Createask row mapping structure for each row chunk """
    return da.blockwise(_bda_mapper_wrapper, ("row",),
                        time, ("row",),
                        interval, ("row",),
                        antenna1, ("row",),
                        antenna2, ("row",),
                        uvw, ("row", "uvw"),
                        chan_width, ("chan",),
                        chan_freq, ("chan",),
                        max_uvw_dist, None if max_uvw_dist is None else (),
                        flag_row, None if flag_row is None else ("row",),
                        max_fov=max_fov,
                        decorrelation=decorrelation,
                        time_bin_secs=time_bin_secs,
                        min_nchan=min_nchan,
                        adjust_chunks={"row": lambda x: np.nan},
                        meta=np.empty((0, 0), dtype=object))


def _bda_row_average_wrapper(meta, ant1, ant2, flag_row,
                             time_centroid, exposure, uvw,
                             weight, sigma):
    return np_bda_row_avg(meta, ant1, ant2, flag_row,
                          time_centroid, exposure,
                          None if uvw is None else uvw[0],
                          None if weight is None else weight[0],
                          None if sigma is None else sigma[0])


def _ragged_row_getitem(avg, idx, meta):
    return avg[idx][meta.offsets[:-1], ...]


def _bda_getitem_row(avg, idx, array, dims, meta, format="flat"):
    """ Extract row-like arrays from a dask array of tuples """
    assert dims[0] == "row"

    name = "row-average-getitem-%s-" % idx
    name += tokenize(avg, idx)
    new_axes = dict(zip(dims[1:], array.shape[1:]))
    numblocks = {avg.name: avg.numblocks}

    if format == "flat":
        layers = db.blockwise(getitem, name, dims,
                              avg.name, ("row",),
                              idx, None,
                              new_axes=new_axes,
                              numblocks=numblocks)

    elif format == "ragged":
        numblocks[meta.name] = meta.numblocks
        layers = db.blockwise(_ragged_row_getitem, name, dims,
                              avg.name, ("row",),
                              idx, None,
                              meta.name, ("row",),
                              new_axes=new_axes,
                              numblocks=numblocks)
    else:
        raise ValueError("Invalid format %s" % format)

    graph = HighLevelGraph.from_collections(name, layers, (avg,))
    chunks = avg.chunks + tuple((s,) for s in array.shape[1:])
    meta = np.empty((0,)*len(dims), dtype=array.dtype)

    return da.Array(graph, name, chunks, meta=meta)


def _ragged_row_chan_getitem(avg, idx, meta):
    data = avg[idx]

    if isinstance(data, tuple):
        return tuple({"r%d" % (r+1): d[None, s:e, ...]
                      for r, (s, e)
                      in enumerate(zip(meta.offsets[:-1], meta.offsets[1:]))}
                     for d in data)

    return {"r%d" % (r+1): data[None, s:e, ...]
            for r, (s, e)
            in enumerate(zip(meta.offsets[:-1], meta.offsets[1:]))}


def _bda_getitem_row_chan(avg, idx, dtype, format, avg_meta, nchan):
    """ Extract (row, corr) arrays from dask array of tuples """
    f = BDARowChanAverageOutput._fields[idx]
    name = "row-chan-average-getitem-%s-%s-" % (f, format)
    name += tokenize(avg, idx)

    if format == "flat":
        dims = ("row", "corr")
        new_axes = None

        layers = db.blockwise(getitem, name, dims,
                              avg.name, ("row", "corr"),
                              idx, None,
                              numblocks={avg.name: avg.numblocks})

        chunks = avg.chunks
        meta = np.empty((0, 0), dtype=object)
    elif format == "ragged":
        dims = ("row", "chan", "corr")
        new_axes = {"chan": nchan}

        layers = db.blockwise(_ragged_row_chan_getitem, name, dims,
                              avg.name, ("row", "corr"),
                              idx, None,
                              avg_meta.name, ("row",),
                              new_axes=new_axes,
                              numblocks={
                                  avg.name: avg.numblocks,
                                  avg_meta.name: avg_meta.numblocks})

        chunks = (avg.chunks[0], (nchan,), avg.chunks[1])
        meta = np.empty((0, 0, 0), dtype=object)
    else:
        raise ValueError("Invalid format %s" % format)

    graph = HighLevelGraph.from_collections(name, layers, (avg,))
    return da.Array(graph, name, chunks, meta=meta)


def bda_row_average(meta, ant1, ant2, flag_row=None,
                    time_centroid=None, exposure=None, uvw=None,
                    weight=None, sigma=None,
                    format="flat"):
    """ Average row-based dask arrays """

    rd = ("row",)
    rcd = ("row", "corr")

    # (output, array, dims)
    args = [(False, meta, ("row",)),
            (True, ant1, rd),
            (True, ant2, rd),
            (False, flag_row, None if flag_row is None else rd),
            (True, time_centroid, None if time_centroid is None else rd),
            (True, exposure, None if exposure is None else rd),
            (True, uvw, None if uvw is None else ("row", "uvw")),
            (True, weight, None if weight is None else rcd),
            (True, sigma, None if sigma is None else rcd)]

    avg = da.blockwise(_bda_row_average_wrapper, rd,
                       *(v for pair in args for v in pair[1:]),
                       align_arrays=False,
                       adjust_chunks={"row": lambda x: np.nan},
                       meta=np.empty((0,)*len(rd), dtype=object),
                       dtype=object)

    # ant1, ant2, time_centroid, exposure, uvw, weight, sigma
    out_args = [(a, dims) for out, a, dims in args if out is True]

    tuple_gets = [None if a is None
                  else _bda_getitem_row(avg, i, a, dims, meta, format=format)
                  for i, (a, dims) in enumerate(out_args)]

    return BDARowAverageOutput(*tuple_gets)


def _bda_row_chan_average_wrapper(avg_meta, flag_row, weight,
                                  vis, flag,
                                  weight_spectrum,
                                  sigma_spectrum):
    return np_bda_row_chan_avg(
                avg_meta, flag_row, weight,
                None if vis is None else vis[0],
                None if flag is None else flag[0],
                None if weight_spectrum is None else weight_spectrum[0],
                None if sigma_spectrum is None else sigma_spectrum[0])


def bda_row_chan_average(avg_meta, flag_row=None, weight=None,
                         visibilities=None, flag=None,
                         weight_spectrum=None,
                         sigma_spectrum=None,
                         format="flat"):
    """ Average (row,chan,corr)-based dask arrays """
    if all(v is None for v in (visibilities,
                               flag,
                               weight_spectrum,
                               sigma_spectrum)):
        return BDARowChanAverageOutput(None, None, None, None)

    # We don't know how many rows are in each row chunk,
    adjust_chunks = {"row": lambda r: np.nan}

    if format == "flat":
        bda_dims = ("row", "corr")
    elif format == "ragged":
        bda_dims = ("row", "chan", "corr")
    else:
        raise ValueError(f"Invalid format {format}")

    flag_row_dims = None if flag_row is None else ("row",)
    weight_dims = None if weight is None else ("row", "corr")
    vis_dims = None if visibilities is None else _row_chan_avg_dims
    flag_dims = None if flag is None else _row_chan_avg_dims
    ws_dims = None if weight_spectrum is None else _row_chan_avg_dims
    ss_dims = None if sigma_spectrum is None else _row_chan_avg_dims

    have_vis_tuple = False
    nvis_elements = 0

    if isinstance(visibilities, da.Array):
        nchan = visibilities.shape[1]
    elif isinstance(visibilities, (tuple, list)):
        nchan = visibilities[0].shape[1]

        if not all(isinstance(a, da.Array) for a in visibilities):
            raise ValueError("Visibility tuple must exclusively "
                             "contain dask arrays")

        # If we received a tuple of visibility arrays
        # convert them into an array of tuples of visibilities
        have_vis_tuple = True
        nvis_elements = len(visibilities)
        meta = np.empty((0,)*len(bda_dims), dtype=visibilities[0].dtype)

        visibilities = da.blockwise(lambda *a: a, _row_chan_avg_dims,
                                    *[elem for a in visibilities
                                      for elem in (a, vis_dims)],
                                    meta=meta)
    elif isinstance(flag, da.Array):
        nchan = flag.shape[1]
    elif isinstance(weight_spectrum, da.Array):
        nchan = weight_spectrum[1]
    elif isinstance(sigma_spectrum, da.Array):
        nchan = sigma_spectrum[1]
    else:
        raise ValueError("Couldn't infer nchan")

    avg = da.blockwise(_bda_row_chan_average_wrapper, ("row", "corr"),
                       avg_meta, ("row",),
                       flag_row, flag_row_dims,
                       weight, weight_dims,
                       visibilities, vis_dims,
                       flag, flag_dims,
                       weight_spectrum, ws_dims,
                       sigma_spectrum, ss_dims,
                       align_arrays=False,
                       adjust_chunks=adjust_chunks,
                       meta=np.empty((0, 0), dtype=object),
                       dtype=object)

    tuple_gets = (None if a is None else
                  _bda_getitem_row_chan(avg, i, a.dtype,
                                        format, avg_meta, nchan)
                  for i, a in enumerate([visibilities, flag,
                                         weight_spectrum,
                                         sigma_spectrum]))

    # If we received an array of tuples of visibilities
    # convert them into a tuple of visibility arrays
    if have_vis_tuple:
        tuple_gets = tuple(tuple_gets)
        vis_array = tuple_gets[0]
        tuple_vis = []

        for v in range(nvis_elements):
            v = da.blockwise(getitem, bda_dims,
                             vis_array, bda_dims,
                             v, None,
                             dtype=vis_array.dtype)
            tuple_vis.append(v)

        tuple_gets = (tuple(tuple_vis),) + tuple_gets[1:]

    return BDARowChanAverageOutput(*tuple_gets)


@requires_optional("dask.array", dask_import_error)
def bda(time, interval, antenna1, antenna2,
        time_centroid=None, exposure=None, flag_row=None,
        uvw=None, weight=None, sigma=None,
        chan_freq=None, chan_width=None,
        effective_bw=None, resolution=None,
        visibilities=None, flag=None,
        weight_spectrum=None,
        sigma_spectrum=None,
        max_uvw_dist=None,
        max_fov=3.0,
        decorrelation=0.98,
        time_bin_secs=None,
        min_nchan=1,
        format="flat"):

    if uvw is None:
        raise ValueError("uvw must be supplied")

    if chan_width is None:
        raise ValueError("chan_width must be supplied")

    if chan_freq is None:
        raise ValueError("chan_freq must be supplied")

    if not len(chan_width.chunks[0]) == 1:
        raise ValueError("Chunking in channel is not "
                         "currently supported.")

    if max_uvw_dist is None:
        max_uvw_dist = da.sqrt((uvw**2).sum(axis=1)).max()

    # row_chan_arrays = (vis, flag, weight_spectrum, sigma_spectrum)
    # chan_arrays = (chan_freq, chan_width, effective_bw, resolution)

    # The flow of this function should match that of the numba
    # bda implementation

    # Merge flag_row and flag arrays
    flag_row = merge_flags(flag_row, flag)

    # Generate row mapping metadata
    meta = bda_mapper(time, interval, antenna1, antenna2, uvw,
                      chan_width, chan_freq,
                      max_uvw_dist,
                      flag_row=flag_row,
                      max_fov=max_fov,
                      decorrelation=decorrelation,
                      time_bin_secs=time_bin_secs,
                      min_nchan=min_nchan)

    # Average row data
    row_data = bda_row_average(meta, antenna1, antenna2,
                               flag_row=flag_row,
                               time_centroid=time_centroid,
                               exposure=exposure,
                               uvw=uvw,
                               weight=weight, sigma=sigma,
                               format=format)

    # Average channel data
    row_chan_data = bda_row_chan_average(meta,
                                         flag_row=flag_row, weight=weight,
                                         visibilities=visibilities, flag=flag,
                                         weight_spectrum=weight_spectrum,
                                         sigma_spectrum=sigma_spectrum,
                                         format=format)

    # chan_data = chan_average(chan_meta,
    #                          chan_freq=chan_freq,
    #                          chan_width=chan_width,
    #                          effective_bw=effective_bw,
    #                          resolution=resolution)

    fake_map = da.zeros((time.shape[0], chan_width.shape[0]),
                        chunks=time.chunks + chan_width.chunks,
                        dtype=np.uint32)

    fake_ints = da.zeros_like(time, dtype=np.uint32)
    fake_floats = da.zeros_like(chan_width)

    meta_map = _bda_getitem_row(meta, 0, fake_map, ("row", "chan"), meta)
    meta_offsets = _bda_getitem_row(meta, 1, fake_ints, ("row",), meta)
    meta_decorr_cw = _bda_getitem_row(meta, 2, fake_floats, ("row",), meta)
    meta_time = _bda_getitem_row(meta, 3, time, ("row",),
                                 meta, format=format)
    meta_interval = _bda_getitem_row(meta, 4, interval, ("row",),
                                     meta, format=format)
    meta_chan_width = _bda_getitem_row(meta, 5, chan_width, ("row",), meta)
    meta_flag_row = (_bda_getitem_row(meta, 6, flag_row, ("row",),
                                      meta, format=format)
                     if flag_row is not None else None)

    # Merge output tuples
    return BDAAverageOutput(meta_map,
                            meta_offsets,
                            meta_decorr_cw,
                            meta_time,
                            meta_interval,
                            meta_chan_width,
                            meta_flag_row,
                            row_data.antenna1,
                            row_data.antenna2,
                            row_data.time_centroid,
                            row_data.exposure,
                            row_data.uvw,
                            row_data.weight,
                            row_data.sigma,
                            # None,  # chan_data.chan_freq,
                            # None,  # chan_data.chan_width,
                            # None,  # chan_data.effective_bw,
                            # None,  # chan_data.resolution,
                            row_chan_data.visibilities,
                            row_chan_data.flag,
                            row_chan_data.weight_spectrum,
                            row_chan_data.sigma_spectrum)


try:
    time_and_channel.__doc__ = TC_AVERAGING_DOCS.substitute(
                                    array_type=":class:`dask.array.Array`")
    bda.__doc__ = BDA_DOCS.substitute(array_type=":class:`dask.array.Array`")
except AttributeError:
    pass
