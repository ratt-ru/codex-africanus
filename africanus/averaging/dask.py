# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from operator import getitem

from africanus.averaging.time_and_channel_avg import (
                        time_and_channel as np_time_and_channel,
                        TIME_AND_CHANNEL_DOCS)
from africanus.util.requirements import requires_optional

import numpy as np

try:
    import dask.array as da
except ImportError as e:
    dask_import_error = e
else:
    dask_import_error = None


@requires_optional("dask.array", dask_import_error)
def time_and_channel(time, ant1, ant2, vis, flags,
                     avg_time=None, avg_chan=None,
                     return_time=False,
                     return_antenna=False):

    adjust_chunks = {
        # We're not really sure how many rows we'll end up with in each chunk
        "row": tuple(np.nan for c in vis.chunks[0]),
        # Channel averaging is more predictable
        "chan": tuple((c + avg_chan - 1) // avg_chan for c in vis.chunks[1])
    }

    if return_time or return_antenna:
        raise NotImplementedError("return of time and antenna is not "
                                  "yet implemented in dask")

    corr_dims = tuple("corr-%d" % i for i in range(len(vis.shape[2:])))

    return da.blockwise(np_time_and_channel, ("row", "chan") + corr_dims,
                        time, ("row",),
                        ant1, ("row",),
                        ant2, ("row",),
                        vis, ("row", "chan") + corr_dims,
                        flags, ("row", "chan") + corr_dims,
                        avg_time=avg_time,
                        avg_chan=avg_chan,
                        return_time=return_time,
                        return_antenna=return_antenna,
                        adjust_chunks=adjust_chunks,
                        dtype=vis.dtype)


try:
    time_and_channel.__doc__ = TIME_AND_CHANNEL_DOCS.substitute(
                                    array_type=":class:`dask.array.Array`")
except AttributeError:
    pass
