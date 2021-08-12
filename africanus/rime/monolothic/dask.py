from operator import concat
import dask.array as da
import numpy as np

from africanus.rime.monolothic.rime import rime_factory


def rime_dask_wrapper(factory, names, nconcat_dims, *args):
    assert len(names) == len(args)
    out = factory(**dict(zip(names, args)))
    return out[(slice(None),)*out.ndim + (None,)*nconcat_dims]


def rime(terms=None, **kwargs):
    factory = rime_factory(terms=terms)
    names, args = factory.dask_blockwise_args(**kwargs)

    dims = ("source", "row", "chan", "corr")
    concat_dims = set(d for t in args[1::2] for d in t) - set(dims)

    out = da.blockwise(rime_dask_wrapper, dims + tuple(concat_dims),
                       factory, None,
                       names, None,
                       len(concat_dims) + 1, None,
                       *args,
                       concatenate=False,
                       adjust_chunks={d: 1 for d in concat_dims},
                       dtype=np.complex64)

    axes = tuple(range(len(dims), len(dims) + len(concat_dims)))
    return out.sum(axis=axes)