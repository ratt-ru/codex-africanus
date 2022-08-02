import numpy as np

from africanus.util.requirements import requires_optional
from africanus.experimental.rime.fused.core import (
    RimeFactory, consolidate_args)

try:
    import dask.array as da
except ImportError as e:
    opt_import_err = e
else:
    opt_import_err = None


def rime_dask_wrapper(factory, names, nconcat_dims, *args):
    # Call the rime factory
    assert len(names) == len(args)
    out = factory(**dict(zip(names, args)))
    # (1) Reintroduce source dimension,
    # (2) slice the existing dimes
    # (3) expand by the contraction dims which will
    #     be removed in the later dask reduction
    return out[(None,) + (slice(None),)*out.ndim + (None,)*nconcat_dims]


@requires_optional("dask.array", opt_import_err)
def rime(rime_spec, *args, **kw):
    """Like :func:`~africanus.experimental.rime.fused.core.rime`, but for
    a dask paradigm"""
    factory = RimeFactory(rime_spec)
    names, args = factory.dask_blockwise_args(**consolidate_args(args, kw))

    dims = ("source", "row", "chan", "corr")
    contract_dims = set(d for ds in args[1::2] if ds is not None for d in ds)
    contract_dims -= set(dims)
    out_dims = dims + tuple(contract_dims)

    # Source and concatenation dimension are reduced to 1 element
    adjust_chunks = {"source": 1, **{d: 1 for d in contract_dims}}

    # This is needed otherwise, dask will call rime_dask_wrapper
    # with dummy arugments to infer the output dtype.
    # This incurs memory allocations within numba, as well as
    # exceptions, leading to memory leaks as described
    # in https://github.com/numba/numba/issues/3263
    meta = np.empty((0,)*len(out_dims), dtype=np.complex128)

    # Construct the wrapper call from given arguments
    out = da.blockwise(rime_dask_wrapper, out_dims,
                       factory, None,
                       names, None,
                       len(contract_dims), None,
                       *args,
                       concatenate=False,
                       adjust_chunks=adjust_chunks,
                       meta=meta)

    # Contract over source and concatenation dims
    axes = (0,) + tuple(range(len(dims), len(dims) + len(contract_dims)))
    return out.sum(axis=axes)
