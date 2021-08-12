try:
    import dask.array as da
except ImportError as e:
    opt_import_err = e
else:
    opt_import_err = None

import numpy as np

from africanus.util.requirements import requires_optional
from africanus.rime.monolothic.rime import rime_factory


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
def rime(terms=None, **kwargs):
    factory = rime_factory(terms=terms)
    names, args = factory.dask_blockwise_args(**kwargs)

    dims = ("source", "row", "chan", "corr")
    contract_dims = set(d for ds in args[1::2] if ds is not None for d in ds)
    contract_dims -= set(dims)
    out_dims = dims + tuple(contract_dims)

    # Source and concatentation dimension are reduced to 1 element
    adjust_chunks = {"source": 1}
    adjust_chunks.update((d, 1) for d in contract_dims)

    # Construct the wrapper call from given arguments
    out = da.blockwise(rime_dask_wrapper, out_dims,
                       factory, None,
                       names, None,
                       len(contract_dims), None,
                       *args,
                       concatenate=False,
                       adjust_chunks=adjust_chunks,
                       dtype=np.complex64)

    # Contract over source and concatenation dims
    axes = (0,) + tuple(range(len(dims), len(dims) + len(contract_dims)))
    return out.sum(axis=axes)
