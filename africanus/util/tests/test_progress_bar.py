import pytest

from africanus.util.dask import EstimatingProgressBar


def test_progress_bar():
    da = pytest.importorskip("dask.array")

    A = da.zeros((10, 10, 2))
    B = A.map_blocks(lambda a: a + 1.0)

    with EstimatingProgressBar(out=None):
        B.compute()
