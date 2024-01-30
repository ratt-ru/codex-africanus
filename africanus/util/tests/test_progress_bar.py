import pytest

from africanus.util.dask_util import EstimatingProgressBar, format_time


def test_progress_bar():
    da = pytest.importorskip("dask.array")

    A = da.zeros((10, 10, 2))
    B = A.map_blocks(lambda a: a + 1.0)

    with EstimatingProgressBar(out=None):
        B.compute()

    assert "    0s" == format_time(0)
    assert "   59s" == format_time(59)
    assert " 1m 0s" == format_time(60)
    assert " 1m 1s" == format_time(61)

    assert " 2h 6m" == format_time(2 * 60 * 60 + 6 * 60)
    assert " 2h 6m" == format_time(2 * 60 * 60 + 6 * 60 + 59)
    assert " 2h 7m" == format_time(2 * 60 * 60 + 7 * 60)
    assert " 2h 7m" == format_time(2 * 60 * 60 + 7 * 60 + 1)

    assert " 5d 2h" == format_time(5 * 60 * 60 * 24 + 2 * 60 * 60 + 500)

    assert " 5w 2d" == format_time(5 * 60 * 60 * 24 * 7 + 2 * 60 * 60 * 24 + 500)
