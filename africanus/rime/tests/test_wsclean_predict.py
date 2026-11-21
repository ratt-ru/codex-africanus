# -*- coding: utf-8 -*-

import numba
from numba import types
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest

from africanus.rime import phase_delay
from africanus.model.shape.gaussian_shape import gaussian
from africanus.model.wsclean.spec_model import spectra
from africanus.rime.wsclean_predict import wsclean_predict

chunk_parametrization = pytest.mark.parametrize(
    "chunks",
    [
        {
            "source": (2, 3, 4, 2, 2, 2, 2, 2, 2),
            "time": (2, 1, 1),
            "rows": (4, 4, 2),
            "antenna": (4,),
            "channels": (3, 2),
        }
    ],
)

requires_two_threads = pytest.mark.skipif(
    numba.config.NUMBA_NUM_THREADS < 2, reason="Requires two numba threads"
)

nthreads_parametrization = pytest.mark.parametrize(
    "nthreads", [1, pytest.param(2, marks=requires_two_threads)]
)


@pytest.fixture
def restore_num_threads():
    """Restores the numba thread count of the test thread"""
    nthreads = numba.get_num_threads()

    try:
        yield
    finally:
        numba.set_num_threads(nthreads)


def predict_args(row=10, src=21, chan=5):
    rs = np.random.RandomState(42)
    source_sel = rs.randint(0, 2, src).astype(np.bool_)
    source_type = np.where(source_sel, "POINT", "GAUSSIAN")

    gauss_shape = rs.normal(size=(src, 3))
    uvw = rs.normal(size=(row, 3))
    lm = rs.normal(size=(src, 2)) * 1e-5
    flux = rs.normal(size=src)
    coeffs = rs.normal(size=(src, 2))
    log_poly = rs.randint(0, 2, src, dtype=np.bool_)
    flux[log_poly] = np.abs(flux[log_poly])
    coeffs[log_poly] = np.abs(coeffs[log_poly])
    freq = np.linspace(0.856e9, 2 * 0.856e9, chan)
    ref_freq = np.full(src, freq[freq.shape[0] // 2])

    return (
        uvw,
        lm,
        source_type,
        flux,
        coeffs,
        log_poly,
        ref_freq,
        gauss_shape,
        freq,
    )


@nthreads_parametrization
@chunk_parametrization
def test_wsclean_predict(chunks, nthreads):
    row = sum(chunks["rows"])
    src = sum(chunks["source"])
    chan = sum(chunks["channels"])

    rs = np.random.RandomState(42)
    source_sel = rs.randint(0, 2, src).astype(np.bool_)
    source_type = np.where(source_sel, "POINT", "GAUSSIAN")

    gauss_shape = rs.normal(size=(src, 3))
    uvw = rs.normal(size=(row, 3))
    lm = rs.normal(size=(src, 2)) * 1e-5
    flux = rs.normal(size=src)
    coeffs = rs.normal(size=(src, 2))
    log_poly = rs.randint(0, 2, src, dtype=np.bool_)
    flux[log_poly] = np.abs(flux[log_poly])
    coeffs[log_poly] = np.abs(coeffs[log_poly])
    freq = np.linspace(0.856e9, 2 * 0.856e9, chan)
    ref_freq = np.full(src, freq[freq.shape[0] // 2])

    # WSClean visibilities
    vis = wsclean_predict(
        uvw,
        lm,
        source_type,
        flux,
        coeffs,
        log_poly,
        ref_freq,
        gauss_shape,
        freq,
        nthreads,
    )

    # Compute it another way. Note the CASA coordinate convention
    # used by wsclean
    phase = phase_delay(lm, uvw, freq, convention="casa")
    spectrum = spectra(flux, coeffs, log_poly, ref_freq, freq)
    shape = gaussian(uvw, freq, gauss_shape)
    # point sources don't' contribute to the shape
    shape[source_sel] = 1.0
    np_vis = np.einsum("srf,srf,sf->rf", shape, phase, spectrum)[:, :, None]

    assert_almost_equal(np_vis, vis)

    # The serial and parallel kernels sum sources in the same order
    serial_vis = wsclean_predict(
        uvw, lm, source_type, flux, coeffs, log_poly, ref_freq, gauss_shape, freq
    )
    assert_array_equal(serial_vis, vis)


def test_wsclean_predict_default_is_cacheable():
    """
    numba cannot cache functions that set the thread count.
    Check that this does not apply when nthreads is defaulted.
    """
    wsclean_predict(*predict_args())

    default_overloads = [
        cres
        for sig, cres in wsclean_predict.overloads.items()
        if isinstance(sig[-1], types.Omitted)
    ]

    assert len(default_overloads) > 0
    assert not any(cres.library.has_dynamic_globals for cres in default_overloads)


@requires_two_threads
def test_wsclean_predict_restores_num_threads(restore_num_threads):
    args = predict_args()

    # Differs from the number of threads requested from the kernel
    numba.set_num_threads(1)
    vis = wsclean_predict(*args, nthreads=2)
    assert numba.get_num_threads() == 1

    assert_array_equal(wsclean_predict(*args), vis)


@pytest.mark.parametrize("nthreads", [0, -1, numba.config.NUMBA_NUM_THREADS + 1])
def test_wsclean_predict_invalid_nthreads(nthreads, restore_num_threads):
    expected_nthreads = numba.get_num_threads()

    with pytest.raises(ValueError, match="number of threads"):
        wsclean_predict(*predict_args(), nthreads=nthreads)

    assert numba.get_num_threads() == expected_nthreads


@nthreads_parametrization
def test_wsclean_predict_invalid_source_type(nthreads, restore_num_threads):
    uvw, lm, source_type, *other_args = predict_args()
    source_type[source_type.shape[0] // 2] = "DISK"
    expected_nthreads = numba.get_num_threads()

    with pytest.raises(ValueError, match="source_type"):
        wsclean_predict(uvw, lm, source_type, *other_args, nthreads=nthreads)

    assert numba.get_num_threads() == expected_nthreads


@nthreads_parametrization
@chunk_parametrization
def test_dask_wsclean_predict(chunks, nthreads, restore_num_threads):
    da = pytest.importorskip("dask.array")

    from africanus.rime.dask_predict import wsclean_predict as dask_wsclean_predict

    row = sum(chunks["rows"])
    src = sum(chunks["source"])
    chan = sum(chunks["channels"])

    rs = np.random.RandomState(42)
    source_sel = rs.randint(0, 2, src).astype(np.bool_)
    source_type = np.where(source_sel, "POINT", "GAUSSIAN")

    gauss_shape = rs.normal(size=(src, 3))

    uvw = rs.normal(size=(row, 3))
    lm = rs.normal(size=(src, 2)) * 1e-5
    flux = rs.normal(size=src)
    coeffs = rs.normal(size=(src, 2))
    log_poly = rs.randint(0, 2, src, dtype=np.bool_)
    flux[log_poly] = np.abs(flux[log_poly])
    coeffs[log_poly] = np.abs(coeffs[log_poly])
    freq = np.linspace(0.856e9, 2 * 0.856e9, chan)
    ref_freq = np.full(src, freq[freq.shape[0] // 2])

    da_uvw = da.from_array(uvw, chunks=(chunks["rows"], 3))
    da_lm = da.from_array(lm, chunks=(chunks["source"], 2))
    da_source_type = da.from_array(source_type, chunks=chunks["source"])
    da_gauss_shape = da.from_array(gauss_shape, chunks=(chunks["source"], 3))
    da_flux = da.from_array(flux, chunks=chunks["source"])
    da_coeffs = da.from_array(coeffs, chunks=(chunks["source"], 2))
    da_log_poly = da.from_array(log_poly, chunks=chunks["source"])
    da_ref_freq = da.from_array(ref_freq, chunks=chunks["source"])
    da_freq = da.from_array(freq)

    vis = wsclean_predict(
        uvw, lm, source_type, flux, coeffs, log_poly, ref_freq, gauss_shape, freq
    )
    da_vis = dask_wsclean_predict(
        da_uvw,
        da_lm,
        da_source_type,
        da_flux,
        da_coeffs,
        da_log_poly,
        da_ref_freq,
        da_gauss_shape,
        da_freq,
        nthreads=nthreads,
    )

    # The default threaded scheduler calls the kernel from multiple threads.
    # This is always safe for the serial kernel, but the parallel kernel
    # would need a thread-safe numba threading layer.
    scheduler = "threads" if nthreads == 1 else "sync"
    expected_nthreads = numba.get_num_threads()
    assert_almost_equal(vis, da_vis.compute(scheduler=scheduler))
    assert numba.get_num_threads() == expected_nthreads
