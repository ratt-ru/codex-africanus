import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from africanus.coordinates import radec_to_lm

from africanus.rime.phase import phase_delay
from africanus.model.spectral import spectral_model
from africanus.model.shape.gaussian_shape import gaussian
from africanus.model.coherency import convert

from africanus.experimental.rime.fused.specification import (
    RimeSpecification, parse_rime)
from africanus.experimental.rime.fused.core import rime
from africanus.experimental.rime.fused.dask import rime as dask_rime


@pytest.mark.skip
@pytest.mark.parametrize("rime_spec", [
    # "G_{p}[E_{stpf}L_{tpf}K_{stpqf}B_{spq}L_{tqf}E_{q}]G_{q}",
    # "Gp[EpLpKpqBpqLqEq]sGq",
    "[Gp, (Ep, Lp, Kpq, Bpq, Lq, Eq), Gq]: [I, Q, U, V] -> [XX, XY, YX, YY]",
    # "[Gp x (Ep x Lp x Kpq x Bpq x Lq x Eq) x Gq] -> [XX, XY, YX, YY]",
])
def test_rime_specification(rime_spec):
    # custom_mapping = {"Kpq": MyCustomPhase}
    print(parse_rime(rime_spec))
    pass


chunks = [
    {
        "source": (129, 67, 139),
        "row": (2, 2, 2, 2),
        "spi": (2,),
        "chan": (2, 2),
        "corr": (4,),
        "radec": (2,),
        "uvw": (3,),
    },
    {
        "source": (5,),
        "row": (2, 2, 2, 2),
        "spi": (2,),
        "chan": (2, 2),
        "corr": (4,),
        "radec": (2,),
        "uvw": (3,),
    },
]


@pytest.mark.parametrize("chunks", chunks)
@pytest.mark.parametrize("stokes_schema", [["I", "Q", "U", "V"]], ids=str)
@pytest.mark.parametrize("corr_schema", [["XX", "XY", "YX", "YY"]], ids=str)
def test_fused_rime(chunks, stokes_schema, corr_schema):
    nsrc = sum(chunks["source"])
    nrow = sum(chunks["row"])
    nspi = sum(chunks["spi"])
    nchan = sum(chunks["chan"])
    ncorr = sum(chunks["corr"])

    stokes_to_corr = "".join(("[", ",".join(stokes_schema),
                              "] -> [",
                              ",".join(corr_schema), "]"))

    time = np.linspace(0.1, 1.0, nrow)
    antenna1 = np.zeros(nrow, dtype=np.int32)
    antenna2 = np.arange(nrow, dtype=np.int32)
    feed1 = feed2 = antenna1
    radec = np.random.random(size=(nsrc, 2))*1e-5
    phase_dir = np.random.random(2)*1e-5
    uvw = np.random.random(size=(nrow, 3))
    chan_freq = np.linspace(.856e9, 2*.859e9, nchan)
    stokes = np.random.random(size=(nsrc, ncorr))
    spi = np.random.random(size=(nsrc, nspi, ncorr))
    ref_freq = np.random.random(size=nsrc)*.856e9
    lm = radec_to_lm(radec, phase_dir)

    dataset = {
        "time": time,
        "antenna1": antenna1,
        "antenna2": antenna2,
        "feed1": feed1,
        "feed2": feed2,
        "radec": radec,
        "phase_dir": phase_dir,
        "uvw": uvw,
        "chan_freq": chan_freq,
        "stokes": stokes,
        "spi": spi,
        "ref_freq": ref_freq,
    }

    out = rime(f"(Kpq, Bpq): {stokes_to_corr}",
               dataset, convention="casa", spi_base="standard")
    P = phase_delay(lm, uvw, chan_freq, convention="casa")
    SM = spectral_model(stokes, spi, ref_freq, chan_freq, base="std")
    B = convert(SM, stokes_schema, corr_schema)
    expected = (P[:, :, :, None]*B[:, None, :, :]).sum(axis=0)
    assert_array_almost_equal(expected, out)

    out = rime(f"(Kpq, Bpq): {stokes_to_corr}",
               dataset, convention="fourier")

    P = phase_delay(lm, uvw, chan_freq, convention="fourier")
    SM = spectral_model(stokes, spi, ref_freq, chan_freq, base="std")
    B = convert(SM, stokes_schema, corr_schema)
    expected = (P[:, :, :, None]*B[:, None, :, :]).sum(axis=0)
    assert_array_almost_equal(expected, out)

    out = rime(f"(Kpq, Bpq): {stokes_to_corr}", dataset)
    P = phase_delay(lm, uvw, chan_freq, convention="fourier")
    SM = spectral_model(stokes, spi, ref_freq, chan_freq, base="std")
    B = convert(SM, stokes_schema, corr_schema)
    expected = (P[:, :, :, None]*B[:, None, :, :]).sum(axis=0)
    assert_array_almost_equal(expected, out)

    gauss_shape = np.random.random((nsrc, 3))
    rime_spec = RimeSpecification(f"(Cpq, Kpq, Bpq): {stokes_to_corr}",
                                  terms={"Cpq": "Gaussian"})
    out = rime(rime_spec, {**dataset, "gauss_shape": gauss_shape})

    P = phase_delay(lm, uvw, chan_freq, convention="fourier")
    SM = spectral_model(stokes, spi, ref_freq, chan_freq, base="std")
    B = convert(SM, stokes_schema, corr_schema)
    G = gaussian(uvw, chan_freq, gauss_shape)
    expected = (G[:, :, :, None]*P[:, :, :, None]*B[:, None, :, :]).sum(axis=0)
    assert_array_almost_equal(expected, out)


@pytest.mark.parametrize("chunks", chunks)
@pytest.mark.parametrize("stokes_schema", [["I", "Q", "U", "V"]], ids=str)
@pytest.mark.parametrize("corr_schema", [["XX", "XY", "YX", "YY"]], ids=str)
def test_fused_dask_rime(chunks, stokes_schema, corr_schema):
    da = pytest.importorskip("dask.array")

    nsrc = sum(chunks["source"])
    nrow = sum(chunks["row"])
    nspi = sum(chunks["spi"])
    nchan = sum(chunks["chan"])
    ncorr = sum(chunks["corr"])

    stokes_to_corr = "".join(("[", ",".join(stokes_schema),
                              "] -> [",
                              ",".join(corr_schema), "]"))

    time = np.linspace(0.1, 1.0, nrow)
    antenna1 = np.zeros(nrow, dtype=np.int32)
    antenna2 = np.arange(nrow, dtype=np.int32)
    feed1 = feed2 = antenna1
    radec = np.random.random(size=(nsrc, 2))*1e-5
    phase_dir = np.random.random(size=(2,))*1e-5
    uvw = np.random.random(size=(nrow, 3))*1e5
    chan_freq = np.linspace(.856e9, 2*.859e9, nchan)
    stokes = np.random.random(size=(nsrc, ncorr))
    spi = np.random.random(size=(nsrc, nspi, ncorr))
    ref_freq = np.random.random(size=nsrc)*.856e9

    def darray(array, dims):
        return da.from_array(array, tuple(chunks[d] for d in dims))

    dask_dataset = {
        "time": darray(time, ("row",)),
        "antenna1": darray(antenna1, ("row",)),
        "antenna2":  darray(antenna2, ("row",)),
        "feed1": darray(feed1, ("row",)),
        "feed2": darray(feed2, ("row",)),
        "radec": darray(radec, ("source", "radec")),
        "phase_dir": darray(phase_dir, ("radec",)),
        "uvw": darray(uvw, ("row", "uvw")),
        "chan_freq": darray(chan_freq, ("chan",)),
        "stokes": darray(stokes, ("source", "corr")),
        "spi": darray(spi, ("source", "spi", "corr")),
        "ref_freq": darray(ref_freq, ("source",)),
    }

    rime_spec = RimeSpecification(f"(Kpq, Bpq): {stokes_to_corr}")
    dask_out = dask_rime(rime_spec, dask_dataset, convention="casa")

    dataset = {
        "time": time,
        "antenna1": antenna1,
        "antenna2":  antenna2,
        "feed1": feed1,
        "feed2": feed2,
        "radec": radec,
        "phase_dir": phase_dir,
        "uvw": uvw,
        "chan_freq": chan_freq,
        "stokes": stokes,
        "spi": spi,
        "ref_freq": ref_freq,
    }

    out = rime(rime_spec, dataset, convention="casa")

    assert_array_almost_equal(dask_out.compute(
        scheduler="single-threaded"), out)
