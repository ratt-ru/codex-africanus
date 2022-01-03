import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from africanus.coordinates import radec_to_lm

from africanus.rime.phase import phase_delay
from africanus.model.spectral import spectral_model
from africanus.model.coherency import convert
from africanus.model.shape import gaussian

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
        "lw": (10,),
        "mh": (10,),
        "nud": (10,),
    },
    {
        "source": (5,),
        "row": (2, 2, 2, 2),
        "spi": (2,),
        "chan": (2, 2),
        "corr": (4,),
        "radec": (2,),
        "uvw": (3,),
        "lw": (10,),
        "mh": (10,),
        "nud": (10,),
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
    assert np.count_nonzero(out) > .8 * out.size

    out = rime(f"(Kpq, Bpq): {stokes_to_corr}",
               dataset, convention="fourier")

    P = phase_delay(lm, uvw, chan_freq, convention="fourier")
    SM = spectral_model(stokes, spi, ref_freq, chan_freq, base="std")
    B = convert(SM, stokes_schema, corr_schema)
    expected = (P[:, :, :, None]*B[:, None, :, :]).sum(axis=0)
    assert_array_almost_equal(expected, out)
    assert np.count_nonzero(out) > .8 * out.size

    out = rime(f"(Kpq, Bpq): {stokes_to_corr}", dataset)
    P = phase_delay(lm, uvw, chan_freq, convention="fourier")
    SM = spectral_model(stokes, spi, ref_freq, chan_freq, base="std")
    B = convert(SM, stokes_schema, corr_schema)
    expected = (P[:, :, :, None]*B[:, None, :, :]).sum(axis=0)
    assert_array_almost_equal(expected, out)
    assert np.count_nonzero(out) > .8 * out.size

    gauss_shape = np.random.random((nsrc, 3))
    gauss_shape[:, :2] *= 1e-5

    rime_spec = RimeSpecification(f"(Cpq, Kpq, Bpq): {stokes_to_corr}",
                                  terms={"C": "Gaussian"})
    out = rime(rime_spec,
               {**dataset, "gauss_shape": gauss_shape})
    P = phase_delay(lm, uvw, chan_freq, convention="fourier")
    SM = spectral_model(stokes, spi, ref_freq, chan_freq, base="std")
    B = convert(SM, stokes_schema, corr_schema)
    G = gaussian(uvw, chan_freq, gauss_shape)
    expected = (G[:, :, :, None]*P[:, :, :, None]*B[:, None, :, :]).sum(axis=0)
    assert_array_almost_equal(expected, out)
    assert np.count_nonzero(out) > .8 * out.size


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
    lw = sum(chunks["lw"])
    mh = sum(chunks["mh"])
    nud = sum(chunks["nud"])

    stokes_to_corr = "".join(("[", ",".join(stokes_schema),
                              "] -> [",
                              ",".join(corr_schema), "]"))

    from africanus.rime.tests.test_parangles import _observation_endpoints

    start, end = _observation_endpoints(2021, 10, 9, 8)
    time = np.linspace(start, end, nrow)
    antenna1 = np.zeros(nrow, dtype=np.int32)
    antenna2 = np.arange(nrow, dtype=np.int32)
    feed1 = feed2 = antenna1
    radec = np.random.random(size=(nsrc, 2))*1e-5
    phase_dir = np.random.random(size=(2,))*1e-5
    uvw = np.random.random(size=(nrow, 3))*1e5
    chan_freq = np.linspace(.856e9, 2*.859e9, nchan)
    stokes = np.random.random(size=(nsrc, ncorr))
    spi = np.random.random(size=(nsrc, nspi, ncorr))
    ref_freq = np.random.random(size=nsrc)*.856e9 + .856e9
    beam = np.random.random(size=(lw, mh, nud, ncorr))
    beam_lm_extents = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    beam_freq_map = np.random.uniform(
        low=chan_freq[0], high=chan_freq[-1], size=nud)
    beam_freq_map.sort()
    gauss_shape = np.random.random((nsrc, 3))
    gauss_shape[:, :2] *= 1e-5

    uant = np.unique([antenna1, antenna2])
    antenna_position = np.random.random(size=(uant.size, 3)) * 1000
    ufeed = np.unique([feed1, feed2])
    receptor_angle = np.random.random((ufeed.shape[0], 2))

    def darray(array, dims):
        return da.from_array(array, tuple(chunks.get(d, d) for d in dims))

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
        "antenna_position": darray(antenna_position, antenna_position.shape),
        "beam": darray(beam, ("lw", "mh", "nud", "corr")),
        "beam_lm_extents": darray(beam_lm_extents, (2, 2)),
        "beam_freq_map": darray(beam_freq_map, ("nud",)),
        "gauss_shape": darray(gauss_shape, ("source", 3)),
        "receptor_angle": darray(receptor_angle, receptor_angle.shape),
    }

    rime_spec = RimeSpecification(f"(Kpq, Bpq, Lq): {stokes_to_corr}")
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
        "antenna_position": antenna_position,
        "beam": beam,
        "beam_lm_extents": beam_lm_extents,
        "beam_freq_map": beam_freq_map,
        "gauss_shape": gauss_shape,
        "receptor_angle": receptor_angle,
    }

    out = rime(rime_spec, dataset, convention="casa")
    dout = dask_out.compute()
    assert_array_almost_equal(dout, out)
    assert np.count_nonzero(out) > .8 * out.size
