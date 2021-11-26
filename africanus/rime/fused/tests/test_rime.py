import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from africanus.coordinates import radec_to_lm

from africanus.rime.phase import phase_delay
from africanus.model.spectral import spectral_model
from africanus.model.coherency import convert

from africanus.rime.fused.parser import parse_rime
from africanus.rime.fused.rime import rime_factory
from africanus.rime.fused.dask import rime as dask_rime


@pytest.mark.parametrize("rime_spec", [
    # "G_{p}[E_{stpf}L_{tpf}K_{stpqf}B_{spq}L_{tqf}E_{q}]G_{q}",
    # "Gp[EpLpKpqBpqLqEq]sGq",
    "[Gp, (Ep, Lp, Kpq, Bpq, Lq, Eq), Gq]: [I, Q, U, V] -> [XX, XY, YX, YY]",
    # "[Gp x (Ep x Lp x Kpq x Bpq x Lq x Eq) x Gq] -> [XX, XY, YX, YY]",
])
def test_rime_parser(rime_spec):
    # custom_mapping = {"Kpq": MyCustomPhaseTerm}
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
def test_fused_rime(chunks):
    nsrc = sum(chunks["source"])
    nrow = sum(chunks["row"])
    nspi = sum(chunks["spi"])
    nchan = sum(chunks["chan"])
    ncorr = sum(chunks["corr"])

    time = np.linspace(0.1, 1.0, nrow)
    antenna1 = np.zeros(nrow, dtype=np.int32)
    antenna2 = np.arange(nrow, dtype=np.int32)
    feed1 = feed2 = antenna1
    radec = np.random.random(size=(nsrc, 2))*1e-5
    phase_centre = np.random.random(2)*1e-5
    uvw = np.random.random(size=(nrow, 3))
    chan_freq = np.linspace(.856e9, 2*.859e9, nchan)
    stokes = np.random.random(size=(nsrc, ncorr))
    spi = np.random.random(size=(nsrc, nspi, ncorr))
    ref_freq = np.random.random(size=nsrc)*.856e9
    lm = radec_to_lm(radec, phase_centre)

    rime = rime_factory()
    out = rime(time, antenna1, antenna2, feed1, feed2,
               radec=radec, phase_centre=phase_centre,
               uvw=uvw, chan_freq=chan_freq, stokes=stokes,
               spi=spi, ref_freq=ref_freq,
               convention="casa", spi_base="standard")
    P = phase_delay(lm, uvw, chan_freq, convention="casa")
    SM = spectral_model(stokes, spi, ref_freq, chan_freq, base="std")
    B = convert(SM, ["I", "Q", "U", "V"], ["XX", "XY", "YX", "YY"])
    expected = (P[:, :, :, None]*B[:, None, :, :]).sum(axis=0)
    assert_array_almost_equal(expected, out)

    out = rime(time, antenna1, antenna2, feed1, feed2,
               lm=lm, uvw=uvw, chan_freq=chan_freq, stokes=stokes,
               spi=spi, ref_freq=ref_freq, convention="fourier")

    P = phase_delay(lm, uvw, chan_freq, convention="fourier")
    SM = spectral_model(stokes, spi, ref_freq, chan_freq, base="std")
    B = convert(SM, ["I", "Q", "U", "V"], ["XX", "XY", "YX", "YY"])
    expected = (P[:, :, :, None]*B[:, None, :, :]).sum(axis=0)
    assert_array_almost_equal(expected, out)

    out = rime(time, antenna1, antenna2, feed1, feed2,
               lm=lm, uvw=uvw, chan_freq=chan_freq, stokes=stokes,
               spi=spi, ref_freq=ref_freq)
    P = phase_delay(lm, uvw, chan_freq, convention="fourier")
    SM = spectral_model(stokes, spi, ref_freq, chan_freq, base="std")
    B = convert(SM, ["I", "Q", "U", "V"], ["XX", "XY", "YX", "YY"])
    expected = (P[:, :, :, None]*B[:, None, :, :]).sum(axis=0)
    assert_array_almost_equal(expected, out)


@pytest.mark.parametrize("chunks", chunks)
def test_fused_dask_rime(chunks):
    da = pytest.importorskip("dask.array")

    nsrc = sum(chunks["source"])
    nrow = sum(chunks["row"])
    nspi = sum(chunks["spi"])
    nchan = sum(chunks["chan"])
    ncorr = sum(chunks["corr"])

    time = np.linspace(0.1, 1.0, nrow)
    antenna1 = np.zeros(nrow, dtype=np.int32)
    antenna2 = np.arange(nrow, dtype=np.int32)
    feed1 = feed2 = antenna1
    radec = np.random.random(size=(nsrc, 2))*1e-5
    phase_centre = np.random.random(size=(2,))*1e-5
    uvw = np.random.random(size=(nrow, 3))*1e5
    chan_freq = np.linspace(.856e9, 2*.859e9, nchan)
    stokes = np.random.random(size=(nsrc, ncorr))
    spi = np.random.random(size=(nsrc, nspi, ncorr))
    ref_freq = np.random.random(size=nsrc)*.856e9

    def darray(array, dims):
        return da.from_array(array, tuple(chunks[d] for d in dims))

    dask_time = darray(time, ("row",))
    dask_antenna1 = darray(antenna1, ("row",))
    dask_antenna2 = darray(antenna2, ("row",))
    dask_feed1 = darray(feed1, ("row",))
    dask_feed2 = darray(feed2, ("row",))
    dask_radec = darray(radec, ("source", "radec"))
    dask_phase_centre = darray(phase_centre, ("radec",))
    dask_uvw = darray(uvw, ("row", "uvw"))
    dask_chan_freq = darray(chan_freq, ("chan",))
    dask_stokes = darray(stokes, ("source", "corr"))
    dask_spi = darray(spi, ("source", "spi", "corr"))
    dask_ref_freq = darray(ref_freq, ("source",))

    dask_out = dask_rime(dask_time,
                         dask_antenna1, dask_antenna2,
                         dask_feed1, dask_feed2,
                         radec=dask_radec, phase_centre=dask_phase_centre,
                         uvw=dask_uvw, stokes=dask_stokes,
                         spi=dask_spi, chan_freq=dask_chan_freq,
                         ref_freq=dask_ref_freq, convention="casa")

    rime = rime_factory()
    out = rime(time, antenna1, antenna2, feed1, feed2,
               radec=radec, phase_centre=phase_centre,
               uvw=uvw, chan_freq=chan_freq, stokes=stokes,
               spi=spi, ref_freq=ref_freq, convention="casa")

    assert_array_almost_equal(dask_out.compute(
        scheduler="single-threaded"), out)


@pytest.mark.parametrize("chunks", chunks)
def test_rime_wrapper(chunks):
    nsrc = sum(chunks["source"])
    nrow = sum(chunks["row"])
    nspi = sum(chunks["spi"])
    nchan = sum(chunks["chan"])
    ncorr = sum(chunks["corr"])

    time = np.linspace(0.1, 1.0, nrow)
    antenna1 = np.zeros(nrow, dtype=np.int32)
    antenna2 = np.arange(nrow, dtype=np.int32)
    feed1 = feed2 = antenna1
    radec = np.random.random(size=(nsrc, 2))*1e-5
    phase_centre = np.random.random(size=(2,))*1e-5
    uvw = np.random.random(size=(nrow, 3))*1e5
    chan_freq = np.linspace(.856e9, 2*.859e9, nchan)
    stokes = np.random.random(size=(nsrc, ncorr))
    spi = np.random.random(size=(nsrc, nspi, ncorr))
    ref_freq = np.random.random(size=nsrc)*.856e9

    kw = {
        "time": time,
        "antenna1": antenna1,
        "antenna2": antenna2,
        "feed1": feed1,
        "feed2": feed2,
        "radec": radec,
        "phase_centre": phase_centre,
        "uvw": uvw,
        "chan_freq": chan_freq,
        "stokes": stokes,
        "spi": spi,
        "ref_freq": ref_freq
    }

    def _maybe_convert_xarray_dataset(mapping):
        try:
            import xarray as xr
        except ImportError:
            return mapping

        if isinstance(mapping, xr.Dataset):
            return {k: v.data for k, v in mapping.items()}
        else:
            return mapping

    def rime(*other, **kwargs):
        terms = kwargs.pop("terms", None)
        transformers = kwargs.pop("transformers", None)

        factory = rime_factory(terms=terms, transformers=transformers)  # noqa
        from collections.abc import Mapping

        if len(other) == 0:
            pass
        elif len(other) == 1:
            try:
                k, v = other
            except (ValueError, TypeError):
                mapping = other[0]

                if not isinstance(mapping, Mapping):
                    raise TypeError(f"Singleton *other does not "
                                    f"contain a mapping, but "
                                    f"{mapping}")

                mapping = _maybe_convert_xarray_dataset(mapping)
            else:
                kwargs[k] = v
        else:
            try:
                for k, v in other:
                    pass
            except (ValueError, TypeError):
                pass

    rime(**kw)
