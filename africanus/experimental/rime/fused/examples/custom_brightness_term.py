import argparse
import numpy as np
from numba import literal_unroll
from numba import types
from numba.core.errors import TypingError
from numba.cpython.unsafe.tuple import tuple_setitem

# this enables tuple constriction in the sampler
from africanus.experimental.rime.fused.terms.cube_dde import zero_vis_factory
from africanus.experimental.rime.fused.specification import RimeSpecification
from africanus.experimental.rime.fused.terms.core import Term
from africanus.experimental.rime.fused.core import rime
from daskms import xds_from_storage_ms as xds_from_ms
from daskms import xds_from_storage_table as xds_from_table


class ModelFlux(Term):
    """
    A custom model flux provider to substitute the standard Brightness term.
    Here out model flux is provided as an array of shape (nsource, nchan, nstokes)
    """

    def __init__(self, configuration, stokes):
        super().__init__(configuration)
        self.stokes = stokes

    def dask_schema(
        self,
        model_flux,
    ):
        return {
            "model_flux": ("source", "chan", "stokes"),
        }

    def init_fields(self, typingctx, init_state, model_flux):  # (ndir, nchan, nstokes)
        if not isinstance(model_flux, types.Array) or model_flux.ndim != 3:
            raise TypingError(
                f"model_flux {model_flux} should be a (ndir, nchan, nstokes) array"
            )

        return [], lambda init_state, model_flux: None

    def sampler(self):
        NSTOKES = len(self.stokes)
        zero_vis = zero_vis_factory(NSTOKES)

        def model_sample(state, s, r, t, f1, f2, a1, a2, c):
            flux = zero_vis(state.model_flux.dtype.type(0))
            for co in literal_unroll(range(NSTOKES)):
                flux = tuple_setitem(flux, co, state.model_flux[s, c, co])
            return flux

        return model_sample


def main():
    parser = argparse.ArgumentParser(
        description="Apply MdV beams to predict vis for source."
    )

    parser.add_argument(
        "ms",
        # type=DaskMSStore,
        help="Path to input measurement set, e.g. path/to/dir/foo.MS. Also "
        "accepts valid s3 urls.",
    )

    opts = parser.parse_args()
    xds = xds_from_ms(opts.ms)[0]
    freq = xds_from_table(f"{opts.ms}::SPECTRAL_WINDOW")[0].CHAN_FREQ.values.squeeze()
    phase_dir = xds_from_table(f"{opts.ms}::FIELD")[0].PHASE_DIR.values.squeeze()
    ant_pos = xds_from_table(f"{opts.ms}::ANTENNA")[0].POSITION.values

    if freq.ndim == 2:
        freq = freq[0]
    if phase_dir.ndim == 2:
        phase_dir = phase_dir[0]
    print(phase_dir)
    print(freq.shape)
    print(phase_dir.shape)
    # 1 fake source location
    nsource = 1
    radec = np.zeros((nsource, 2))
    # fake random flux
    nchan = freq.size
    # 4 Stokes components
    model_flux = np.ones((nsource, nchan, 4))

    # dataset dict
    ds = {
        "time": xds.TIME.values,
        "antenna1": xds.ANTENNA1.values,
        "antenna2": xds.ANTENNA2.values,
        "feed1": xds.FEED1.values,
        "feed2": xds.FEED2.values,
        "radec": radec,
        "phase_dir": phase_dir,
        "uvw": xds.UVW.values,
        "chan_freq": freq,
        "model_flux": model_flux,
        # "antenna_position": ant_pos,
    }

    # rime_str = "(Kpq,Bpq): [I,Q,U,V] -> [XX,XY,YX,YY]"
    rime_str = "(Kpq,Bpq): [I,Q,U,V] -> [XX,XY,YX,YY]"
    # We want to use the rime with our new custom "Brightness" term
    spec = RimeSpecification(rime_str, terms={"B": ModelFlux})

    model_vis = rime(spec, ds)
    print(model_vis)


if __name__ == "__main__":
    main()
