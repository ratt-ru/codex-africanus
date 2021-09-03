import numpy as np

from numba.core import cgutils, types, errors
from numba.extending import intrinsic


from africanus.util.casa_types import STOKES_TYPES
from africanus.rime.monolithic.terms import Term


STOKES_CONVERSION = {
    'RR': {('I', 'V'): lambda i, v: i + v},
    'RL': {('Q', 'U'): lambda q, u: q + u*1j},
    'LR': {('Q', 'U'): lambda q, u: q - u*1j},
    'LL': {('I', 'V'): lambda i, v: i - v},

    'XX': {('I', 'Q'): lambda i, q: i + q},
    'XY': {('U', 'V'): lambda u, v: u + v*1j},
    'YX': {('U', 'V'): lambda u, v: u - v*1j},
    'YY': {('I', 'Q'): lambda i, q: i - q},
}


def conversion_factory(stokes_schema, corr_schema):
    @intrinsic
    def corr_convert(typingctx, spectral_model, source_index, chan_index):
        if (not isinstance(spectral_model, types.Array) or
                spectral_model.ndim != 3):
            raise errors.TypingError(f"'spectral_model' should be 3D array. "
                                     f"Got {spectral_model}")

        if not isinstance(source_index, types.Integer):
            raise errors.TypingError(f"'source_index' should be an integer. "
                                     f"Got {source_index}")

        if not isinstance(chan_index, types.Integer):
            raise errors.TypingError(f"'chan_index' should be an integer. "
                                     f"Got {chan_index}")

        spectral_model_map = {s: i for i, s in enumerate(stokes_schema)}
        conv_map = {}

        for corr in corr_schema:
            try:
                conv_schema = STOKES_CONVERSION[corr]
            except KeyError:
                raise ValueError(f"No conversion schema "
                                 f"registered for correlation {corr}")

            i1 = -1
            i2 = -1

            for (s1, s2), fn in conv_schema.items():
                try:
                    i1 = spectral_model_map[s1]
                    i2 = spectral_model_map[s2]
                except KeyError:
                    continue

            if i1 == -1 or i2 == -1:
                raise ValueError(f"No conversion found for correlation {corr}."
                                 f" {stokes_schema} are available, but one "
                                 f"of the following combinations "
                                 f"{set(conv_schema.values())} is needed "
                                 f"for conversion to {corr}")

            conv_map[corr] = (fn, i1, i2)

        cplx_type = typingctx.unify_types(
            spectral_model.dtype, types.complex64)
        ret_type = types.Tuple([cplx_type] * len(corr_schema))
        sig = ret_type(spectral_model, source_index, chan_index)

        def indexer_factory(stokes_index):
            """
            Extracts a stokes parameter from a 2D stokes array
            at a variable source_index and constant stokes_index
            """
            def indexer(stokes_array, source_index, chan_index):
                return stokes_array[source_index, chan_index, stokes_index]

            return indexer

        def codegen(context, builder, signature, args):
            array, source_index, chan_index = args
            array_type, source_index_type, chan_index_type = signature.args
            llvm_type = context.get_value_type(signature.return_type)
            corrs = cgutils.get_null_value(llvm_type)

            for c, (conv_fn, i1, i2) in enumerate(conv_map.values()):
                # Extract the first stokes parameter from the stokes array
                sig = array_type.dtype(
                    array_type, source_index_type, chan_index_type)
                s1 = context.compile_internal(
                    builder, indexer_factory(i1),
                    sig, [array, source_index, chan_index])

                # Extract the second stokes parameter from the stokes array
                s2 = context.compile_internal(
                    builder, indexer_factory(i2),
                    sig, [array, source_index, chan_index])

                # Compute correlation from stokes parameters
                sig = signature.return_type[c](
                    array_type.dtype, array_type.dtype)
                corr = context.compile_internal(
                    builder, conv_fn, sig, [s1, s2])

                # Insert result of tuple_getter into the tuple
                corrs = builder.insert_value(corrs, corr, c)

            return corrs

        return sig, codegen

    return corr_convert


class BrightnessTerm(Term):
    def __init__(self, corr_schema="[I,Q,U,V] -> [XX,XY,YX,YY]"):
        bits = [s.strip() for s in corr_schema.strip().split("->")]

        bad_schema = ValueError("corr_schema must have the following form "
                                "\"[I,Q,U,V] -> [XX,XY,YX,YY]\"")

        if len(bits) != 2:
            raise bad_schema

        stokes, corrs = bits

        if not (stokes.startswith("[") and stokes.endswith("]")):
            raise bad_schema

        if not (corrs.startswith("[") and corrs.endswith("]")):
            raise bad_schema

        self.stokes = [s.strip().upper() for s in stokes[1:-1].split(",")]
        self.corrs = [c.strip().upper() for c in corrs[1:-1].split(",")]

        if not all(s in STOKES_TYPES for s in self.stokes):
            raise ValueError(f"{self.stokes} contains "
                             f"invalid stokes parameters")

        if not all(c in STOKES_TYPES for c in self.corrs):
            raise ValueError(f"{self.corrs} contains "
                             f"invalid correlations")

    def dask_schema(self, stokes, spi, ref_freq,
                    chan_freq, spi_base="standard"):
        assert stokes.ndim == 2
        assert spi.ndim == 3
        assert ref_freq.ndim == 1
        assert chan_freq.ndim == 1
        assert isinstance(spi_base, str)

        return {
            "stokes": ("source", "corr"),
            "spi": ("source", "spi", "corr"),
            "ref_freq": ("source",),
            "chan_freq": ("chan",),
            "spi_base": None
        }

    def fields(self, stokes, spi, ref_freq, chan_freq, spi_base="standard"):
        return [("spectral_model", stokes.dtype[:, :, :])]

    STANDARD = 0
    LOG = 1
    LOG10 = 2

    def initialiser(self, state, stokes, spi, ref_freq,
                    chan_freq, spi_base="standard"):
        expected_nstokes = len(self.stokes)

        def brightness(state, stokes, spi, ref_freq,
                       chan_freq, spi_base="standard"):
            nsrc, nstokes = stokes.shape
            nchan, = chan_freq.shape
            nspi = spi.shape[1]

            if nstokes != expected_nstokes:
                raise ValueError("corr_schema stokes don't match "
                                 "provided number of stokes")

            if ((spi_base.startswith("[") and spi_base.endswith("]")) or
                    (spi_base.startswith("(") and spi_base.endswith(")"))):

                list_spi_base = [s.strip().lower()
                                 for s in spi_base.split(",")]

                if len(list_spi_base) != nstokes:
                    raise ValueError("List of spectral bases must equal "
                                     "number of stokes parameters")
            else:
                list_spi_base = [spi_base.lower()] * nstokes

            spectral_model = np.empty((nsrc, nchan, nstokes), stokes.dtype)

            for p, b in enumerate(list_spi_base):
                if b == "standard":
                    for s in range(nsrc):
                        rf = ref_freq[s]

                        for f in range(nchan):
                            freq_ratio = chan_freq[f] / rf
                            spec_model = stokes[s, p]

                            for si in range(0, nspi):
                                term = freq_ratio ** spi[s, si, p]
                                spec_model *= term

                            spectral_model[s, f, p] = spec_model
                elif b == "log":
                    for s in range(nsrc):
                        rf = ref_freq[s]

                        for f in range(nchan):
                            freq_ratio = np.log(chan_freq[f] / rf)
                            spec_model = np.log(stokes[s, p])

                            for si in range(0, nspi):
                                term = spi[s, si, p] * freq_ratio**(si + 1)
                                spec_model += term

                            spectral_model[s, f, p] = np.exp(spec_model)
                elif b == "log10":
                    for s in range(nsrc):
                        rf = ref_freq[s]

                        for f in range(nchan):
                            freq_ratio = np.log10(chan_freq[f] / rf)
                            spec_model = np.log10(stokes[s, p])

                            for si in range(0, nspi):
                                term = spi[s, si, p] * freq_ratio**(si + 1)
                                spec_model += term

                            spectral_model[s, f, p] = 10**spec_model
                else:
                    raise ValueError(
                        "spi_base not in (\"standard\", \"log\", \"log10\")")

            state.spectral_model = spectral_model

        return brightness

    def sampler(self):
        converter = conversion_factory(self.stokes, self.corrs)

        def brightness_sampler(state, s, r, t, a1, a2, c):
            return converter(state.spectral_model, s, c)

        return brightness_sampler
