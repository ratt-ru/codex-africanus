from numba.extending import intrinsic
from numba.experimental import structref
from numba.core import cgutils, compiler, types, errors
from numba.core.typed_passes import type_inference_stage


from africanus.util.casa_types import STOKES_ID_MAP, STOKES_TYPES
from africanus.rime.monolothic.terms import TermStructRef, Term


STOKES_CONVERSION = {
    'RR': {('I', 'V'): lambda i, v: i + v + 0j},
    'RL': {('Q', 'U'): lambda q, u: q + u*1j},
    'LR': {('Q', 'U'): lambda q, u: q - u*1j},
    'LL': {('I', 'V'): lambda i, v: i - v + 0j},

    'XX': {('I', 'Q'): lambda i, q: i + q + 0j},
    'XY': {('U', 'V'): lambda u, v: u + v*1j},
    'YX': {('U', 'V'): lambda u, v: u - v*1j},
    'YY': {('I', 'Q'): lambda i, q: i - q + 0j},
}


def conversion_factory(stokes_schema, corr_schema):
    @intrinsic
    def corr_convert(typingctx, stokes, index):
        if not isinstance(stokes, types.Array) or stokes.ndim != 2:
            raise errors.TypingError(f"'stokes' should be 2D array. Got {stokes}")

        if not isinstance(index, types.Integer):
            raise errors.TypingError(f"'index' should be an integer. Got {index}")

        stokes_map = {s: i for i, s in enumerate(stokes_schema)}
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
                    i1 = stokes_map[s1]
                    i2 = stokes_map[s2]
                except KeyError:
                    continue

            if i1 == -1 or i2 == -1:
                raise ValueError(f"No conversion found for correlation {corr}. "
                                 f"{stokes_schema} are available, but one "
                                 f"of the following combinations "
                                 f"{set(conv_schema.values())} is needed "
                                 f"for conversion to {corr}")

            conv_map[corr] = (fn, i1, i2)

        cplx_type = typingctx.unify_types(stokes.dtype, types.complex64)
        ret_type = types.Tuple([cplx_type] * len(corr_schema))

        ir = [compiler.run_frontend(tup[0]) for tup in conv_map.values()]
        sig = ret_type(stokes, index)

        def codegen(context, builder, signature, args):
            ret_type = signature.return_type
            llvm_type = context.get_value_type(signature.return_type)
            return cgutils.get_null_value(llvm_type)            

        return sig, codegen

    return corr_convert


@structref.register
class BrightnessType(TermStructRef):
    pass


class BrightnessTerm(Term):
    arg_schema = {
        "stokes": ("source", "corr"),
        "chan_freq": ("chan",)
    }

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

    def term_type(self, stokes, chan_freq):
        return BrightnessType([
            ("stokes", stokes),
            ("chan_freq", chan_freq)
        ])

    def initialiser(self, stokes, chan_freq):
        struct_type = self.term_type(stokes, chan_freq)
        specced_stokes = len(self.stokes)

        def brightness(stokes, chan_freq):
            _, nstokes = stokes.shape

            if nstokes != specced_stokes:
                raise ValueError("corr_schema stokes don't match "
                                 "provided number of stokes")

            state = structref.new(struct_type)
            state.stokes = stokes
            state.chan_freq = chan_freq
            return state

        return brightness

    def sampler(self):
        converter = conversion_factory(self.stokes, self.corrs)

        def brightness_sampler(state, s, r, t, a1, a2, c):
            return converter(state.stokes, s)

        return brightness_sampler
