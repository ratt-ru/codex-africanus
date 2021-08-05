from numba.extending import intrinsic
from numba.experimental import structref

from africanus.util.casa_types import STOKES_ID_MAP, STOKES_TYPES
from africanus.rime.monolothic.terms import TermStructRef, Term


stokes_conv = {
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
    def corr_convert(typingctx, stokes):
        pass

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

        def brightness(stokes, chan_freq):
            state = structref.new(struct_type)
            state.stokes = stokes
            state.chan_freq = chan_freq
            return state

        return brightness

    def sampler(self):
        def brightness_sampler(state, s, r, t, a1, a2, c):
            # I Q U V
            XX = (state.stokes[s, 0] + state.stokes[s, 1]) / 2.0
            XY = (state.stokes[s, 2] + state.stokes[s, 3])*1j / 2.0
            YX = (state.stokes[s, 2] - state.stokes[s, 3])*1j / 2.0
            YY = (state.stokes[s, 0] - state.stokes[s, 1]) / 2.0
            return XX, XY, YX, YY

        return brightness_sampler
