from numba.experimental import structref

from africanus.rime.monolothic.terms import TermStructRef, Term


@structref.register
class BrightnessType(TermStructRef):
    pass


class BrightnessTerm(Term):
    term_args = ["stokes", "chan_freq"]
    term_kwargs = []
    arg_schema = {
        "stokes": ("source", "corr"),
        "chan_freq": ("chan",)
    }

    abstract_type = BrightnessType

    @classmethod
    def term_type(cls, stokes, chan_freq):
        return cls.abstract_type([
            ("stokes", stokes),
            ("chan_freq", chan_freq)
        ])

    @classmethod
    def initialiser(cls, stokes, chan_freq):
        struct_type = cls.term_type(stokes, chan_freq)

        def brightness(stokes, chan_freq):
            state = structref.new(struct_type)
            state.stokes = stokes
            state.chan_freq = chan_freq
            return state

        return brightness

    @classmethod
    def sampler(cls):
        def brightness_sampler(state, s, r, t, a1, a2, c):
            # I Q U V
            XX = (state.stokes[s, 0] + state.stokes[s, 1]) / 2.0
            XY = (state.stokes[s, 2] + state.stokes[s, 3])*1j / 2.0
            YX = (state.stokes[s, 2] - state.stokes[s, 3])*1j / 2.0
            YY = (state.stokes[s, 0] - state.stokes[s, 1]) / 2.0
            return XX, XY, YX, YY

        return brightness_sampler
