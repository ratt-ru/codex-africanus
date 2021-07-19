from numba.experimental import structref
import numpy as np

from africanus.rime.monolothic.terms import TermStructRef, Term


@structref.register
class PhaseType(TermStructRef):
    pass


class PhaseTerm(Term):
    term_args = ["lm", "uvw", "chan_freq"]
    optional_args = ["convention"]
    abstract_type = PhaseType

    @classmethod
    def term_type(cls, *args):
        assert len(cls.term_args) == len(args)
        lm, uvw, chan_freq = args
        phase_dot = cls.result_type(lm, uvw, chan_freq)
        return cls.abstract_type([
            ("lm", lm),
            ("uvw", uvw),
            ("chan_freq", chan_freq),
            ("phase_dot", phase_dot[:, :])
        ])

    @classmethod
    def initialiser(cls, *args):
        struct_type = cls.term_type(*args)
        dot_dtype = struct_type.field_dict["phase_dot"].dtype

        def phase(lm, uvw, chan_freq):
            nsrc, _ = lm.shape
            nrow, _ = uvw.shape
            nchan, = chan_freq.shape

            state = structref.new(struct_type)
            state.lm = lm
            state.uvw = uvw
            state.chan_freq = chan_freq
            state.phase_dot = np.empty((nsrc, nrow), dtype=dot_dtype)

            zero = lm.dtype.type(0.0)
            one = lm.dtype.type(1.0)
            C = dot_dtype(-2*np.pi/3e8)

            for s in range(nsrc):
                l = lm[s, 0]  # noqa
                m = lm[s, 1]
                n = one - l**2 - m**2
                n = np.sqrt(zero if n < zero else n) - one

                for r in range(nrow):
                    u = uvw[r, 0]
                    v = uvw[r, 1]
                    w = uvw[r, 2]

                    state.phase_dot[s, r] = C*(l*u + m*v + n*w)

            return state

        return phase

    @classmethod
    def sampler(cls):
        def phase_sample(state, s, r, t, a1, a2, c):
            return np.exp(state.phase_dot[s, r] * state.chan_freq[c])

        return phase_sample
