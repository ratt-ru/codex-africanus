from africanus.constants import c as lightspeed
import numpy as np

from africanus.experimental.rime.fused.terms.core import Term


class Phase(Term):
    """Phase Delay Term"""
    def dask_schema(self, lm, uvw, chan_freq, convention="fourier"):
        assert lm.ndim == 2
        assert uvw.ndim == 2
        assert chan_freq.ndim == 1
        assert isinstance(convention, str)

        return {"lm": ("source", "lm"),
                "uvw": ("row", "uvw"),
                "chan_freq": ("chan",),
                "convention": None}

    def init_fields(self, typingctx, lm, uvw, chan_freq, convention="fourier"):
        phase_dt = typingctx.unify_types(lm.dtype, uvw.dtype, chan_freq.dtype)
        fields = [("phase_dot", phase_dt[:, :])]

        def phase(lm, uvw, chan_freq, convention="fourier"):
            nsrc, _ = lm.shape
            nrow, _ = uvw.shape
            nchan, = chan_freq.shape

            phase_dot = np.empty((nsrc, nrow), dtype=phase_dt)

            zero = lm.dtype.type(0.0)
            one = lm.dtype.type(1.0)

            if convention == "fourier":
                C = phase_dt(-2.0*np.pi/lightspeed)
            elif convention == "casa":
                C = phase_dt(2.0*np.pi/lightspeed)
            else:
                raise ValueError("convention not in (\"fourier\", \"casa\")")

            for s in range(nsrc):
                l = lm[s, 0]  # noqa
                m = lm[s, 1]
                n = one - l**2 - m**2
                n = np.sqrt(zero if n < zero else n) - one

                for r in range(nrow):
                    u = uvw[r, 0]
                    v = uvw[r, 1]
                    w = uvw[r, 2]

                    phase_dot[s, r] = C*(l*u + m*v + n*w)

            return phase_dot

        return fields, phase

    def sampler(self):
        def phase_sample(state, s, r, t, f1, f2, a1, a2, c):
            p = state.phase_dot[s, r] * state.chan_freq[c]
            return np.cos(p) + np.sin(p)*1j

        return phase_sample
