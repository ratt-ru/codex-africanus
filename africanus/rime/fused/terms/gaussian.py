from africanus.constants import c as lightspeed
import numpy as np

from africanus.rime.fused.terms.core import Term


class GaussianTerm(Term):
    def dask_schema(self, uvw, chan_freq, gauss_shape):
        assert uvw.ndim == 2
        assert chan_freq.ndim == 1
        assert gauss_shape.ndim == 2

        return {"uvw": ("row", "uvw"),
                "chan_freq": ("chan",),
                "gauss_shape": ("source", "gauss_shape_params")}

    def fields(self, uvw, chan_freq, gauss_shape):
        unified = self.result_type(uvw, chan_freq, gauss_shape)
        return [("gauss_uv", unified[:, :, :]),
                ("scaled_freq", chan_freq)]

    def initialiser(self, state, uvw, chan_freq, gauss_shape):
        guv_dtype = state.field_dict["gauss_uv"].dtype

        fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
        fwhminv = 1.0 / fwhm
        gauss_scale = fwhminv * np.sqrt(2.0) * np.pi / lightspeed

        def gaussian_init(state, uvw, chan_freq, gauss_shape):
            nsrc, _ = gauss_shape.shape
            nrow, _ = uvw.shape

            state.gauss_uv = np.empty((nsrc, nrow, 2), dtype=guv_dtype)
            state.scaled_freq = chan_freq*gauss_scale

            for s in range(nsrc):
                emaj, emin, angle = gauss_shape[s]

                # Convert to l-projection, m-projection, ratio
                el = emaj * np.sin(angle)
                em = emaj * np.cos(angle)
                er = emin / (1.0 if emaj == 0.0 else emaj)

                for r in range(uvw.shape[0]):
                    u, v, w = uvw[r]

                    state.gauss_uv[s, r, 0] = (u*em - v*el)*er
                    state.gauss_uv[s, r, 1] = u*el + v*em

        return gaussian_init

    def sampler(self):
        def gaussian_sample(state, s, r, t, a1, a2, c):
            fu1 = state.gauss_uv[s, r, 0] * state.scaled_freq[c]
            fv1 = state.gauss_uv[s, r, 1] * state.scaled_freq[c]
            return np.exp(-(fu1*fu1 + fv1*fv1))

        return gaussian_sample
