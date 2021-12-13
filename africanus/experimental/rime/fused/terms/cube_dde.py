from numba.core import types
import numpy as np

from africanus.experimental.rime.fused.terms.core import Term


class CubeDDE(Term):
    def __init__(self, configuration):
        if configuration not in {"left", "right"}:
            raise ValueError(f"CubeDDE configuration must be"
                             f"either 'left' or 'right'. "
                             f"Got {configuration}")

        super().__init__(configuration)


    def init_fields(self, typingctx,
                    beam, beam_lm_extents, beam_freq_map,
                    lm, parangle_sincos, chan_freq,
                    beam_point_errors=None,
                    beam_antenna_scaling=None):
        
        fields = [("beam_freq_data", chan_freq.copy(ndim=2))]

        def beam(beam, beam_lm_extents, beam_freq_map,
                 lm, parangle_sincos, chan_freq,
                 beam_point_errors=None,
                 beam_antenna_scaling=None):

            freq_data = np.empty((chan_freq.shape[0], 3), chan_freq.dtype)
            beam_nud = beam_freq_map.shape[0]
            beam_lw, beam_mh, beam_nud = beam.shape[:3]
            corrs = beam.shape[4]

            if beam_lw < 2 or beam_mh < 2 or beam_nud < 2:
                raise ValueError("beam_lw, beam_mh and beam_nud must be >= 2")

            for f in range(chan_freq.shape[0]):
                freq = chan_freq[f]
                lower = 0
                upper = beam_nud - 1

                while lower <= upper:
                    mid = lower + (upper - lower) // 2
                    beam_freq = beam_freq_map[mid]

                    if beam_freq < freq:
                        lower = mid + 1
                    elif beam_freq > freq:
                        upper = mid - 1
                    else:
                        lower = mid
                        break

                # This handles the lower <= upper in the while loop
                lower = min(lower, upper)
                upper = lower + 1

                # Set up scaling, lower weight, lower grid pos
                if lower == -1:
                    freq_data[f, 0] = freq / beam_freq_map[0]
                    freq_data[f, 1] = 1.0
                    freq_data[f, 2] = 0
                elif upper == beam_nud:
                    freq_data[f, 0] = freq / beam_freq_map[beam_nud - 1]
                    freq_data[f, 1] = 0.0
                    freq_data[f, 2] = beam_nud - 2
                else:
                    freq_data[f, 0] = 1.0
                    freq_low = beam_freq_map[lower]
                    freq_high = beam_freq_map[upper]
                    freq_diff = freq_high - freq_low
                    freq_data[f, 1] = (freq_high - freq) / freq_diff
                    freq_data[f, 2] = lower

                return freq_data

            return fields, beam

    def sampler(self):
        left = self.configuration == "left"

        def cube_dde(state, s, r, t, a1, a2, c):
            a = a1 if left else a2
            sin_pa = state.parangle_sincos[t, a, 0]
            cos_pa = state.parangle_sincos[t, a, 1]

            l = state.lm[s, 0]  # noqa
            m = state.lm[s, 1]

            # Unpack frequency data
            freq_scale = state.beam_freq_data[c, 0]
            # lower and upper frequency weights
            nud = state.beam_freq_data[c, 1]
            inv_nud = state.beam_freq_data.dtype.type(1.0) - nud
            # lower and upper frequency grid position
            gc0 = np.int32(state.beam_freq_data[c, 2])
            gc1 = gc0 + np.int32(1)

            # Apply any frequency scaling
            sl = l * freq_scale
            sm = m * freq_scale

            # Add pointing errors
            tl = sl + point_errors[t, a, c, 0]
            tm = sm + point_errors[t, a, c, 1]

            # Rotate lm coordinate angle
            vl = tl*cos_pa - tm*sin_pa
            vm = tl*sin_pa + tm*cos_pa

            # Scale by antenna scaling
            vl *= antenna_scaling[a, f, 0]
            vm *= antenna_scaling[a, f, 1]

            # Shift into the cube coordinate system
            vl = lscale*(vl - lower_l)
            vm = mscale*(vm - lower_m)

            # Clamp the coordinates to the edges of the cube
            vl = max(zero, min(vl, lmaxf))
            vm = max(zero, min(vm, mmaxf))

            # Snap to the lower grid coordinates
            gl0 = np.int32(np.floor(vl))
            gm0 = np.int32(np.floor(vm))

            # Snap to the upper grid coordinates
            gl1 = min(gl0 + np.int32(1), lmaxi)
            gm1 = min(gm0 + np.int32(1), mmaxi)

            # Difference between grid and offset coordinates
            ld = vl - gl0
            md = vm - gm0


        return cube_dde


