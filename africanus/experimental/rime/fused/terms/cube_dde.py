
from collections import namedtuple

from numba.core import cgutils, types
from numba.extending import intrinsic
from numba.cpython.unsafe.tuple import tuple_setitem
import numpy as np

from africanus.experimental.rime.fused.terms.core import Term


def zero_vis_factory(ncorr):
    @intrinsic
    def zero_vis(typingctx, value):
        sig = types.Tuple([value]*ncorr)(value)

        def codegen(context, builder, signature, args):
            llvm_ret_type = context.get_value_type(signature.return_type)
            tup = cgutils.get_null_value(llvm_ret_type)

            for i in range(ncorr):
                tup = builder.insert_value(tup, args[0], i)

            return tup

        return sig, codegen

    return zero_vis


BeamInfo = namedtuple("BeamInfo", [
    "lscale", "mscale",
    "lmaxi", "mmaxi", "lmaxf", "mmaxf"])


class BeamCubeDDE(Term):
    """Voxel Beam Cube Term"""
    def __init__(self, configuration, corrs):
        if configuration not in {"left", "right"}:
            raise ValueError(f"BeamCubeDDE configuration must be"
                             f"either 'left' or 'right'. "
                             f"Got {configuration}")

        super().__init__(configuration)
        self.corrs = corrs

    def dask_schema(self, beam, beam_lm_extents, beam_freq_map,
                    lm, beam_parangle, chan_freq,
                    beam_point_errors=None,
                    beam_antenna_scaling=None):
        return {
            "beam": ("beam_lw", "beam_mh", "beam_nud", "corr"),
            "beam_lm_extents": ("lm_ext", "lm_ext_comp"),
            "beam_freq_map": ("beam_nud",),
            "lm": ("source", "lm"),
            "chan_freq": ("chan",),
        }

    def init_fields(self, typingctx,
                    beam, beam_lm_extents, beam_freq_map,
                    lm, beam_parangle, chan_freq,
                    beam_point_errors=None,
                    beam_antenna_scaling=None):

        ncorr = len(self.corrs)
        ex_dtype = beam_lm_extents.dtype
        beam_info_types = [ex_dtype]*2 + [types.int64]*2 + [types.float64]*2
        beam_info_type = types.NamedTuple(beam_info_types, BeamInfo)

        fields = [("beam_freq_data", chan_freq.copy(ndim=2)),
                  ("beam_info", beam_info_type)]

        def beam(beam, beam_lm_extents, beam_freq_map,
                 lm, beam_parangle, chan_freq,
                 beam_point_errors=None,
                 beam_antenna_scaling=None):

            if beam.shape[3] != ncorr:
                raise ValueError(
                    "Beam correlations don't match specification corrs")

            freq_data = np.empty((chan_freq.shape[0], 3), chan_freq.dtype)
            beam_nud = beam_freq_map.shape[0]
            beam_lw, beam_mh, beam_nud = beam.shape[:3]

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

            # Beam Extents
            lower_l, upper_l = beam_lm_extents[0]
            lower_m, upper_m = beam_lm_extents[1]

            # Maximum l and m indices in float and int
            lmaxf = ex_dtype(beam_lw - types.int64(1))
            mmaxf = ex_dtype(beam_mh - types.int64(1))
            lmaxi = beam_lw - types.int64(1)
            mmaxi = beam_mh - types.int64(1)

            lscale = lmaxf / (upper_l - lower_l)
            mscale = mmaxf / (upper_m - lower_m)

            beam_info = BeamInfo(lscale, mscale, lmaxi, mmaxi, lmaxf, mmaxf)
            return freq_data, beam_info

        return fields, beam

    def sampler(self):
        left = self.configuration == "left"
        ncorr = len(self.corrs)
        zero_vis = zero_vis_factory(ncorr)

        def cube_dde(state, s, r, t, f1, f2, a1, a2, c):
            a = state.antenna1_index[r] if left else state.antenna2_index[r]
            feed = state.feed1_index[r] if left else state.feed2_index[r]
            sin_pa = state.beam_parangle[t, feed, a, 0]
            cos_pa = state.beam_parangle[t, feed, a, 1]

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
            # tl = sl + point_errors[t, a, c, 0]
            # tm = sm + point_errors[t, a, c, 1]
            tl = sl
            tm = sm

            # Rotate lm coordinate angle
            vl = tl*cos_pa - tm*sin_pa
            vm = tl*sin_pa + tm*cos_pa

            # Scale by antenna scaling
            # vl *= antenna_scaling[a, f, 0]
            # vm *= antenna_scaling[a, f, 1]

            # Beam Extents
            lower_l, upper_l = state.beam_lm_extents[0]
            lower_m, upper_m = state.beam_lm_extents[1]

            # Shift into the cube coordinate system
            vl = state.beam_info.lscale*(vl - lower_l)
            vm = state.beam_info.mscale*(vm - lower_m)

            # Clamp the coordinates to the edges of the cube
            vl = max(0.0, min(vl, state.beam_info.lmaxf))
            vm = max(0.0, min(vm, state.beam_info.mmaxf))

            # Snap to the lower grid coordinates
            gl0 = np.int32(np.floor(vl))
            gm0 = np.int32(np.floor(vm))

            # Snap to the upper grid coordinates
            gl1 = min(gl0 + np.int32(1), state.beam_info.lmaxi)
            gm1 = min(gm0 + np.int32(1), state.beam_info.mmaxi)

            # Difference between grid and offset coordinates
            ld = vl - gl0
            md = vm - gm0

            corr_sum = zero_vis(state.beam.dtype.type(0))
            absc_sum = zero_vis(state.beam.real.dtype.type(0))

            # Lower cube
            weight = (1.0 - ld)*(1.0 - md)*nud

            for co in range(ncorr):
                value = state.beam[gl0, gm0, gc0, co]
                absc_sum = tuple_setitem(absc_sum, co,
                                         weight*np.abs(value) + absc_sum[co])
                corr_sum = tuple_setitem(corr_sum, co,
                                         weight*value + corr_sum[co])

            weight = ld*(1.0 - md)*nud

            for co in range(ncorr):
                value = state.beam[gl1, gm0, gc0, co]
                absc_sum = tuple_setitem(absc_sum, co,
                                         weight*np.abs(value) + absc_sum[co])
                corr_sum = tuple_setitem(corr_sum, co,
                                         weight*value + corr_sum[co])

            weight = (1.0 - ld)*md*nud

            for co in range(ncorr):
                value = state.beam[gl0, gm1, gc0, co]
                absc_sum = tuple_setitem(absc_sum, co,
                                         weight*np.abs(value) + absc_sum[co])
                corr_sum = tuple_setitem(corr_sum, co,
                                         weight*value + corr_sum[co])

            weight = ld*md*nud

            for co in range(ncorr):
                value = state.beam[gl1, gm1, gc0, co]
                absc_sum = tuple_setitem(absc_sum, co,
                                         weight*np.abs(value) + absc_sum[co])
                corr_sum = tuple_setitem(corr_sum, co,
                                         weight*value + corr_sum[co])

            # Upper cube
            weight = (1.0 - ld)*(1.0 - md)*inv_nud

            for co in range(ncorr):
                value = state.beam[gl0, gm0, gc1, co]
                absc_sum = tuple_setitem(absc_sum, co,
                                         weight*np.abs(value) + absc_sum[co])
                corr_sum = tuple_setitem(corr_sum, co,
                                         weight*value + corr_sum[co])

            weight = ld*(1.0 - md)*inv_nud

            for co in range(ncorr):
                value = state.beam[gl1, gm0, gc1, co]
                absc_sum = tuple_setitem(absc_sum, co,
                                         weight*np.abs(value) + absc_sum[co])
                corr_sum = tuple_setitem(corr_sum, co,
                                         weight*value + corr_sum[co])

            weight = (1.0 - ld)*md*inv_nud

            for co in range(ncorr):
                value = state.beam[gl0, gm1, gc1, co]
                absc_sum = tuple_setitem(absc_sum, co,
                                         weight*np.abs(value) + absc_sum[co])
                corr_sum = tuple_setitem(corr_sum, co,
                                         weight*value + corr_sum[co])

            weight = ld*md*inv_nud

            for co in range(ncorr):
                value = state.beam[gl1, gm1, gc1, co]
                absc_sum = tuple_setitem(absc_sum, co,
                                         weight*np.abs(value) + absc_sum[co])
                corr_sum = tuple_setitem(corr_sum, co,
                                         weight*value + corr_sum[co])

            for co in range(ncorr):
                div = np.abs(corr_sum[co])
                value = corr_sum[co]*absc_sum[co]

                if div != 0.0:
                    value /= div

                corr_sum = tuple_setitem(corr_sum, co, value)

            return corr_sum

        return cube_dde
