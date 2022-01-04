import numpy as np
from numba.core import cgutils, types, errors
from numba import njit, objmode

from africanus.rime.parangles_casa import casa_parallactic_angles
from africanus.experimental.rime.fused.transformers.core import Transformer


class ParallacticTransformer(Transformer):
    OUTPUTS = ["parangle_sincos"]

    def init_fields(self, typingctx,
                    utime, ufeed, uantenna,
                    antenna_position, phase_dir,
                    receptor_angle=None):
        dt = typingctx.unify_types(utime.dtype, ufeed.dtype,
                                   antenna_position.dtype,
                                   phase_dir.dtype)
        fields = [("parangle_sincos", dt[:, :, :, :, :])]
        parangle_dt = types.Array(types.float64, 2, "C")
        have_ra = not cgutils.is_nonelike(receptor_angle)

        if have_ra and (not isinstance(receptor_angle, types.Array) or
                        receptor_angle.ndim != 2):
            raise errors.TypingError("receptor_angle must be a 2D array")

        @njit(inline="never")
        def parangle_stub(time, antenna, phase_dir):
            with objmode(out=parangle_dt):
                out = casa_parallactic_angles(time, antenna, phase_dir)

            return out

        def parangles(utime, ufeed, uantenna,
                      antenna_position, phase_dir,
                      receptor_angle=None):

            ntime, = utime.shape
            nant, = uantenna.shape
            nfeed, = ufeed.shape

            # Select out the antennae we're interested in
            antenna_position = antenna_position[uantenna]

            parangles = parangle_stub(utime, antenna_position, phase_dir)
            result = np.empty((ntime, nfeed, nant, 2, 2), parangles.dtype)

            if have_ra:
                if receptor_angle.ndim != 2:
                    raise ValueError("receptor_angle.ndim != 2")

                if receptor_angle.shape[1] != 2:
                    raise ValueError("Only 2 receptor angles "
                                     "currently supported")

                # Select out the feeds we're interested in
                receptor_angle = receptor_angle[ufeed, :]

            for t in range(ntime):
                for f in range(nfeed):
                    ra1 = receptor_angle[f, 0] if have_ra else dt(0)
                    ra2 = receptor_angle[f, 1] if have_ra else dt(0)

                    for a in range(nant):
                        pa1 = parangles[t, a] + ra1
                        pa2 = parangles[t, a] + ra2

                        result[t, f, a, 0, 0] = np.sin(pa1)
                        result[t, f, a, 0, 1] = np.cos(pa1)
                        result[t, f, a, 1, 0] = np.sin(pa2)
                        result[t, f, a, 1, 1] = np.cos(pa2)

            return result

        return fields, parangles

    def dask_schema(self, utime, ufeed, uantenna,
                    antenna_position, phase_dir,
                    receptor_angle=None):
        dt = np.result_type(utime, ufeed, antenna_position,
                            phase_dir, receptor_angle)
        inputs = {"antenna_position": ("antenna", "ant-comp"),
                  "phase_dir": ("radec",)}

        if receptor_angle is not None:
            inputs["receptor_angle"] = ("feed", "receptor_angle")
        else:
            input["receptor_angle"] = None

        outputs = {"parangle_sincos": np.empty((0,)*3, dt)}
        return inputs, outputs
