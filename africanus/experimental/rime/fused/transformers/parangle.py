import numpy as np
from numba.core import types
from numba import njit, objmode

from africanus.rime.parangles_casa import casa_parallactic_angles
from africanus.experimental.rime.fused.transformers.core import Transformer


class ParallacticTransformer(Transformer):
    OUTPUTS = ["parangle_sincos"]

    def init_fields(self, typingctx, utime, antenna_position, phase_dir):
        dt = typingctx.unify_types(utime.dtype,
                                   antenna_position.dtype,
                                   phase_dir.dtype)
        fields = [("parangle_sincos", dt[:, :, :])]
        parangle_dt = types.Array(types.float64, 2, "C")

        @njit(inline="never")
        def parangle_stub(time, antenna, phase_dir):
            with objmode(out=parangle_dt):
                out = casa_parallactic_angles(time, antenna, phase_dir)

            return out

        def parangles(utime, antenna_position, phase_dir):
            parangles = parangle_stub(utime, antenna_position, phase_dir)
            ntime, nant = parangles.shape
            result = np.empty((ntime, nant, 2), parangles.dtype)

            for t in range(ntime):
                for a in range(nant):
                    result[t, a, 0] = np.sin(parangles[t, a])
                    result[t, a, 1] = np.cos(parangles[t, a])

            return result

        return fields, parangles

    def dask_schema(self, utime, antenna_position, phase_dir):
        dt = np.result_type(utime, antenna_position, phase_dir)
        inputs = {"antenna_position": ("antenna", "ant-comp"),
                  "phase_dir": ("radec",)}
        outputs = {"parangle_sincos": np.empty((0,)*3, dt)}
        return inputs, outputs
