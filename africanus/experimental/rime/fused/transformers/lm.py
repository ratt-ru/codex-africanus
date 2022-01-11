import numpy as np

from africanus.experimental.rime.fused.transformers.core import Transformer


class LMTransformer(Transformer):
    OUTPUTS = ["lm"]

    def init_fields(self, typingctx, radec, phase_dir):
        dt = typingctx.unify_types(radec.dtype, phase_dir.dtype)
        fields = [("lm", dt[:, :])]

        def lm(radec, phase_dir):
            lm = np.empty_like(radec)
            pc_ra = phase_dir[0]
            pc_dec = phase_dir[1]

            sin_pc_dec = np.sin(pc_dec)
            cos_pc_dec = np.cos(pc_dec)

            for s in range(radec.shape[0]):
                da = radec[s, 0] - pc_ra
                sin_ra_delta = np.sin(da)
                cos_ra_delta = np.cos(da)

                sin_dec = np.sin(radec[s, 1])
                cos_dec = np.cos(radec[s, 1])

                lm[s, 0] = cos_dec*sin_ra_delta
                lm[s, 1] = sin_dec*cos_pc_dec - cos_dec*sin_pc_dec*cos_ra_delta

            return lm

        return fields, lm

    def dask_schema(self, radec, phase_dir):
        assert radec.ndim == 2
        assert phase_dir.ndim == 1

        inputs = {
            "radec": ("source", "radec"),
            "phase_dir": ("radec",)
        }

        outputs = {"lm": np.empty((0, 0), dtype=radec.dtype)}

        return inputs, outputs
