import numpy as np
from numba.core import types

from africanus.rime.monolithic.transformers.core import Transformer


class LMTransformer(Transformer):
    OUTPUTS = ["lm"]

    def outputs(self, radec, phase_centre):
        if not isinstance(radec, types.Array) or radec.ndim != 2:
            raise ValueError(f"{radec} must be a (source, radec) array")

        if not isinstance(phase_centre, types.Array) or radec.ndim != 1:
            raise ValueError(f"{phase_centre} must be a 1D array")

        dt = self.result_type(radec.dtype, phase_centre.dtype)
        return [("lm", types.Array(dt, radec.ndim, radec.layout))]

    def transform(self):
        def lm(radec, phase_centre):
            lm = np.empty_like(radec)
            pc_ra = phase_centre[0]
            pc_dec = phase_centre[1]

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

        return lm

    def dask_schema(self, radec, phase_centre):
        return {"radec": ("source", "radec"),
                "phase_centre": ("radec",)}
