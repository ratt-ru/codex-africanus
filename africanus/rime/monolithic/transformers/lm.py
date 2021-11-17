import numpy as np
from numba.core import types

from africanus.rime.monolithic.transformers.core import Transformer


class LMTransformer(Transformer):
    OUTPUTS = ["lm"]

    def fields(self, radec, phase_centre):
        if not isinstance(radec, types.Array) or radec.ndim != 2:
            raise ValueError(f"{radec} must be a (source, radec) array")

        if not isinstance(phase_centre, types.Array) or radec.ndim != 1:
            raise ValueError(f"{phase_centre} must be a 1D array")

        dt = self.result_type(radec.dtype, phase_centre.dtype)
        return [("lm", types.Array(dt, radec.ndim, radec.layout))]

    def transform(self):
        def lm(radec, phase_centre):
            return np.arange(radec.shape[0])

        return lm

    def dask_schema(self, radec, phase_centre):
        return {"radec": ("source", "radec"),
                "phase_centre": ("radec",)}
