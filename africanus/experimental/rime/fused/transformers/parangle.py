import threading

import numpy as np
from numba.core import types
from numba import njit, objmode

from africanus.experimental.rime.fused.transformers.core import Transformer


class ParallacticTransformer(Transformer):
    OUTPUTS = ["parangle_sin_cos"]

    def __init__(self):
        super().__init__()
        self._thread_local = threading.local()

    def casa_parangle_factory(self, dtype):
        try:
            import pyrap.measures as pm
            import pyrap.quanta as pq
        except ImportError:
            raise ImportError("pip install codex-africanus[python-casacore]")

        _thread_local = self._thread_local

        def _casa_parangles(time, antenna, phase_dir, zenith_frame="AZEL"):
            try:
                meas_serv = _thread_local.meas_serv
            except AttributeError:
                # Create a measures server
                _thread_local.meas_serv = meas_serv = pm.measures()

            # Create direction measure for the zenith
            zenith = meas_serv.direction(zenith_frame, '0deg', '90deg')

            # Create position measures for each antenna
            reference_positions = [meas_serv.position(
                                        'itrf',
                                        *(pq.quantity(x, 'm') for x in pos))
                                   for pos in antenna]

            # Compute field centre in radians
            fc_rad = meas_serv.direction('J2000', *(pq.quantity(f, 'rad')
                                                    for f in phase_dir))

            return np.asarray([
                # Set current time as the reference frame
                meas_serv.do_frame(meas_serv.epoch("UTC", pq.quantity(t, "s")))
                and
                [   # Set antenna position as the reference frame
                    meas_serv.do_frame(rp)
                    and
                    meas_serv.posangle(fc_rad, zenith).get_value("rad")
                    for rp in reference_positions
                ]
                for t in time], np.float64)

        out_dtype = types.Array(types.float64, 2, "C")

        @njit(inline="never")
        def parangle_stub(time, antenna, phase_dir):
            with objmode(out=out_dtype):
                out = _casa_parangles(time, antenna, phase_dir)

            return out

        return parangle_stub

    def init_fields(self, typingctx, utime, antenna_position, phase_dir):
        dt = typingctx.unify_types(utime.dtype,
                                   antenna_position.dtype,
                                   phase_dir.dtype)
        fields = [("parangle_sin_cos", dt[:, :, :])]

        parangle_stub = self.casa_parangle_factory(dt)

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
        inputs = {"parallactic_angles": ("time", "ant")}
        outputs = {"parangle_sin_cos": np.empty((0,)*3, dt)}
        return inputs, outputs
