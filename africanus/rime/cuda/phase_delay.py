from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from numba import cuda


@cuda.jit
def phase_delay_kernel(lm, uvw, frequency, out):
    chan, row = cuda.grid(2)

    if chan >= frequency.shape[0] or row >= uvw.shape[0]:
        return

    u, v, w = uvw[row]
    freq = frequency[chan]

    shared_uvw = cuda.shared.array((TPB, 3), dtype=uvw_dtype)
    shared_freq = cuda.shared.array(TPB, dtype=freq_dtype)

    for source in range(lm.shape[0]):
        l, m = lm[source]
        n = math.sqrt(1.0 - l**2 - m**2) - 1.0
        real_phase = -2.0*math.pi*(l*u + m*v + n*w)*lightspeed/freq
        out[source, row, chan] = math.cos(real_phase) + 1j*math.sin(real_phase)
