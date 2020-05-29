# -*- coding: utf-8 -*-

import numpy as np

from africanus.config import config
from africanus.constants import minus_two_pi_over_c
from africanus.util.docs import DocstringTemplate
from africanus.util.numba import generated_jit
from africanus.util.type_inference import infer_complex_dtype

cfg = config.numba_parallel("rime.phase_delay.parallel")
parallel = cfg.get('parallel', False)
axes = cfg.get('axes', set(('source', 'row')) if parallel else ())


@generated_jit(nopython=True, nogil=True,
               cache=not parallel, parallel=parallel)
def phase_delay(lm, uvw, frequency, convention='fourier'):
    # Bake constants in with the correct type
    one = lm.dtype(1.0)
    neg_two_pi_over_c = lm.dtype(minus_two_pi_over_c)
    out_dtype = infer_complex_dtype(lm, uvw, frequency)

    from numba import prange, set_num_threads, get_num_threads

    srange = prange if 'source' in axes else range
    rrange = prange if 'row' in axes else range
    threads = cfg.get('threads', None) if parallel else None

    def _phase_delay_impl(lm, uvw, frequency, convention='fourier'):
        if parallel and threads is not None:
            prev_threads = get_num_threads()
            set_num_threads(threads)

        if convention == 'fourier':
            constant = neg_two_pi_over_c
        elif convention == 'casa':
            constant = -neg_two_pi_over_c
        else:
            raise ValueError("convention not in ('fourier', 'casa')")

        shape = (lm.shape[0], uvw.shape[0], frequency.shape[0])
        complex_phase = np.zeros(shape, dtype=out_dtype)

        # For each source
        for source in srange(lm.shape[0]):
            l, m = lm[source]
            n = np.sqrt(one - l**2 - m**2) - one

            # For each uvw coordinate
            for row in rrange(uvw.shape[0]):
                u, v, w = uvw[row]
                # e^(-2*pi*(l*u + m*v + n*w)/c)
                real_phase = constant * (l * u + m * v + n * w)

                # Multiple in frequency for each channel
                for chan in range(frequency.shape[0]):
                    p = real_phase * frequency[chan]

                    # Our phase input is purely imaginary
                    # so we can can elide a call to exp
                    # and just compute the cos and sin
                    complex_phase[source, row, chan] = np.cos(p) + np.sin(p)*1j

        if parallel and threads is not None:
            set_num_threads(prev_threads)

        return complex_phase

    return _phase_delay_impl


PHASE_DELAY_DOCS = DocstringTemplate(
    r"""
    Computes the phase delay (K) term:

    .. math::

        & {\Large e^{-2 \pi i (u l + v m + w (n - 1))} }

        & \textrm{where } n = \sqrt{1 - l^2 - m^2}

    Notes
    -----

    Corresponds to the complex exponential of the `Van Cittert-Zernike Theorem
    <https://en.wikipedia.org/wiki/Van_Cittert%E2%80%93Zernike_theorem_>`_.

    `MeqTrees
    <https://github.com/ska-sa/meqtrees-timba/blob/
    6a7e873d4d1fe538981dec5851418cbd371b8388/MeqNodes/src/PSVTensor.cc#L314_>`_
    uses the CASA sign convention.

    Parameters
    ----------

    lm : $(array_type)
        LM coordinates of shape :code:`(source, 2)` with
        L and M components in the last dimension.
    uvw : $(array_type)
        UVW coordinates of shape :code:`(row, 3)` with
        U, V and W components in the last dimension.
    frequency : $(array_type)
        frequencies of shape :code:`(chan,)`
    convention : {'fourier', 'casa'}
        Uses the :math:`e^{-2 \pi \mathit{i}}` sign convention
        if ``fourier`` and :math:`e^{2 \pi \mathit{i}}` if
        ``casa``.

    Returns
    -------
    complex_phase : $(array_type)
        complex of shape :code:`(source, row, chan)`
    """)

try:
    phase_delay.__doc__ = PHASE_DELAY_DOCS.substitute(
                            array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
