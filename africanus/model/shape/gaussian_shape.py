# -*- coding: utf-8 -*-


import numpy as np

from africanus.util.docs import DocstringTemplate
from africanus.util.numba import njit, overload, JIT_OPTIONS
from africanus.constants import c as lightspeed


@njit(**JIT_OPTIONS)
def gaussian(uvw, frequency, shape_params):
    return gaussian_impl(uvw, frequency, shape_params)


def gaussian_impl(uvw, frequency, shape_params):
    raise NotImplementedError


@overload(gaussian_impl, jit_options=JIT_OPTIONS)
def nb_gaussian(uvw, frequency, shape_params):
    # https://en.wikipedia.org/wiki/Full_width_at_half_maximum
    fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
    fwhminv = 1.0 / fwhm
    gauss_scale = fwhminv * np.sqrt(2.0) * np.pi / lightspeed

    dtype = np.result_type(
        *(np.dtype(a.dtype.name) for a in (uvw, frequency, shape_params))
    )

    def impl(uvw, frequency, shape_params):
        nsrc = shape_params.shape[0]
        nrow = uvw.shape[0]
        nchan = frequency.shape[0]

        shape = np.empty((nsrc, nrow, nchan), dtype=dtype)
        scaled_freq = np.empty_like(frequency)

        # Scale each frequency
        for f in range(frequency.shape[0]):
            scaled_freq[f] = frequency[f] * gauss_scale

        for s in range(shape_params.shape[0]):
            emaj, emin, angle = shape_params[s]

            # Convert to l-projection, m-projection, ratio
            el = emaj * np.sin(angle)
            em = emaj * np.cos(angle)
            er = emin / (1.0 if emaj == 0.0 else emaj)

            for r in range(uvw.shape[0]):
                u, v, w = uvw[r]

                u1 = (u * em - v * el) * er
                v1 = u * el + v * em

                for f in range(scaled_freq.shape[0]):
                    fu1 = u1 * scaled_freq[f]
                    fv1 = v1 * scaled_freq[f]

                    shape[s, r, f] = np.exp(-(fu1 * fu1 + fv1 * fv1))

        return shape

    return impl


GAUSSIAN_DOCS = DocstringTemplate(
    r"""
Computes the Gaussian Shape Function.

.. math::

    & \nu^\prime = \frac{\nu \pi}{2 c \sqrt{\log{2}}} \\
    & r = \frac{e_{min}}{e_{maj}} \\
    & u_{1} = (u \, e_{maj} \, cos(\alpha) - v \, e_{maj} \, sin(\alpha))
      r \nu^\prime \\
    & v_{1} = (u \, e_{maj} \, sin(\alpha) - v \, e_{maj} \, cos(\alpha))
      \nu^\prime \\
    & \textrm{shape} = e^{(-(u_{1}^2) - (v_{1}^2))}

where:

- :math:`u` and :math:`v` are the UV coordinates and
  :math:`\nu` the frequency.
- :math:`e_{maj}` and :math:`e_{min}` are the major and minor axes
  and :math:`\alpha` the position angle.

Parameters
----------
uvw : $(array_type)
    UVW coordinates of shape :code:`(row, 3)`
frequency : $(array_type)
    frequencies of shape :code:`(chan,)`
shape_param : $(array_type)
    Gaussian Shape Parameters of shape :code:`(source, 3)`
    where the second dimension contains the
    `(emajor, eminor, angle)` parameters describing
    the shape of the Gaussian

Returns
-------
gauss_shape : $(array_type)
    Shape parameters of shape :code:`(source, row, chan)`
"""
)

try:
    gaussian.__doc__ = GAUSSIAN_DOCS.substitute(array_type=":class:`numpy.ndarray`")
except KeyError:
    pass
