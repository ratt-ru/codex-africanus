.. _convolution-filter-api:

-------------------
Convolution Filters
-------------------

Convolution filters suitable for use in gridding and degridding.

API
~~~

.. currentmodule:: africanus.filters

.. autosummary::
    convolution_filter


.. autofunction:: convolution_filter
.. autodata:: ConvolutionFilter


.. _kaiser-bessel-filter:

Kaiser Bessel
~~~~~~~~~~~~~

The `Kaiser Bessel
<https://www.dsprelated.com/freebooks/sasp/Kaiser_Window.html>`_
function.

.. currentmodule:: africanus.filters.kaiser_bessel_filter

.. autosummary::
    kaiser_bessel
    kaiser_bessel_with_sinc
    kaiser_bessel_fourier
    estimate_kaiser_bessel_beta


.. autofunction:: kaiser_bessel
.. autofunction:: kaiser_bessel_with_sinc
.. autofunction:: kaiser_bessel_fourier
.. autofunction:: estimate_kaiser_bessel_beta


Sinc
~~~~

The `Sinc <https://en.wikipedia.org/wiki/Sinc_function>`_ function.

