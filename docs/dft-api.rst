Direct Fourier Transform
------------------------

Functions used to compute the discretised
direct Fourier transform (DFT)
for an ideal unpolarised interferometer.
The DFT for an ideal interferometer is
defined as

.. math::

    V(u,v,w) = \int I(l,m) e^{-2\pi i
        \left( ul + vm + w(n-1)\right)}
        \frac{dl dm}{n}

where :math:`u,v,w` are data (visibility :math:`V`) space
coordinates and :math:`l,m,n` are signal (image :math:`I`)
space coordinates.
Note that the data space coordinates have an implicit
dependence on frequency and time and that the image
has an implicit dependence on frequency.
The discretised form of the DFT can be written as

.. math::

    V(u,v,w) = \sum_s \frac{1}{n_s} e^{-2 \pi i
        (u l_s + v m_s + w (n_s - 1))} \cdot I_s

where :math:`s` labels the source (or pixel) location.
This can be cast into a matrix equation as follows

.. math::

    V = R I

where :math:`R` is the operator that maps an
image to visibility space (note that the
coordinate :math:`n_s` gets absorbed into
the definition of :math:`R`). An imaging
algorithm also requires the adjoint denoted
:math:`R^\dagger` which is simply the
complex conjugate transpose of :math:`R`
(note that this implies there is a similar
factor of :math:`\frac{1}{n}` involved in
the definition of :math:`R^\dagger`). The
dirty image is obtained by applying the
adjoint operator to the visibilities

.. math::

    I^D = R^\dagger V

Note that, since our definition of
:math:`R^\dagger` contains the factor of
:math:`\frac{1}{n}`, this notion of the
dirty image differs from that usually
encountered in radio astronomy but is
required to ensure that the operator
is self-adjoint.


Numpy
~~~~~

.. currentmodule:: africanus.dft

.. autosummary::
    im_to_vis
    vis_to_im

.. autofunction:: im_to_vis
.. autofunction:: vis_to_im

Dask
~~~~

.. currentmodule:: africanus.dft.dask

.. autosummary::
    im_to_vis
    vis_to_im

.. autofunction:: im_to_vis
.. autofunction:: vis_to_im

