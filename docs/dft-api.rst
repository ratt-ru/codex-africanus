------------------------
Direct Fourier Transform
------------------------

Functions used to compute the discretised
direct Fourier transform (DFT)
for an ideal interferometer.
The DFT for an ideal interferometer is
defined as

.. math::

    V(u,v,w) = \int B(l,m) e^{-2\pi i
        \left( ul + vm + w(n-1)\right)}
        \frac{dl dm}{n}

where :math:`u,v,w` are data space coordinates and
where visibilities :math:`V` have been obtained.
The :math:`l,m,n` are signal space coordinates at which
we wish to reconstruct the signal :math:`B`. Note that
the signal correspondes to the brightness matrix
and not the Stokes parameters. We adopt the convention
where we absorb the fixed coordinate :math:`n` in the
denominator into the image.
Note that the data space coordinates have an implicit
dependence on frequency and time and that the image
has an implicit dependence on frequency.
The discretised form of the DFT can be written as

.. math::

    V(u,v,w) = \sum_s e^{-2 \pi i
        (u l_s + v m_s + w (n_s - 1))} \cdot B_s

where :math:`s` labels the source (or pixel) location.
If only a single correlation is present :math:`B = I`,
this can be cast into a matrix equation as follows

.. math::

    V = R I

where :math:`R` is the operator that maps an
image to visibility space. This mapping is
implemented by the :func:`~africanus.dft.im_to_vis`
function. If multiple correlations are present then
each one is mapped to its corresponding visibility.
An imaging algorithm also requires the adjoint
denoted :math:`R^\dagger` which is simply the
complex conjugate transpose of :math:`R`.
The dirty image is obtained by applying the
adjoint operator to the visibilities

.. math::

    I^D = R^\dagger V

This is implemented by the
:func:`~africanus.dft.vis_to_im`
function.
Note that an imaging algorithm using these
operators will actually reconstruct
:math:`\frac{I}{n}` but that it is trivial
to obtain :math:`I` since :math:`n` is
known at each location in the image.


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

