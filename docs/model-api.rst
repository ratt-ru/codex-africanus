Model
------------------------

Functions used to compute sky model
attributes. For example, we may want
to compute the spectral indices of
components in a sky model defined by

.. math::

    I(\nu) = I(\nu_0) \left(\frac{\nu}{\nu_0}\right)^\alpha

where :math:`\nu` are frequencies ay
which we want to construct the intensity
of a Stokes I image and the :math:`\nu_0`
is the corresponding reference frequency.
The spectral index :math:`\alpha`
determines how quickly the intensity grows
or decays as a function of frequency.
Given a list of model image components
(preferably with the residuals added back
in) we can recover the corresponding
spectral indices and reference intensities
using the :func:`~africanus.model.spi.fit_spi_components`
function. This will also return a lower bound
on the associated uncertainties on these
components.

Numpy
~~~~~

.. currentmodule:: africanus.model.spi

.. autosummary::
    fit_spi_components

.. autofunction:: fit_spi_components


Dask
~~~~~

.. currentmodule:: africanus.model.spi.dask

.. autosummary::
    fit_spi_components

.. autofunction:: fit_spi_components
