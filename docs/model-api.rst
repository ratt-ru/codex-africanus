---------
Sky Model
---------

Functionality related to the Sky Model.

Coherency Conversion
--------------------

Utilities for converting back and forth between
stokes parameters and correlations

Numpy
~~~~~

.. currentmodule:: africanus.model.coherency

.. autosummary::
    convert

.. autofunction:: convert

Cuda
~~~~

.. currentmodule:: africanus.model.coherency.cuda

.. autosummary::
    convert

.. autofunction:: convert

Dask
~~~~

.. currentmodule:: africanus.model.coherency.dask

.. autosummary::
    convert

.. autofunction:: convert


Spectral Model
--------------

Functionality for computing a Spectral Model.


Numpy
~~~~~

.. currentmodule:: africanus.model.spectral

.. autosummary::
    spectral_model

.. autofunction:: spectral_model

Dask
~~~~

.. currentmodule:: africanus.model.spectral.dask

.. autosummary::
    spectral_model

.. autofunction:: spectral_model


Spectral Index
--------------

Functionality related to the spectral index.

For example, we may want
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


Source Morphology
-----------------

Shape functions for different Source Morphologies

Numpy
~~~~~

.. currentmodule:: africanus.model.shape

.. autosummary::
    gaussian

.. autofunction:: gaussian

Dask
~~~~

.. currentmodule:: africanus.model.shape.dask

.. autosummary::
    gaussian

.. autofunction:: gaussian


WSClean Spectral Model
----------------------

Utilities for creating a spectral model from a wsclean component file.

Numpy
~~~~~

.. currentmodule:: africanus.model.wsclean

.. autosummary::
    load
    spectra

.. autofunction:: load
.. autofunction:: spectra

Dask
~~~~

.. currentmodule:: africanus.model.wsclean.dask

.. autosummary::
    spectra

.. autofunction:: spectra
