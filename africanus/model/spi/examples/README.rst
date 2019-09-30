Simple SPI Fitter
=================

Fits a simple spectral index model to image cubes. Usage is as follows

.. code-block:: bash

    $ ./simple_spi_fitter.py --fitsmodel=/path/to/model.fits 

Run

.. code-block:: bash

    $ ./simple_spi_fitter.py -h

for documentation of the various options. In principle the model file is the
only compulsary input if the beam parameters are specified.
If they are not supplied the residual image cube needs to be provided as input
so that these can be taken from the header. This means you either have to
specify the beam parameters manually or pass in a residual cube with a header
which contains beam parameters. 

The residual is also used to determine the weights in the different imaging
bands. The weights will be set as 1/rms**2 in each imaging band, given that
the rms is not 0 in which case the weight is also set to zero for that band.

The threshold above which to fit components is set as a multiple of the rms
in the residual, where the multiple is specified by the --threshold parameter.
If the residual is not provided, then this threshold can be specified through
the --maxDR parameter. In this case only components above model.max()/maxDR
will be fit.

It is also possible to perform an image space correction for the primary beam
pattern. Currently only real and imaginary fits beams are supported.
Please see the documentation for the --beammodel parameter for the required
format.

For parallel FFT's you will also need to install pypocketfft which can be
found at git+https://gitlab.mpcdf.mpg.de/mtr/pypocketfft. Install via

.. code-block:: bash

    $ pip3 install git+https://gitlab.mpcdf.mpg.de/mtr/pypocketfft