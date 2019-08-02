Simple SPI Fitter
=================

Fits a simple spectral index model to image cubes. Usage is as follows

$ ./simple_spi_fitter.py --fitsmodel=/path/to/model.fits --fitsresidual=path/to/residual.fits ...

Run

$ ./simple_spi_fitter.py -h

for documentation of the various options. In principle the model file is the
only compulsary input if the beam parameters are specified through the emaj,
emin and pa options. If these are not specified the restored image needs to
be provided as input so that these can be taken from the header (required
since DDF only stores the beam parameters in the header of the restored
image). The residual is not strictly speaking required but it helps to
approximate the noise in image space. If it is given it will be added to
the convolved image. It is also required to determine the
threshold above which to fit components, determined as a multiple of the
rms in the residual where the multiple is specified as --threshold.
If the residual is not provided (as might be the case if the image has not
been fully deconvolved) then this threshold can be specified through the
--maxDR parameter. In this case only components above model.max()/maxDR
will be fit. 
