 #!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from collections import namedtuple

import numpy as np

try:
    import dask
    import dask.array as da
    from dask.diagnostics import ProgressBar
    import Tigger
    import xarray as xr
    from xarrayms import xds_from_ms, xds_from_table, xds_to_table
except ImportError as e:
    opt_import_error = e
else:
    opt_import_error = None

from africanus.coordinates.dask import radec_to_lm
from africanus.rime.dask import phase_delay, predict_vis
from africanus.model.coherency.dask import convert
from africanus.model.spectral.dask import spectral_model
from africanus.model.shape.dask import gaussian as gaussian_shape
from africanus.model.shape.dask import shapelet as shapelet_fn
from africanus.util.requirements import requires_optional

# Testing stuff
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def as_csv(arr):
    return ','.join([str(x) for x in arr])

sky_model = Tigger.load("sky-model-shapelets.txt")
r_sky_model = open("random-sky-model.txt", "w+")
r_sky_model.write("#format: name ra_d dec_d i spi freq0 sbetal_s sbetam_s shapelet_coeffs")
print(sky_model.sources[0].typecode)
sbeta1 = sky_model.sources[0].shape.sbetal
sbeta2 = sky_model.sources[0].shape.sbetam
ra = np.rad2deg(sky_model.sources[0].pos.ra)
dec = np.rad2deg(sky_model.sources[0].pos.dec)
print("RA, DEC IS ", ra, dec)
I = sky_model.sources[0].flux.I * 0.0001
spi = sky_model.sources[0].spectrum.spi
freq0 = sky_model.sources[0].spectrum.freq0
print(I)
for i in range(128):
    # Calculate size
    b1 = sbeta1 + (0.5 * sbeta1) * (np.random.random() - 0.5)
    b2 = sbeta2 + (0.5 * sbeta2) * (np.random.random() - 0.5)

    # Generate random coefficients
    ncoeffs = 8
    coeffs_lm = 10 * np.random.random(ncoeffs)
        
    # Random position
    r = ra + (6 * (np.random.uniform(0,1) - 0.5))
    d = dec + (6 * (np.random.uniform(0,1) - 0.5))
    print(r, d)
    # print(as_csv(coeffs_lm.flatten()))

    r_sky_model.write("\nJ%d %f %f %f %f %f %f %f %s" %(i, r, d, I, spi, freq0, b1, b2, as_csv(coeffs_lm)))   
    