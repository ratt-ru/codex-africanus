# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join as pjoin

import numpy as np


from africanus.model.apps.wsclean_file_model import wsclean, spectral_model

_WSCLEAN_MODEL_FILE = ("""
Format = Name, Type, Ra, Dec, I, SpectralIndex, LogarithmicSI, ReferenceFrequency='125584411.621094', MajorAxis, MinorAxis, Orientation
s0c0,POINT,-08:28:05.152,39.35.08.511,0.000748810650400475,[-0.00695379313004673,-0.0849693907803257],false,125584411.621094,,,
s0c1,POINT,08:22:27.658,39.37.38.353,-0.000154968071120503,[-0.000898135869319762,0.0183710297781511],false,125584411.621094,,,
s0c2,POINT,08:18:44.309,39.38.37.773,0.000233552686127518,[-0.000869089801859608,0.0828587947079702],false,125584411.621094,,,
s0c3,POINT,08:03:07.538,39.37.02.717,0.000919058240247659,[0.001264109956439,0.0201438425344451],false,125584411.621094,,,
s1c0,GAUSSIAN,08:31:10.37,41.47.17.131,0.000723326710524984,[0.00344317919656096,-0.115990377833407],false,125584411.621094,83.6144111272856,83.6144111272856,0
s1c1,GAUSSIAN,07:51:09.24,42.32.46.177,0.000660490865128381,[0.00404869217508666,-0.011844732049232],false,125584411.621094,83.6144111272856,83.6144111272856,0
""")  # noqa


def test_wsclean_model_file(tmpdir):
    filename = pjoin(str(tmpdir), "model.txt")

    with open(filename, "w") as f:
        f.write(_WSCLEAN_MODEL_FILE)

    point, gaussian = wsclean(filename)

    _, _, _, _, I, spi, log_si, ref_freq = point
    freq = np.linspace(.856e9, .856e9*2, 16)

    I = np.asarray(I)  # noqa
    spi = np.asarray(spi)
    log_si = np.asarray(log_si)
    ref_freq = np.asarray(ref_freq)

    # log_si array of bools
    model = spectral_model(I, spi, log_si, ref_freq, freq)

    # True or False log_si
    model = spectral_model(I, spi, True, ref_freq, freq)
    model = spectral_model(I, spi, False, ref_freq, freq)  # noqa
