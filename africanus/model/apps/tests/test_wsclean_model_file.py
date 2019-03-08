# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join as pjoin

import numpy as np
from numpy.testing import assert_array_almost_equal


from africanus.model.apps.wsclean_file_model import wsclean, spectral_model

_WSCLEAN_MODEL_FILE = ("""
Format = Name, Type, Ra, Dec, I, SpectralIndex, LogarithmicSI, ReferenceFrequency='125584411.621094', MajorAxis, MinorAxis, Orientation
s0c0,POINT,-08:28:05.152,39.35.08.511,0.000748810650400475,[-0.00695379313004673,-0.0849693907803257],true,125584411.621094,,,
s0c1,POINT,08:22:27.658,39.37.38.353,-0.000154968071120503,[-0.000898135869319762,0.0183710297781511],false,125584411.621094,,,
s0c2,POINT,08:18:44.309,39.38.37.773,0.000233552686127518,[-0.000869089801859608,0.0828587947079702],false,125584411.621094,,,
s0c3,POINT,08:03:07.538,39.37.02.717,0.000919058240247659,[0.001264109956439,0.0201438425344451],false,125584411.621094,,,
s1c0,GAUSSIAN,08:31:10.37,41.47.17.131,0.000723326710524984,[0.00344317919656096,-0.115990377833407],true,125584411.621094,83.6144111272856,83.6144111272856,0
s1c1,GAUSSIAN,07:51:09.24,42.32.46.177,0.000660490865128381,[0.00404869217508666,-0.011844732049232],false,125584411.621094,83.6144111272856,83.6144111272856,0
s1c2,GAUSSIAN,07:51:09.24,42.32.46.177,0.000660490865128381,[0.00404869217508666,-0.011844732049232],false,,83.6144111272856,83.6144111272856,0
""")  # noqa


def ordinary_spectral_model(I, spi, log_si, freq, ref_freq):
    spi_idx = np.arange(1, spi.shape[1] + 1)
    # (source, chan, spi-comp)
    term = (freq[None, :, None] / ref_freq[:, None, None]) - 1.0
    term = term**spi_idx[None, None, :]
    term = spi[:, None, :]*term
    return I[:, None] + term.sum(axis=2)


def log_spectral_model(I, spi, log_si, freq, ref_freq):
    # No negative flux
    I = np.where(log_si == False, 1.0, I)  # noqa
    spi_idx = np.arange(1, spi.shape[1] + 1)
    # (source, chan, spi-comp)
    term = np.log(freq[None, :, None] / ref_freq[:, None, None])
    term = term**spi_idx[None, None, :]
    term = spi[:, None, :]*term
    return np.exp(np.log(I)[:, None] + term.sum(axis=2))


def test_wsclean_model_file(tmpdir):
    filename = pjoin(str(tmpdir), "model.txt")

    with open(filename, "w") as f:
        f.write(_WSCLEAN_MODEL_FILE)

    sources = wsclean(filename)

    name, stype, _, _, I, spi, log_si, ref_freq, _, _, _ = sources
    freq = np.linspace(.856e9, .856e9*2, 16)

    I = np.asarray(I)  # noqa
    spi = np.asarray(spi)
    log_si = np.asarray(log_si)
    ref_freq = np.asarray(ref_freq)

    # Compute spectral model with numpy implementations
    ordinary_spec_model = ordinary_spectral_model(I, spi, log_si,
                                                  freq, ref_freq)
    log_spec_model = log_spectral_model(I, spi, log_si,
                                        freq, ref_freq)
    spec_model = np.where(log_si[:, None] == True,  # noqa
                          log_spec_model,
                          ordinary_spec_model)

    model = spectral_model(I, spi, log_si, ref_freq, freq)

    # Reduce over spi-comp and add the base flux
    assert_array_almost_equal(model, spec_model)

    # True or False log_si
    model = spectral_model(I, spi, True, ref_freq, freq)
    model = spectral_model(I, spi, False, ref_freq, freq)  # noqa

    # Seven sources
    assert (I.shape[0] == spi.shape[0] ==
            log_si.shape[0] == ref_freq.shape[0] == 7)
    # Missing reference frequency set in the last
    assert (ref_freq[-1] == ref_freq[0] and
            name[-1] == "s1c2" and
            stype[-1] == "GAUSSIAN")
