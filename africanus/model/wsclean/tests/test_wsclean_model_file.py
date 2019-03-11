# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from africanus.model.wsclean.file_model import load


def test_wsclean_model_file(wsclean_model_file):
    sources = dict(load(wsclean_model_file))

    name, stype, I, spi, log_si, ref_freq = (sources[n] for n in (
                                            "Name", "Type", "I",
                                            "SpectralIndex", "LogarithmicSI",
                                            "ReferenceFrequency"))

    # Seven sources
    assert (len(I) == len(spi) == len(log_si) == len(ref_freq) == 7)
    # Missing reference frequency set in the last
    assert (ref_freq[-1] == ref_freq[0] and
            name[-1] == "s1c2" and
            stype[-1] == "GAUSSIAN")
