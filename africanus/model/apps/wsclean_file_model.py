# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ast import literal_eval
import math
import re

from africanus.compatibility import string_types

import numba
import numpy as np


hour_re = re.compile(r"(?P<sign>[-]*)"
                     r"(?P<hours>\d+):"
                     r"(?P<mins>\d+):"
                     r"(?P<secs>\d+\.\d+)")

deg_re = re.compile(r"(?P<sign>[-])*"
                    r"(?P<degs>\d+)\."
                    r"(?P<mins>\d+)\."
                    r"(?P<secs>\d+\.\d+)")


def _hour_converter(hour_str):
    m = hour_re.match(hour_str)

    if not m:
        raise ValueError("Error parsing '%s'" % hour_str)

    if m.group("sign") == '-':
        value = float(m.group("hours")) / 24.0
        value += float(m.group("mins")) / (24.0*60.0)
        value += float(m.group("secs")) / (24.0*60.0*60.0)
    else:
        value = float(m.group("hours")) / 24.0
        value -= float(m.group("mins")) / (24.0*60.0)
        value -= float(m.group("secs")) / (24.0*60.0*60.0)

    return 2.0 * math.pi * value


def _deg_converter(deg_str):
    m = deg_re.match(deg_str)

    if not m:
        raise ValueError("Error parsing '%s'" % deg_str)

    if m.group("sign") == '-':
        value = float(m.group("degs")) / 360.0
        value += float(m.group("mins")) / (360.0*60.0)
        value += float(m.group("secs")) / (360.0*60.0*60.0)
    else:
        value = float(m.group("degs")) / 360.0
        value -= float(m.group("mins")) / (360.0*60.0)
        value -= float(m.group("secs")) / (360.0*60.0*60.0)

    return 2.0 * math.pi * value


_COLUMN_CONVERTERS = [
    ('Name', str),
    ('Type', str),
    ('Ra', _hour_converter),
    ('Dec', _deg_converter),
    ('I', float),
    ('SpectralIndex', literal_eval),
    ('LogarithmicSI', lambda x: bool(x == "true")),
    ('ReferenceFrequency', float),
    ('MajorAxis', float),
    ('MinorAxis', float),
    ('Orientation', float),
]


# Split on commas, ignoring within [] brackets
_COMMA_SPLIT_RE = re.compile(r',\s*(?=[^\]]*(?:\[|$))')

# Parse columm headers, handling possible defaults
_COL_HEADER_RE = re.compile(r"^\s*?(?P<name>.*?)"
                            r"(\s*?=\s*?'(?P<default>.*?)'\s*?){0,1}$")


def _parse_col_descriptor(column_descriptor):
    components = [c.strip() for c in column_descriptor.split(",")]

    columns = []
    defaults = []

    for column in components:
        m = _COL_HEADER_RE.search(column)

        if m is None:
            raise ValueError("'%s' is not a valid column header" % column)

        name, default = m.group('name', 'default')

        columns.append(name)
        defaults.append(default)

    return columns, defaults


def _parse_header(header):
    format_str, col_desc = (c.strip() for c in header.split("=", 1))

    if format_str != "Format":
        raise ValueError("'%s' does not appear to be a wsclean header")

    return _parse_col_descriptor(col_desc)


def _parse_lines(fh, line_nr, column_names, defaults, converters):
    source_data = [[] for _ in range(len(column_names))]

    for line_nr, line in enumerate(fh, line_nr):
        components = [c.strip() for c in re.split(_COMMA_SPLIT_RE, line)]

        if len(components) != len(column_names):
            raise ValueError("line %d '%s' should have %d components" %
                             (line_nr, line, len(column_names)))

        # Iterate through each column's data
        it = zip(column_names, components, converters, source_data, defaults)

        for name, comp, conv, data_list, default in it:
            if not comp:
                if default is None:
                    try:
                        default = conv()
                    except Exception as e:
                        raise ValueError("No value supplied for column '%s' "
                                         "on line %d and no default was "
                                         "supplied either. Attempting to "
                                         "generate a default produced the "
                                         "following exception %s"
                                         % (name, line_nr, str(e)))

                value = default
            else:
                value = comp

            data_list.append(conv(value))

    return source_data


def wsclean(filename):
    """
    Parameters
    ----------
    filename : str or iterable
        Filename of wsclean model file or iterable
        producing the lines of the file.
    """

    if isinstance(filename, string_types):
        fh = open(filename, "r")
        fh = iter(fh)
        close_filename = True
    else:
        fh = iter(filename)
        close_filename = False

    try:
        # Search for a header until we find a non-empty string
        header = ''
        line_nr = 1

        for headers in fh:
            header = headers.split("#", 1)[0].strip()

            if header:
                break

            line_nr += 1

        if not header:
            raise ValueError("'%s' does not contain a valid wsclean header"
                             % filename)

        column_names, defaults = _parse_header(header)
        conv_names, converters = zip(*_COLUMN_CONVERTERS)

        if not conv_names == tuple(column_names):
            raise ValueError("header columns '%s' do not match expected '%s'"
                             % (conv_names, column_names))

        return _parse_lines(fh, line_nr, column_names, defaults, converters)
    finally:
        if close_filename:
            fh.close()


def log_si_factory(log_si_type):
    if log_si_type == "array":
        def impl(log_si, src):
            return log_si[src]
    elif log_si_type == "bool":
        def impl(log_si, src):
            return log_si
    else:
        raise ValueError("log_is_type not in ('array', 'bool')")

    return numba.njit(nogil=True, cache=True)(impl)


def ref_freq_factory(ref_freq_type):
    if ref_freq_type == "array":
        def impl(ref_freq, src):
            return ref_freq[src]
    elif ref_freq_type == "float":
        def impl(ref_freq, src):
            return ref_freq
    else:
        raise ValueError("ref_freq_type not in ('array', 'float')")

    return numba.njit(nogil=True, cache=True)(impl)


@numba.generated_jit(nopython=True, nogil=True, cache=True)
def spectral_model(I, spi, log_si, ref_freq, frequency):
    """
    Produces a spectral model as defined by the polynomial
    expansion in wsclean's
    `Component Model
    <https://sourceforge.net/p/wsclean/wiki/ComponentList/_>`_.


    Parameters
    ----------
    I : :class:`numpy.ndarray`
        flux density of shape :code:`(source,)`
    spi : :class:`numpy.ndarray`
        spectral index of shape :code:`(source, spi_comps)`
    log_si : :class:`numpy.ndarray` or bool
        boolean array of shape :code:`(source,)` indicating
        whether logarithmic (True) or ordinary (False)
        polynomials should be used.
    ref_freq : :class:`numpy.ndarray`
        Source reference frequencies of shape :code:`(source,)`
    frequency : :class:`numpy.ndarray`
        frequencies of shape :code:`(chan,)`

    Returns
    -------
    spectral_model : :class:`numpy.ndarray`
        Spectral Model of shape :code:`(source, chan)`
    """
    arg_dtypes = tuple(np.dtype(I.dtype.name) for a
                       in (I, spi, ref_freq, frequency))
    dtype = np.result_type(*arg_dtypes)

    if isinstance(log_si, numba.types.npytypes.Array):
        log_si_fn = log_si_factory("array")
    elif isinstance(log_si, numba.types.scalars.Boolean):
        log_si_fn = log_si_factory("bool")
    else:
        raise ValueError("log_si must be an ndarray or scalar bool")

    if isinstance(ref_freq, numba.types.npytypes.Array):
        ref_freq_fn = ref_freq_factory("array")
    elif isinstance(ref_freq, numba.types.scalars.Float):
        ref_freq_fn = ref_freq_factory("float")
    else:
        raise ValueError("ref_freq must be an ndarray or scalar float")

    def impl(I, spi, log_si, ref_freq, frequency):
        if not (I.shape[0] == spi.shape[0] == ref_freq.shape[0]):
            raise ValueError("first dimensions of I, spi, "
                             "log_si and ref_freq don't match.")

        nsrc = I.shape[0]
        nchan = frequency.shape[0]
        nspi = spi.shape[1]

        spectral_model = np.empty((nsrc, nchan), dtype=dtype)

        for s in range(nsrc):
            rf = ref_freq_fn(ref_freq, s)

            # Logarithmic polynomial
            if log_si_fn(log_si, s):
                base = np.log(I[s])

                for f,  nu in enumerate(frequency):
                    spectral_model[s, f] = base

                    for si in range(nspi):
                        term = spi[s, si]
                        spectral_model[s, f] += term * np.log(nu/rf)**(si + 1)

                    spectral_model[s, f] = np.exp(spectral_model[s, f])
            # Ordinary polynomial
            else:
                base = I[s]

                for f, nu in enumerate(frequency):
                    spectral_model[s, f] = base

                    for si in range(nspi):
                        term = spi[s, si]
                        spectral_model[s, f] += term * (nu/rf - 1)**(si + 1)

        return spectral_model

    return impl
