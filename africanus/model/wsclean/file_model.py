# -*- coding: utf-8 -*-


import math
import re
import warnings

import numpy as np

hour_re = re.compile(r"(?P<sign>[+-]*)"
                     r"(?P<hours>\d+):"
                     r"(?P<mins>\d+):"
                     r"(?P<secs>\d+\.?\d*)")

deg_re = re.compile(r"(?P<sign>[+-])*"
                    r"(?P<degs>\d+)\."
                    r"(?P<mins>\d+)\."
                    r"(?P<secs>\d+\.?\d*)")


def _hour_converter(hour_str):
    m = hour_re.match(hour_str)

    if not m:
        raise ValueError("Error parsing '%s'" % hour_str)

    value = float(m.group("hours")) / 24.0
    value += float(m.group("mins")) / (24.0*60.0)
    value += float(m.group("secs")) / (24.0*60.0*60.0)

    if m.group("sign") == '-':
        value = -value

    return 2.0 * math.pi * value


def _deg_converter(deg_str):
    m = deg_re.match(deg_str)

    if not m:
        raise ValueError(f"Error parsing '{deg_str}'")

    value = float(m.group("degs")) / 360.0
    value += float(m.group("mins")) / (360.0*60.0)
    value += float(m.group("secs")) / (360.0*60.0*60.0)

    if m.group("sign") == '-':
        value = -value

    return 2.0 * math.pi * value


def arcsec2rad(arcseconds=0.0):
    return np.deg2rad(float(arcseconds) / 3600.)


def spi_converter(spi):
    spi = np.asarray([float(c) for c in spi.strip("[] ").split(",")])

    mask = np.isfinite(spi)

    if mask.any():
        warnings.warn("Non-finite spectral indices zeroed during source model parsing")
        spi[~mask] = 0.0

    return spi


def flux_converter(flux):
    flux = float(flux)

    if np.isfinite(flux):
        return flux

    warnings.warn("Non-finite flux zeroed during source model parsing")

    return 0.0


_COLUMN_CONVERTERS = {
    'Name': str,
    'Type': str,
    'Ra': _hour_converter,
    'Dec': _deg_converter,
    'I': flux_converter,
    'SpectralIndex': spi_converter,
    'LogarithmicSI': lambda x: bool(x == "true"),
    'ReferenceFrequency': float,
    'MajorAxis': arcsec2rad,
    'MinorAxis': arcsec2rad,
    'Orientation': lambda x=0.0: np.deg2rad(float(x)),
}


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
            raise ValueError(f"'{column}' is not a valid column header")

        name, default = m.group('name', 'default')

        columns.append(name)
        defaults.append(default)

    return columns, defaults


def _parse_header(header):
    format_str, col_desc = (c.strip() for c in header.split("=", 1))

    if format_str != "Format":
        raise ValueError(f"'{format_str}' does not appear to be a wsclean header")

    return _parse_col_descriptor(col_desc)


def _parse_lines(fh, line_nr, column_names, defaults, converters):
    source_data = [[] for _ in range(len(column_names))]

    for line_nr, line in enumerate(fh, line_nr):
        components = [c.strip() for c in re.split(_COMMA_SPLIT_RE, line)]

        if len(components) != len(column_names):
            raise ValueError(f"line {line_nr} '{line}' should "
                             f"have {len(column_names)} components")

        # Iterate through each column's data
        it = zip(column_names, components, converters, source_data, defaults)

        for name, comp, conv, data_list, default in it:
            if not comp:
                if default is None:
                    try:
                        default = conv()
                    except Exception as e:
                        raise ValueError(f"No value supplied for column '{name}' "
                                         f"on line {line_nr} and no default was "
                                         f"supplied either. Attempting to "
                                         f"generate a default produced the "
                                         f"following exception {e}")

                value = default
            else:
                value = comp

            data_list.append(conv(value))

    return zip(*(column_names, source_data))


def load(filename):
    """
    Loads wsclean component model.

    .. code-block:: python

        sources = load("components.txt")
        sources = dict(sources)  # Convert to dictionary

        I = sources["I"]
        ref_freq = sources["ReferenceFrequency"]

    See the `WSClean Component List
    <https://sourceforge.net/p/wsclean/wiki/ComponentList/>`_
    for further details.

    Parameters
    ----------
    filename : str or iterable
        Filename of wsclean model file or iterable
        producing the lines of the file.

    See Also
    --------
    africanus.model.wsclean.spectra

    Returns
    -------
    list of (name, list of values) tuples
        list of column (name, value) tuples
    """

    if isinstance(filename, str):
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
            raise ValueError(f"'{filename}' does not contain "
                             f"a valid wsclean header")

        column_names, defaults = _parse_header(header)

        try:
            converters = [_COLUMN_CONVERTERS[n] for n in column_names]
        except KeyError as e:
            raise ValueError("No converter registered for column {e}")

        return _parse_lines(fh, line_nr, column_names, defaults, converters)
    finally:
        if close_filename:
            fh.close()
