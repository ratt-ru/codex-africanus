# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ast import literal_eval
import math
import re

from africanus.compatibility import string_types

hour_re = re.compile(r"(?P<sign>[-]*)"
                     "(?P<hours>\d+):"
                     "(?P<mins>\d+):"
                     "(?P<secs>\d+\.\d+)")

deg_re = re.compile(r"(?P<sign>[-])*"
                    "(?P<degs>\d+)\."
                    "(?P<mins>\d+)\."
                    "(?P<secs>\d+\.\d+)")


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


# Split on commas, ignoring within []
_COMMA_SPLIT_RE = re.compile(r',\s*(?=[^\]]*(?:\[|$))')


def _parse_col_descriptor(column_descriptor):
    components = [c.strip() for c in column_descriptor.split(",")]

    ref_freq = None
    columns = []

    for column in components:
        if column.startswith('ReferenceFrequency'):
            columns.append("ReferenceFrequency")
            str_value = column.split("=")[-1]
            str_value = str_value[1:-1]

            try:
                ref_freq = float(str_value)
            except ValueError as e:
                raise ValueError("Unable to extract reference frequency "
                                 "'%s' from '%s': %s\n",
                                 str_value, column, e)
        else:
            columns.append(column)

    return columns, ref_freq


def _parse_lines(fh, line_nr, column_names, converters):
    point_data = [[] for _ in range(len(column_names) - 3)]
    gauss_data = [[] for _ in range(len(column_names))]

    for line_nr, line in enumerate(fh, line_nr):
        components = [c.strip() for c in re.split(_COMMA_SPLIT_RE, line)]

        if len(components) != len(column_names):
            raise ValueError("line %d '%s' should have %d components" %
                             (line_nr, line, len(column_names)))

        source_type = components[1].upper()

        if source_type == "POINT":
            it = zip(components[:-3], converters[:-3], point_data)
            for comp, conv, data_list in it:
                data_list.append(conv(comp))
        elif source_type == "GAUSSIAN":
            it = zip(components, converters, gauss_data)
            for comp, conv, data_list in it:
                data_list.append(conv(comp))
        else:
            raise ValueError("type '%s' not in (POINT, GAUSSIAN)"
                             % source_type)

    return point_data, gauss_data


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

        format_str, col_desc = (c.strip() for c in header.split("=", 1))

        if format_str != "Format":
            raise ValueError("'%s' does not appear to be a wsclean header")

        column_names, ref_freq = _parse_col_descriptor(col_desc)
        conv_names, converters = zip(*_COLUMN_CONVERTERS)

        if not conv_names == tuple(column_names):
            raise ValueError("header columns '%s' do not match expected '%s'"
                             % (conv_names, column_names))

        return _parse_lines(fh, line_nr, column_names, converters)
    finally:
        if close_filename:
            fh.close()
