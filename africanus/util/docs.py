# -*- coding: utf-8 -*-


import os
import re
from string import Template

_on_rtd = bool(os.environ.get("READTHEDOCS"))


def on_rtd():
    return _on_rtd


def mod_docs(docstring, replacements):
    for search, replace in replacements:
        docstring = docstring.replace(search, replace)

    return docstring


def doc_tuple_to_str(doc_tuple, replacements=None):
    fields = getattr(doc_tuple, "_fields", None)

    if fields is not None:
        fields = (getattr(doc_tuple, f) for f in doc_tuple._fields)
    elif isinstance(doc_tuple, dict):
        fields = doc_tuple.values()

    if replacements is not None:
        fields = (mod_docs(f, replacements) for f in fields)

    return ''.join(fields)


class DefaultOut(object):
    def __init__(self, arg):
        self.arg = arg

    def __repr__(self):
        return self.arg

    __str__ = __repr__


class DocstringTemplate(Template):
    """
    Overrides the ${identifer} braced pattern in the string Template
    with a $(identifier) braced pattern
    """
    pattern = r"""
    %(delim)s(?:
      (?P<escaped>%(delim)s)   |   # Escape sequence of two delimiters
      (?P<named>%(id)s)        |   # delimiter and a Python identifier
      \((?P<braced>%(id)s)\)   |   # delimiter and a braced identifier
      (?P<invalid>)                # Other ill-formed delimiter exprs
    )
    """ % {'delim': re.escape(Template.delimiter),
           'id': Template.idpattern}
