# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from collections import namedtuple
import os
import re


def on_rtd():
    return bool(os.environ.get("READTHEDOCS"))


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
