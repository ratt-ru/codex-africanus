# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from africanus.util.requirements import requires_optional
from africanus.util.code import SingletonMixin

try:
    from jinja2 import Environment, PackageLoader, select_autoescape
except ImportError:
    class Jinja2Environment(SingletonMixin):
        @requires_optional('jinja2')
        def __init__(self):
            pass
else:
    class Jinja2Environment(Environment, SingletonMixin):
        @requires_optional('jinja2')
        def __init__(self):
            loader = PackageLoader('africanus', '.')
            autoescape = select_autoescape(['cu.j2'])
            super(Jinja2Environment, self).__init__(loader=loader,
                                                    autoescape=autoescape)
