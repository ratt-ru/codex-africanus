# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from africanus.util.requirements import requires_optional
from africanus.util.code import SingletonMixin

try:
    from jinja2 import Environment, PackageLoader, select_autoescape
except ImportError:
    class jinja_env(SingletonMixin):
        @requires_optional('jinja2')
        def __init__(self):
            pass
else:
    def register_assign_cycles(N, case=0):
        """
        Determine cycles that stem from performing
        an in-place assignment of the elements of an array.
        In the following. we cannot naively assign the source index
        to the dest index,
        If we assigned the value at index 0 to index 1, the assignment
        from index 1 to index 3 would be invalid, for example:

        src:  [3, 0, 2, 1]
        dest: [0, 1, 2, 3]

        assignment cycles can be broken by using a
        temporary variable to store the contents of a source index
        until dependent assignments have been performed.

        In this way, the minimum number of registers can be used
        to perform the in-place assignment.

        Returns
        -------
        list of lists of tuples
            For example, `[[(0, 2), (2, 0)], [1, 3], [3, 1]]`

        """
        dest = range(N)
        src = [(N - case + n) % N for n in dest]

        deps = {d: s for d, s in zip(dest, src) if d != s}
        saved_deps = deps.copy()

        for di, d in enumerate(dest):
            si = src.index(d)
            if si > di:
                deps[si] = di

        cycles = []

        while len(deps) > 0:
            k, v = deps.popitem()
            cycle = [(k, v)]

            while True:
                try:
                    k = v
                    v = deps.pop(k)
                except KeyError:
                    # Check that the last key we're trying
                    # to get is the first one in the cycle
                    assert k == cycle[0][0]
                    break

                cycle.append((k, v))

            cycles.append(cycle)

        return cycles

    class CupyTemplatingException(Exception):
        def __init__(self, msg):
            super(CupyTemplatingException, self).__init__(msg)

    def throw_helper(msg):
        raise CupyTemplatingException(msg)

    class jinja_env(Environment, SingletonMixin):
        @requires_optional('jinja2')
        def __init__(self):
            loader = PackageLoader('africanus', '.')
            autoescape = select_autoescape(['j2', 'cu.j2'])
            super(jinja_env, self).__init__(loader=loader,
                                            autoescape=autoescape)

            self.globals['register_assign_cycles'] = register_assign_cycles
            self.globals['throw'] = throw_helper
