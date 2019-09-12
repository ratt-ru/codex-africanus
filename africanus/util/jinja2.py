# -*- coding: utf-8 -*-


from africanus.util.requirements import requires_optional


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


class TemplatingException(Exception):
    def __init__(self, msg):
        super(TemplatingException, self).__init__(msg)


def throw_helper(msg):
    raise TemplatingException(msg)


class FakeEnvironment(object):
    """
    Fake jinja2 environment, for which attribute/dict
    type access will fail
    """
    @requires_optional("jinja2")
    def __getitem__(self, key):
        raise NotImplementedError()

    @requires_optional("jinja2")
    def __setitem__(self, key, value):
        raise NotImplementedError()

    @requires_optional("jinja2")
    def __delitem__(self, key):
        raise NotImplementedError()

    @requires_optional("jinja2")
    def __getattr__(self, name):
        raise NotImplementedError()

    @requires_optional("jinja2")
    def __setattr__(self, name, value):
        raise NotImplementedError()

    @requires_optional("jinja2")
    def __delattr__(self, name):
        raise NotImplementedError()


def _jinja2_env_factory():
    try:
        from jinja2 import Environment, PackageLoader, select_autoescape
    except ImportError:
        return FakeEnvironment()

    loader = PackageLoader('africanus', '.')
    autoescape = select_autoescape(['j2', 'cu.j2'])
    env = Environment(loader=loader,
                      autoescape=autoescape,
                      extensions=['jinja2.ext.do'])

    # TODO(sjperkins)
    # Find a better way to set globals
    # perhaps search the package tree for e.g.
    # `jinja2_setup`.py files, whose contents
    # are inspected and assigned into the globals dict
    env.globals['register_assign_cycles'] = register_assign_cycles
    env.globals['throw'] = throw_helper

    return env


jinja_env = _jinja2_env_factory()
