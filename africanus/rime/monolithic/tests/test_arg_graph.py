import pytest


def test_arg_graph():
    from abc import abstractmethod, ABC

    class Node(ABC):
        @property
        @abstractmethod
        def outputs(self):
            raise NotImplementedError

    class Argument(Node):
        def __init__(self, name, outputs=None):
            self.name = name
            self._outputs = set(outputs) if outputs else set()

        def add_output(self, output):
            self._outputs.add(output)

        def __repr__(self):
            return self.name

        __str__ = __repr__

        @property
        def outputs(self):
            return self._outputs

    class Term(Node):
        @property
        def outputs(self):
            return ()

        def __repr__(self):
            return self.__class__.__name__

        __str__ = __repr__

    class BrightnessTerm(Term):
        inputs = ["chan_freq", "stokes"]

    class PhaseTerm(Term):
        inputs = ["lm", "uvw", "chan_freq"]

    class Transformer(Node):
        def __init__(self, name, inputs, outputs):
            self.name = name
            self.inputs = set(inputs)
            self._outputs = set(Argument(o) for o in outputs)

        def add_outputs(self, output):
            self._outputs.add(output)

        @property
        def outputs(self):
            return self._outputs

        def __repr__(self):
            return self.name

        __str__ = __repr__

    terms = [BrightnessTerm(), PhaseTerm()]
    args = {"phase_centre", "radec", "uvw",
            "chan_freq", "stokes", "spi"}
    variables = {n: Argument(n) for n in args}

    xformer_args = [
        ["RadecTransformer", ["phase_centre", "radec"], ["lm"]],
    ]
    xformers = [Transformer(n, i, o) for n, i, o in xformer_args]
    xformers = {o.name: t for t in xformers for o in t.outputs}

    for term in terms:
        for a in term.inputs:
            try:
                # Try obtain the argument from existing variables
                arg = variables[a]
            except KeyError:
                # Doesn't exist, attempt to find a transformer that
                # produces it
                try:
                    xformer = xformers[a]
                except KeyError:
                    raise ValueError(f"'{a}' not in {list(variables.keys())} "
                                     f"and no transformers exist"
                                     f"to create it")

                # The transformer inputs must be present in variables
                for input in xformer.inputs:
                    try:
                        arg = variables[input]
                    except KeyError:
                        raise ValueError(f"{xformer} needs '{input}' to "
                                         f"produce '{a}' but '{input}' is not "
                                         f"present in the supplied "
                                         f"arguments")
                    else:
                        # Add the transformer as the argument's neighbour
                        arg.add_output(xformer)

                output = next(o for o in xformer.outputs if o.name == a)
                output.add_output(term)
            else:
                arg.add_output(term)

    def toposort(graph):
        stack = []
        visited = set()

        def _recurse(node):
            if node in visited:
                return

            visited.add(node)
            has_outputs = False

            for o in node.outputs:
                _recurse(o)
                has_outputs = True

            # Don't visit argument's with no outputs
            if isinstance(node, Argument) and not has_outputs:
                return

            stack.append(node)

        for arg in graph:
            _recurse(arg)

        return stack[::-1]

    order = toposort(variables.values())
    print(order)


@pytest.mark.xfail
def test_arg_graph_2():
    from africanus.rime.monolithic.terms.phase import PhaseTerm
    from africanus.rime.monolithic.terms.brightness import BrightnessTerm
    from africanus.rime.monolithic.transformers.core import LMTransformer

    terms = [PhaseTerm(), BrightnessTerm()]
    xformers = [LMTransformer()]

    class NO_DEFAULT:
        def __repr__(self):
            return "<no default>"

        __str__ = __repr__

    MISSING = NO_DEFAULT()
    desired = {}

    for term in terms:
        for a in term.ARGS:
            desired[a] = MISSING

        for k, d in term.KWARGS.items():
            desired[k] = d

    for transformer in xformers:
        for output in transformer.OUTPUTS:
            if output in desired:
                continue

    print(desired)

    supplied_args = set(("radec", "uvw",
                         "chan_freq", "stokes"))

    for arg, default in desired.items():
        if arg in supplied_args:
            continue

        missing = False

        for transformer in xformers:
            if missing:
                break

            if arg in transformer.OUTPUTS:
                for a in transformer.ARGS:
                    if a not in supplied_args:
                        missing = True
                        break

        if default is MISSING:
            missing

        if missing:
            raise ValueError(f"{arg} not supplied or unable "
                             f"to create it from supplied args")


@pytest.mark.xfail
def test_arg_graph_3():
    import numba
    from numba import types
    import numpy as np
    from collections import defaultdict
    
    from africanus.rime.monolithic.terms.phase import PhaseTerm
    from africanus.rime.monolithic.transformers.lm import LMTransformer

    class rime_factory:
        def __init__(self):
            terms = [PhaseTerm()]
            transformers = [LMTransformer()]
            desired = defaultdict(list)
            optional = defaultdict(list)
            can_create = defaultdict(list)

            for t in terms:
                for a in t.ARGS:
                    desired[a].append(t)
                for k, d in t.KWARGS.items():
                    optional[k].append((t, d))

            for t in transformers:
                for o in t.OUTPUTS:
                    can_create[o].append(t)

            @numba.generated_jit
            def fn(arg_names, *args):
                assert len(arg_names) == len(args[0])
                assert all(isinstance(n, types.Literal) for n in arg_names)
                assert all(n.literal_type is types.unicode_type for n in arg_names)
                names = tuple(n.literal_value for n in arg_names)
                missing = set(desired.keys()) - set(names)
                attempts = {}

                # Try create missing argument with transformers
                for arg in list(missing):
                    try:
                        transformers = can_create[arg]
                    except KeyError:
                        continue

                    for transformer in transformers:
                        # We didn't have the arguments, make a note of this
                        if not set(transformer.ARGS).issubset(set(names)):
                            attempts[arg] = (transformer, list(transformer.ARGS))
                            continue

                    if arg in attempts:
                        continue

                    missing.remove(arg)

                for arg in missing:
                    terms_wanting = desired[arg]
                    err_msgs = []
                    err_msgs.append(f"The following terms want {arg}: {terms_wanting}.")

                    try:
                        transformer, transformer_args = attempts[arg]
                    except KeyError:
                        pass
                    else:
                        needed = set(transformer_args)

                        err_msgs.append(f"{transformer} can create {arg} "
                                        f"but needs {needed}, of which "
                                        f"{needed - set(names)} is missing "
                                        f"from the input arguments.")

                    raise ValueError("\n".join(err_msgs))


                def impl(arg_names, *args):
                    print(arg_names)
                    print(args)

                return impl

            self.impl = fn


        def __call__(self, **kwargs):
            keys = tuple(types.literal(k) for k in kwargs.keys())
            return self.impl(keys, *kwargs.values())

    factory = rime_factory()
    factory(radec=1, uvw=np.zeros((10, 3)), chan_freq="hi")