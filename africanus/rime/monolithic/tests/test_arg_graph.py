from abc import abstractmethod, ABC

import pytest


def test_numba_intrinsic():
    from numba.extending import intrinsic
    from numba.core import cgutils, types
    from numba import njit

    defaults = (1, 2.0, "pants")

    @intrinsic
    def optional_intrinsic(typingctx, args):
        assert isinstance(args, types.Tuple)
        default_types = tuple(map(typingctx.resolve_value_type, defaults))
        return_type = types.Tuple(args.types + default_types)
        sig = return_type(args)

        def codegen(context, builder, signature, args):
            llvm_ret_type = context.get_value_type(signature.return_type)
            ret_tuple = cgutils.get_null_value(llvm_ret_type)

            for i in range(len(signature.args[0])):
                data = builder.extract_value(args[0], i)
                context.nrt.incref(builder, signature.args[0][i], data)
                ret_tuple = builder.insert_value(ret_tuple, data, i)

            for d, (typ, default) in enumerate(zip(default_types, defaults)):
                const = context.get_constant_generic(builder, typ, default)
                ret_tuple = builder.insert_value(ret_tuple, const, len(signature.args[0]) + d)

            return ret_tuple

        return sig, codegen

    @njit
    def test2(*args):
        return optional_intrinsic(args)

    print(test2(1, "bob", 2.0))


def test_arg_graph():
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


