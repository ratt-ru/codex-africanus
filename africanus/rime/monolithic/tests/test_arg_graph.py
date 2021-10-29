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


