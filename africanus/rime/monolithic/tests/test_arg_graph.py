def test_arg_graph():
    class Edge:
        def __init__(self, u, v, arg):
            assert isinstance(arg, str)
            self.arg = arg
            self.u = u
            self.v = v

        def __str__(self):
            return self.arg

        def __repr__(self):
            return f"Edge({self.u} -> {self.arg} -> {self.v})"

    class Node:
        def __init__(self, inputs=(), outputs=()):
            self.inputs = (tuple(inputs)
                           if isinstance(inputs, (list, tuple))
                           else (inputs,))
            self.outputs = (tuple(outputs)
                            if isinstance(outputs, (list, tuple))
                            else (outputs,))

        def __str__(self):
            return f"{list(self.inputs)}: {list(self.outputs)}"

        __repr__ = __str__

    from pprint import pprint

    class Term:
        def __repr__(self):
            return self.__class__.__name__

    class BrightnessTerm(Term):
        ARGS = ["chan_freq", "stokes"]

    class PhaseTerm(Term):
        ARGS = ["lm", "uvw", "chan_freq"]

    class ArgPack:
        def __init__(self, *args):
            self.args = args

        def __contains__(self, arg):
            return arg in self.args

        def __repr__(self):
            return "ArgPack"

        __str__ = __repr__

    class Transformer:
        def __init__(self, inputs, outputs):
            self.inputs = set(inputs) if inputs else set()
            self.outputs = set(outputs) if outputs else set()

        def __repr__(self):
            return "Transformer"

    argpack = ArgPack("phase_centre", "radec", "uvw",
                      "chan_freq", "stokes", "spi")
    terms = [PhaseTerm(), BrightnessTerm()]
    graph = [Edge(argpack, term, a) for term in terms for a in term.ARGS]

    in_outs = [
        [["phase_centre", "radec"], ["lm"]],
    ]

    xformers = {o: Transformer(ins, outs)
                for ins, outs in in_outs
                for o in outs}

    graph = []

    for term in terms:
        for a in term.ARGS:
            if a not in argpack:
                try:
                    xformer = xformers[a]
                except KeyError:
                    raise ValueError(f"{a} not in {argpack} "
                                     f"and no transformers available")

                for input in xformer.inputs:
                    graph.append(Edge(argpack, xformer, input))

                for output in xformer.outputs:
                    graph.append(Edge(xformer, term, output))
            else:
                graph.append(Edge(argpack, term, a))

    pprint(graph)
