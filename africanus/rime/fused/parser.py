import ast


class RimeParseError(ValueError):
    pass


class ListTransformer(ast.NodeTransformer):
    def visit_Module(self, node):
        if len(node.body) != 1 or not isinstance(node.body[0], ast.Expr):
            raise RimeParseError("Module must contain a single expression")

        expr = node.body[0]

        if not isinstance(expr.value, ast.List):
            raise RimeParseError("Expression must contain a dictionary")

        return self.visit(expr).value

    def visit_Name(self, node):
        return node.id.upper()

    def visit_List(self, node):
        return list(self.visit(v) for v in node.elts)

    def visit_Num(self, node):
        return node.n


class RimeSpecificationError(ValueError):
    pass


class RimeSpecification:
    def __init__(self, equation, stokes, corrs):
        valid_stokes = {"I", "Q", "U", "V"}

        if valid_stokes & set(stokes) != valid_stokes:
            raise RimeSpecificationError(
                f"{stokes} contains invalid stokes parameters. "
                f"Only {valid_stokes} are accepted")

        self.equation = equation
        self.stokes = stokes
        self.corrs = corrs
        self.feed_type = self._feed_type(corrs)

    @staticmethod
    def _feed_type(corrs):
        linear = {"XX", "XY", "YX", "YY"}
        circular = {"RR", "RL", "LR", "LL"}
        scorrs = set(corrs)

        if scorrs.issubset(linear):
            return "linear"

        if scorrs.issubset(circular):
            return "circular"

        raise RimeSpecificationError(
            f"Correlations must be purely linear or circular. "
            f"Got {corrs}")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"{self.equation}, {self.stokes}, {self.corrs})")

    def __str__(self):
        return f"{self.equation}: {self.stokes} -> {self.corrs}"


def parse_str_list(str_list):
    return ListTransformer().visit(ast.parse(str_list))


def parse_rime(rime: str):
    bits = [s.strip() for s in rime.split(":")]

    try:
        rime_bits, polarisation_bits = bits
    except (ValueError, TypeError):
        raise RimeParseError(
            f"RIME must be of the form "
            f"[Gp, (Kpq, Bpq), Gq]: [I,Q,U,V] -> [XX,XY,YX,YY]. "
            f"Got {rime}.")

    bits = [s.strip() for s in polarisation_bits.split("->")]

    try:
        stokes_bits, corr_bits = bits
    except (ValueError, TypeError):
        raise RimeParseError(
            f"Polarisation specification must be of the form "
            f"[I,Q,U,V] -> [XX,XY,YX,YY]. Got {polarisation_bits}.")

    stokes_bits, corr_bits = bits

    stokes = parse_str_list(stokes_bits)
    corrs = parse_str_list(corr_bits)

    if (not isinstance(stokes, list) or
            not all(isinstance(s, str) for s in stokes)):

        raise RimeParseError(
            f"Stokes specification must be of the form "
            f"[I,Q,U,V]. Got {stokes}.")

    if (not isinstance(corrs, list) or
            not all(isinstance(c, str) for c in corrs)):

        raise RimeParseError(
            f"Correlation specification must be of the form "
            f"[XX,XY,YX,YY]. Got {corrs}.")

    stokes = [s.upper() for s in stokes]
    corrs = [c.upper() for c in corrs]

    return RimeSpecification(rime_bits, stokes, corrs)
