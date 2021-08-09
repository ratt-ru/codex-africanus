import ast


class RimeParseError(ValueError):
    pass


class RimeTransformer(ast.NodeTransformer):
    def visit_Module(self, node):
        if len(node.body) != 1 or not isinstance(node.body[0], ast.Expr):
            raise RimeParseError("Module must contain a single expression")


def parse_rime(rime: str):
    if "->" not in rime:
        raise RimeParseError

    bits = [s.strip() for s in rime.split("->")]

    if len(bits) != 2:
        raise RimeParseError(f"{rime} doesn't adhere to the "
                             f"'rime -> polarisation' formalism")

    rime_bits, polarisation_bits = bits

    polarisation = ast.literal_eval(polarisation_bits)
    rime_expr = ast.literal_eval(rime_bits)
    return rime_expr, polarisation
