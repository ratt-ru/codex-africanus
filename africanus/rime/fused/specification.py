import ast
from pathlib import Path
from importlib import import_module

from africanus.rime.fused.terms.core import Term
from africanus.rime.fused.terms.phase import Phase
from africanus.rime.fused.terms.brightness import Brightness
from africanus.rime.fused import terms as term_mod
from africanus.rime.fused.transformers.core import Transformer
from africanus.rime.fused import transformers as transformer_mod


class RimeParseError(ValueError):
    pass


class RimeTransformer(ast.NodeTransformer):
    def visit_Module(self, node):
        if len(node.body) != 1 or not isinstance(node.body[0], ast.Expr):
            raise RimeParseError("Module must contain a single expression")

        expr = node.body[0]

        if not isinstance(expr.value, (ast.Tuple, ast.List)):
            raise RimeParseError("Expression must be a tuple or list")

        return self.visit(expr).value

    def visit_Name(self, node):
        return node.id

    def visit_List(self, node):
        return list(self.visit(v) for v in node.elts)

    def visit_Tuple(self, node):
        return tuple(self.visit(v) for v in node.elts)

    def visit_Num(self, node):
        return node.n


class RimeSpecificationError(ValueError):
    pass


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

    equation = parse_str_list(rime_bits)

    if (not isinstance(equation, (tuple, list)) or
        any(isinstance(e, (tuple, list)) for e in equation) or
            not all(isinstance(t, str) for t in equation)):
        raise RimeParseError(
                f"RIME must be a tuple/list of Terms "
                f"(Kpq, Bpq). Got {equation}.")

    return equation, stokes, corrs


def search_types(module, typ, exclude=("__init__.py", "core.py")):
    """Searches for subclasses of `typ` in files of  `module`.

    Parameters
    ----------
    module: python module
    typ: type or tuple of types
        types to return
    exclude: tuple of str
        Filenames to exclude from the search.

    Returns
    -------
    types : dict
        a :code:`{name, type}` dictionary
    """
    if isinstance(typ, type):
        typ = (typ,)
    elif isinstance(typ, (list, tuple)):
        typ = tuple(typ)

        if not all(isinstance(t, type) for t in typ):
            raise TypeError(f"typ: {typ} must be a type/tuple of types")
    else:
        raise TypeError(f"typ: {typ} must be a type/tuple of types")

    path = Path(module.__file__).parent
    search = set(path.glob("*.py")) - set(map(path.joinpath, exclude))
    typs = {}

    # For each python module in module, search for typs
    for py_file in search:
        mod = import_module(f"{module.__package__}.{py_file.stem}")

        for k, v in vars(mod).items():
            if (k.startswith("_") or not isinstance(v, type) or
                    not issubclass(v, typ) or v in typ):
                continue

            typs[k] = v

    return typs


class RimeSpecification:
    VALID_STOKES = {"I", "Q", "U", "V"}
    TERM_MAP = {
        "Kpq": "Phase",
        "Bpq": "Brightness"}

    def __init__(self, specification, terms=None, transformers=None):
        if not isinstance(specification, str):
            raise TypeError(f"specification: {specification} is not a str")

        equation, stokes, corrs = parse_rime(specification)

        if self.VALID_STOKES & set(stokes) != self.VALID_STOKES:
            raise RimeSpecificationError(
                f"{stokes} contains invalid stokes parameters. "
                f"Only {self.VALID_STOKES} are accepted")

        term_map = {**self.TERM_MAP, **terms} if terms else self.TERM_MAP
        term_types = search_types(term_mod, Term)
        transformer_types = search_types(transformer_mod, Transformer)

        try:
            terms_wanted = tuple(term_map[t] for t in equation)
        except KeyError as e:
            raise RimeSpecificationError(f"Unknown term {str(e)}")

        try:
            term_types = tuple(term_types[t] for t in terms_wanted)
        except KeyError as e:
            raise RimeSpecificationError(f"Can't find a type for {str(e)}")

        terms = []
        found_phase = found_brightness = False

        for cls in term_types:
            if issubclass(cls, Brightness):
                terms.append(cls(stokes, corrs))
                found_brightness = True
            elif issubclass(cls, Phase):
                terms.append(cls())
                found_phase = True
            else:
                terms.append(cls())

        if not found_phase:
            raise RimeSpecification(
                "RIME must at least contain a Phase term")

        if not found_brightness:
            raise RimeSpecification(
                "RIME must at least contain a Brightness term")

        self.terms = terms
        self.transformers = [cls() for cls in transformer_types.values()]
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
    return RimeTransformer().visit(ast.parse(str_list))
