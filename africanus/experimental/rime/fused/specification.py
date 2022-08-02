import ast
from importlib import import_module
import inspect
import multiprocessing
from pathlib import Path
import re

from africanus.experimental.rime.fused.terms.core import Term
from africanus.experimental.rime.fused.terms.phase import Phase
from africanus.experimental.rime.fused.terms.brightness import Brightness
from africanus.experimental.rime.fused import terms as term_mod
from africanus.experimental.rime.fused.transformers.core import Transformer
from africanus.experimental.rime.fused import transformers as transformer_mod
from africanus.util.patterns import LazyProxy


TERM_STRING_REGEX = re.compile("([A-Z])(pq|p|q)")


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


def _decompose_term_str(term_str):
    match = TERM_STRING_REGEX.match(term_str)

    if not match:
        raise RimeSpecificationError(
            f"{term_str} does not match {TERM_STRING_REGEX.pattern}")

    return tuple(match.groups())


class RimeSpecification:
    """
    Defines a unique Radio Interferometer Measurement Equation (RIME)

    The RIME is composed of a number of Jones Terms, which are multiplied
    together and combined to produce model visibilities.

    The ``RimeSpecification`` specifies the order of these Jones Terms and
    supports custom Jones terms specified by the user.

    One of the simplest RIME's that can be expressed involve a ``Phase`` (Kpq)
    and a ``Brightness`` (Bpq) term.
    The specification for this RIME is as follows:

    .. code-block:: python

        rime_spec = RimeSpecification("(Kpq, Bpq): [I,Q,U,V] -> [XX,XY,YX,YY]")

    ``(Kpq, Bpq)`` specifies the onion more formally defined
    :ref:`here <experimental-fused-rime-api-anchor>`, while
    ``[I,Q,U,V] -> [XX,XY,YX,YY]`` defines the stokes to correlation conversion
    within the RIME.
    It also identifies whether the RIME is handling linear
    or circular feeds.

    **Term Configuration**

    The ``pq`` in Kpq and Bpq signifies that their values are calculated
    per-baseline.
    It is possible to specify per-antenna terms: ``Kp`` and ``Kq``
    for example which
    represent left (ANTENNA1) and right (ANTENNA2) terms respectively.
    Not that the hermitian transpose of the right term is automatically
    performed and does not need to be implemented in the Term itself.
    Thus, for example, ``(Kp, Bpq, Kq)`` specifies a RIME where the
    Phase Term is separated into left and right terms, while the
    Brightness Matrix is calculated per-baseline.

    **Stokes to Correlation Mapping**

    ``[I,Q,U,V] -> [XX,XY,YX,YY]`` specifies a mapping from
    four stokes parameters to four correlations.
    Both linear ``[XX,XY,YX,YY]`` and circular ``[RR,RL,LR,LL]``
    feed types are supported. A variety of mappings are possible:

    .. code-block:: python

        [I,Q,U,V] -> [XX,XY,YX,YY]
        [I,Q] -> [XX,YY]
        [I,Q] -> [RR,LL]

    **Custom Terms**

    Custom Term classes implemented by a user can be added to
    the RIME as follows:

    .. code-block:: python

        class CustomJones(Term):
            ...

        spec = RimeSpecification("(Apq,Kpq,Bpq)", terms={"A": CustomJones})

    Parameters
    ----------
    specification : str
        A string specifying the terms in the RIME and the stokes
        to correlation conversion.
    terms : dict of str or Terms
        A map describing custom
        :class:`~africanus.experimental.rime.fused.terms.core.Term`
        implementations.
        If one has defined a custom Gaussian Term class,
        for use in RIME ``(Cpq, Kpq, Bpq)``, this should be
        supplied as :code:`terms={"C": Gaussian}`.
        strings can be supplied for predefined RIME classes.
    transformers : list of Transformers
        A list of
        :class:`~africanus.experimental.rime.fused.transformers.core.Transformer`
        classes.

    """

    VALID_STOKES = {"I", "Q", "U", "V"}
    TERM_MAP = {
        "K": "Phase",
        "B": "Brightness",
        "L": "FeedRotation",
        "E": "BeamCubeDDE"}

    def __reduce__(self):
        return (RimeSpecification, self._saved_args)

    def __hash__(self):
        return hash(self._saved_args)

    def __eq__(self, rhs):
        return (isinstance(rhs, RimeSpecification) and
                self._saved_args == rhs._saved_args)

    def __init__(self, specification, terms=None, transformers=None):
        # Argument Handling
        if not isinstance(specification, str):
            raise TypeError(f"specification: {specification} is not a str")

        if not terms:
            saved_terms = None
        elif isinstance(terms, dict):
            saved_terms = frozenset(terms.items())
        elif isinstance(terms, (tuple, list, set, frozenset)):
            saved_terms = frozenset(terms)
            terms = dict(saved_terms)
        else:
            raise TypeError(
                f"terms: {terms} must be a dictionary or "
                f"an iterable of (key, value) pairs")

        if not transformers:
            saved_transforms = transformers
        elif isinstance(transformers, (tuple, list, set, frozenset)):
            saved_transforms = frozenset(transformers)
        else:
            raise TypeError(
                f"transformers: {transformers} must be "
                f"an iterable of Transformers")

        # Parse the specification
        equation, stokes, corrs = parse_rime(specification)

        if not set(stokes).issubset(self.VALID_STOKES):
            raise RimeSpecificationError(
                f"{stokes} contains invalid stokes parameters. "
                f"Only {self.VALID_STOKES} are accepted")

        self._saved_args = (specification, saved_terms, saved_transforms)
        self.equation = equation
        self.stokes = stokes
        self.corrs = corrs
        self.feed_type = feed_type = self._feed_type(corrs)

        # Determine term types
        term_map = {**self.TERM_MAP, **terms} if terms else self.TERM_MAP
        term_types = search_types(term_mod, Term)
        transformer_types = search_types(transformer_mod, Transformer)
        term_char, term_cfgs = zip(*(_decompose_term_str(t) for t in equation))

        try:
            terms_wanted = tuple(term_map[t] for t in term_char)
        except KeyError as e:
            raise RimeSpecificationError(f"Unknown term {str(e)}")

        try:
            term_types = tuple(t if isinstance(t, type) and issubclass(t, Term)
                               else term_types[t] for t in terms_wanted)
        except KeyError as e:
            raise RimeSpecificationError(f"Can't find a type for {str(e)}")

        Pool = multiprocessing.get_context("spawn").Pool
        pool = LazyProxy((Pool, RimeSpecification._finalise_pool), 4)

        # Create the terms
        terms = []
        global_kw = {
            "corrs": corrs,
            "stokes": stokes,
            "feed_type": feed_type,
            "process_pool": pool
        }

        for cls, cfg in zip(term_types, term_cfgs):
            if cfg == "pq":
                cfg = "middle"
            elif cfg == "p":
                cfg = "left"
            elif cfg == "q":
                cfg = "right"
            else:
                raise ValueError(f"Illegal configuration {cfg}")

            init_sig = inspect.signature(cls.__init__)
            available_kw = {"configuration": cfg, **global_kw}
            cls_kw = {}

            if "configuration" not in init_sig.parameters:
                raise RimeSpecification(
                    f"{cls}.__init__{init_sig} must take a "
                    f"'configuration' argument and call "
                    f"super().__init__(configuration)")

            for a, p in list(init_sig.parameters.items())[1:]:
                if p.kind not in {p.POSITIONAL_ONLY,
                                  p.POSITIONAL_OR_KEYWORD}:
                    raise RimeSpecification(
                        f"{cls}.__init__{init_sig} may not contain "
                        f"*args or **kwargs")

                try:
                    cls_kw[a] = available_kw[a]
                except KeyError:
                    raise RimeSpecificationError(
                        f"{cls}.__init__{init_sig} wants argument {a} "
                        f"but it is not available. "
                        f"Available args: {available_kw}")

            term = cls(**cls_kw)
            terms.append(term)

        term_type_set = set(term_types)

        if Phase not in term_type_set:
            raise RimeSpecificationError(
                "RIME must at least contain a Phase term")

        if Brightness not in term_type_set:
            raise RimeSpecificationError(
                "RIME must at least contain a Brightness term")

        transformers = []

        for cls in transformer_types.values():
            init_sig = inspect.signature(cls.__init__)
            cls_kw = {}

            for a, p in list(init_sig.parameters.items())[1:]:
                if p.kind not in {p.POSITIONAL_ONLY,
                                  p.POSITIONAL_OR_KEYWORD}:
                    raise RimeSpecification(
                        f"{cls}.__init__{init_sig} may not contain "
                        f"*args or **kwargs")

                try:
                    cls_kw[a] = available_kw[a]
                except KeyError:
                    raise RimeSpecificationError(
                        f"{cls}.__init__{init_sig} wants argument {a} "
                        f"but it is not available. "
                        f"Available args: {available_kw}")

            transformer = cls(**cls_kw)
            transformers.append(transformer)

        self.terms = terms
        self.transformers = transformers

    @staticmethod
    def _finalise_pool(pool):
        pool.terminate()

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

    @staticmethod
    def flatten_eqn(equation):
        if isinstance(equation, (tuple, list)):
            it = iter(map(RimeSpecification.flatten_eqn, equation))
            return "".join(("[", ",".join(it), "]"))
        elif isinstance(equation, str):
            return equation
        else:
            raise TypeError(f"equation: {equation} must "
                            f"be a string or sequence")

    def equation_bits(self):
        return self.flatten_eqn(self.equation)

    def __repr__(self):
        return "".join((self.__class__.__name__, "(\"", str(self), "\")"))

    def __str__(self):
        return "".join((
            self.equation_bits(),
            ": ",
            "".join(("[", ",".join(self.stokes), "]")),
            " -> ",
            "".join(("[", ",".join(self.corrs), "]"))))


def parse_str_list(str_list):
    return RimeTransformer().visit(ast.parse(str_list))
