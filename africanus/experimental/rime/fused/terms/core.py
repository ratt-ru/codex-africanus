import inspect
from functools import partial

from numba.experimental import structref
from numba.core import types

from africanus.experimental.rime.fused.error import InvalidSignature


@structref.register
class StateStructRef(types.StructRef):
    def preprocess_fields(self, fields):
        """ Disallow literal types in field definitions """
        return tuple((n, types.unliteral(t)) for n, t in fields)


def sigcheck_factory(expected_sig):
    def check_constructor_signature(self, fn):
        sig = inspect.signature(fn)
        if sig != expected_sig:
            raise ValueError(f"{fn.__name__}{sig} should be "
                             f"{fn.__name__}{expected_sig}")

    return check_constructor_signature


class TermMetaClass(type):
    """
    Metaclass which checks that the appropriate methods are
    implemented on any subclass of `Term` and that their
    signatures agree with each other.

    Also sets `ARGS`, `KWARGS` and `ALL_ARGS`
    class members on the subclass based on the above
    signatures
    """

    REQUIRED = ("init_fields", "dask_schema", "sampler")

    @classmethod
    def _expand_namespace(cls, name, namespace):
        """
        Check that the expected implementations are in the namespace.

        Also assign the args and kwargs associated with the implementations
        into the namespace

        Returns
        -------
        dict
            A copy of `namespace` with args and kwargs assigned
        """
        methods = []

        for method_name in cls.REQUIRED:
            try:
                method = namespace[method_name]
            except KeyError:
                raise NotImplementedError(f"{name}.{method_name}")
            else:
                methods.append(method)

        methods = dict(zip(cls.REQUIRED, methods))
        init_fields_sig = inspect.signature(methods["init_fields"])
        field_params = list(init_fields_sig.parameters.values())

        if len(init_fields_sig.parameters) < 2:
            raise InvalidSignature(f"{name}.init_fields{init_fields_sig} "
                                   f"should be "
                                   f"{name}.init_fields(self, typingctx, ...)")

        it = iter(init_fields_sig.parameters.items())
        first, second = next(it), next(it)

        if first[0] != "self" or second[0] != "typingctx":
            raise InvalidSignature(f"{name}.init_fields{init_fields_sig} "
                                   f"should be "
                                   f"{name}.init_fields(self, typingctx, ...)")

        for n, p in it:
            if p.kind == p.VAR_POSITIONAL:
                raise InvalidSignature(f"*{n} in "
                                       f"{name}.init_fields{init_fields_sig} "
                                       f"is not supported")

            if p.kind == p.VAR_KEYWORD:
                raise InvalidSignature(f"**{n} in "
                                       f"{name}.init_fields{init_fields_sig} "
                                       f"is not supported")

        dask_schema_sig = inspect.signature(methods["dask_schema"])
        expected_dask_params = field_params[0:1] + field_params[2:]
        expected_dask_sig = init_fields_sig.replace(
            parameters=expected_dask_params)

        if dask_schema_sig != expected_dask_sig:
            raise InvalidSignature(f"{name}.dask_schema{dask_schema_sig} "
                                   f"should be "
                                   f"{name}.dask_schema{expected_dask_sig}")

        Parameter = inspect.Parameter
        expected_init_sig = init_fields_sig.replace(
                                parameters=field_params[2:])
        validator = sigcheck_factory(expected_init_sig)

        sampler_sig = inspect.signature(methods["sampler"])
        params = [Parameter("self", kind=Parameter.POSITIONAL_OR_KEYWORD)]
        expected_sampler_sig = inspect.Signature(parameters=params)

        if sampler_sig != expected_sampler_sig:
            raise InvalidSignature(f"{name}.sampler{sampler_sig} "
                                   f"should be "
                                   f"{name}.sampler{expected_sampler_sig}")

        args = tuple(n for n, p in init_fields_sig.parameters.items()
                     if p.kind in {p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD}
                     and n not in {"self", "typingctx"}
                     and p.default is p.empty)

        kw = [(n, p.default) for n, p in init_fields_sig.parameters.items()
              if p.kind in {p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY}
              and n not in {"self", "typingctx"}
              and p.default is not p.empty]

        namespace = namespace.copy()
        namespace["ARGS"] = args
        namespace["KWARGS"] = dict(kw)
        namespace["ALL_ARGS"] = args + tuple(k for k, _ in kw)
        namespace["validate_constructor"] = validator

        return namespace

    @classmethod
    def term_in_bases(cls, bases):
        """ Is `Term` in bases? """
        for base in bases:
            if base is Term or cls.term_in_bases(base.__bases__):
                return True

        return False

    def __new__(mcls, name, bases, namespace):
        # Check methods on any subclasses of Term
        # and expand the subclass namespace
        if mcls.term_in_bases(bases):
            namespace = mcls._expand_namespace(name, namespace)

        return super(TermMetaClass, mcls).__new__(mcls, name, bases, namespace)


class Term(metaclass=TermMetaClass):
    def __init__(self, configuration):
        self._configuration = configuration

    @property
    def configuration(self):
        return self._configuration

    def __eq__(self, rhs):
        return (isinstance(rhs, Term) and
                self._configuration == rhs._configuration)

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__class__.__name__

    @classmethod
    def validate_sampler(cls, sampler):
        """ Validate the sampler implementation """
        sampler_sig = inspect.signature(sampler)
        Parameter = inspect.Parameter
        P = partial(Parameter, kind=Parameter.POSITIONAL_OR_KEYWORD)
        params = map(P, ["state", "s", "r", "t", "f1", "f2", "a1", "a2", "c"])
        expected_sig = inspect.Signature(params)

        if sampler_sig != expected_sig:
            raise InvalidSignature(f"{sampler.__name__}{sampler_sig}"
                                   f"should be "
                                   f"{sampler.__name__}{expected_sig}")
