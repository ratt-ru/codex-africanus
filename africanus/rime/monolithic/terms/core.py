import inspect
from functools import partial

from numba.experimental import structref
from numba.core import types

from africanus.rime.monolithic.common import result_type
from africanus.rime.monolithic.error import InvalidSignature


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

    REQUIRED = ("fields", "initialiser", "dask_schema", "sampler")

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
        fields_sig = inspect.signature(methods["fields"])

        for i, (n, p) in enumerate(fields_sig.parameters.items()):
            if i == 0 and n != "self":
                raise InvalidSignature(f"{name}.fields{fields_sig} "
                                       f"should be "
                                       f"{name}.fields(self, ...)")

            if p.kind == p.VAR_POSITIONAL:
                raise InvalidSignature(f"*{n} in "
                                       f"{name}.fields{fields_sig} "
                                       f"is not supported")

            if p.kind == p.VAR_KEYWORD:
                raise InvalidSignature(f"**{n} in "
                                       f"{name}.fields{fields_sig} "
                                       f"is not supported")

        # dask_schema_sig = inspect.signature(methods["dask_schema"])

        # if dask_schema_sig != fields_sig:
        #     raise InvalidSignature(f"{name}.dask_schema{dask_schema_sig} "
        #                            f"should be "
        #                            f"{name}.dask_schema{fields_sig}")

        Parameter = inspect.Parameter
        field_params = list(fields_sig.parameters.values())
        expected_init_params = field_params.copy()
        state_param = Parameter("state", Parameter.POSITIONAL_OR_KEYWORD)
        expected_init_params.insert(1, state_param)
        expected_init_sig = fields_sig.replace(parameters=expected_init_params)

        init_sig = inspect.signature(methods["initialiser"])

        if expected_init_sig != init_sig:
            raise InvalidSignature(f"{name}.initialiser{init_sig} "
                                   f"should be "
                                   f"{name}.fields{expected_init_sig}")

        sampler_sig = inspect.signature(methods["sampler"])
        params = [Parameter("self", kind=Parameter.POSITIONAL_OR_KEYWORD)]
        expected_sampler_sig = inspect.Signature(parameters=params)

        if sampler_sig != expected_sampler_sig:
            raise InvalidSignature(f"{name}.sampler{sampler_sig} "
                                   f"should be "
                                   f"{name}.sampler{expected_sampler_sig}")

        args = tuple(n for n, p in fields_sig.parameters.items()
                     if p.kind in {p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD}
                     and p.default is p.empty)

        kw = {n: p.default for n, p in fields_sig.parameters.items()
              if p.kind in {p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY}
              and p.default is not p.empty}

        expected_init_params.pop(0)
        expected_init_sig = fields_sig.replace(parameters=expected_init_params)
        validator = sigcheck_factory(expected_init_sig)

        namespace = namespace.copy()
        namespace["ARGS"] = args[1:]
        namespace["KWARGS"] = kw
        namespace["ALL_ARGS"] = tuple(fields_sig.parameters.keys())[1:]
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
    result_type = staticmethod(result_type)

    def __repr__(self):
        return self.__class__.__name__

    @classmethod
    def validate_sampler(cls, sampler):
        """ Validate the sampler implementation """
        sampler_sig = inspect.signature(sampler)
        Parameter = inspect.Parameter
        P = partial(Parameter, kind=Parameter.POSITIONAL_OR_KEYWORD)
        params = map(P, ["state", "s", "r", "t", "a1", "a2", "c"])
        expected_sig = inspect.Signature(params)

        if sampler_sig != expected_sig:
            raise InvalidSignature(f"{sampler.__name__}{sampler_sig}"
                                   f"should be "
                                   f"{sampler.__name__}{expected_sig}")
