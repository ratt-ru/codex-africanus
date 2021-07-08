from abc import abstractclassmethod, abstractmethod, abstractproperty


import abc
from numba import njit, generated_jit
from numba.core import cgutils, typing, types  # noqa
from numba.extending import (
    overload,
    overload_method,
    register_jitable,
    intrinsic)
from numba.experimental import jitclass, structref
from numba.np.numpy_support import as_dtype
import numba as nb
import numpy as np


class BaseTermType(types.StructRef):
    def preprocess_fields(self, fields):
        """ Disallow literal types in field definitions """
        return tuple((n, types.unliteral(t)) for n, t in fields)    


class Term(abc.ABC):
    @staticmethod
    def result_type(*args):
        types = []

        for arg in args:
            if isinstance(arg, nb.types.Type):
                if isinstance(arg, nb.types.Array):
                    types.append(as_dtype(arg.dtype))
                else:
                    types.append(as_dtype(arg))

            elif isinstance(arg, np.generic):
                types.append(arg)
            else:
                raise TypeError(f"Unknown type {type(arg)} of argument {arg}")

        return nb.typeof(np.result_type(*types)).dtype


@structref.register
class PhaseType(BaseTermType):
    pass

class PhaseProxy(structref.StructRefProxy):
    def __new__(cls, lm, uvw, chan_freq):
        return structref.StructRefProxy.__new__(cls, lm, uvw, chan_freq)


class PhaseTerm(Term):
    args = ["lm", "uvw", "chan_freq"]
    type = PhaseType
    proxy = PhaseProxy

structref.define_proxy(PhaseProxy, PhaseType, PhaseTerm.args)


@structref.register
class BrightnessType(BaseTermType):
    pass

class BrightnessProxy(structref.StructRefProxy):
    def __new__(cls, stokes, chan_freq):
        return structref.StructRefProxy.__new__(cls, stokes, chan_freq)


class BrightnessTerm(Term):
    args = ["stokes", "chan_freq"]
    type = BrightnessType
    proxy = BrightnessProxy

structref.define_proxy(BrightnessProxy, BrightnessType, BrightnessTerm.args)


def term_factory(args, terms, term_arg_inds):
    types = [term.type for term in terms]
    proxies = [term.proxy for term in terms]
    term_arg_types = tuple(tuple(args[j] for j in idx) for idx in term_arg_inds)
    term_arg_names = tuple(tuple(term.args) for term in terms)

    it = zip(term_arg_names, term_arg_types)
    term_fields = [[(n, t) for n, t in zip(names, types)]
                   for names, types in it]

    types = [typ(fields) for typ, fields in zip(types, term_fields)]

    @intrinsic
    def implementation(typginctx, args):
        return_type = nb.types.Tuple(types)
        sig = return_type(args)

        def codegen(context, builder, signature, args):
            return_type = signature.return_type
            ret_type = context.get_value_type(return_type)
            return cgutils.get_null_value(ret_type)

        return sig, codegen

    return implementation

class rime_factory:
    def __init__(self):
        terms = [PhaseTerm, BrightnessTerm]
        args = list(sorted(set(a for t in terms for a in t.args)))
        arg_map = {a: i for i, a in enumerate(args)}
        term_arg_inds = tuple(tuple(arg_map[a] for a in t.args) for t in terms)

        @generated_jit(nopython=True, nogil=True, cache=True)
        def function(*args):
            tfactory = term_factory(args[0], terms, term_arg_inds)

            def impl(*args):
                terms = tfactory(args)  # noqa: F841

            return impl

        self.terms = terms
        self.args = args
        self.arg_map = arg_map
        self.impl = function

    def __call__(self, **kwargs):

        try:
            args = tuple(kwargs[a] for a in self.args)
        except KeyError as e:
            raise ValueError(f"{e} is a required kwarg")
        else:
            return self.impl(*args)


if __name__ == "__main__":
    @njit
    def fn():
        return PhaseProxy(np.ones(100), 2, 3)

    rime = rime_factory()
    lm = np.random.random(size=(10, 2))
    uvw = np.random.random(size=(5, 3))
    chan_freq = np.linspace(.856e9, 2*.859e9, 4)
    stokes = np.random.random(size=(10, 4))

    out = rime(lm=lm, uvw=uvw, chan_freq=chan_freq, stokes=stokes)


    import pdb; pdb.set_trace()

    print(fn())