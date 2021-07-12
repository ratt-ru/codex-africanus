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
    def __new__(cls, *args):
        return structref.StructRefProxy.__new__(cls, *args)


class PhaseTerm(Term):
    args = ["lm", "uvw", "chan_freq"]
    type = PhaseType
    proxy = PhaseProxy

    @classmethod
    def term_type(cls, lm, uvw, chan_freq):
        phase_dot = cls.result_type(lm, uvw, chan_freq)
        return PhaseType([
            ("lm", lm),
            ("uvw", uvw),
            ("chan_freq", chan_freq),
            ("phase_dot", phase_dot[:, :, :])
        ])

    @classmethod
    def initialiser(cls, lm, uvw, chan_freq):
        phase_dot = cls.result_type(lm, uvw, chan_freq)
        struct_type = PhaseType([
            ("lm", lm),
            ("uvw", uvw),
            ("chan_freq", chan_freq),
            ("phase_dot", phase_dot[:, :])
        ])

        def impl(lm, uvw, chan_freq):
            nsrc, _ = lm.shape
            nrow, _ = uvw.shape
            nchan, = chan_freq.shape

            state = structref.new(struct_type)
            state.lm = lm
            state.uvw = uvw
            state.chan_freq = chan_freq
            state.phase_dot = np.empty((nsrc, nrow), dtype=phase_dot)

            zero = lm.dtype.type(0.0)
            one = lm.dtype.type(1.0)
            C = phase_dot(-2*np.pi/3e8)

            for s in range(nsrc):
                l = lm[s, 0]
                m = lm[s, 1]
                n = one - l**2 - m**2
                n = np.sqrt(zero if n < zero else n) - one

                for r in range(nrow):
                    u = uvw[r, 0]
                    v = uvw[r, 1]
                    w = uvw[r, 2]

                    state.phase_dot[s, r] = C*(l*u + m*v + n*w)
                
            return state

        return impl

    @classmethod
    def sampler(cls):
        def impl(state, s, r, t, a1, a2, c):
            return np.exp(state.phase_dot[s, r]*state.chan_freq[c])

        return impl

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
    term_arg_types = tuple(tuple(args[j] for j in idx)
                           for idx in term_arg_inds)
    term_arg_names = tuple(tuple(term.args) for term in terms)

    it = zip(term_arg_names, term_arg_types)
    term_fields = [[(n, t) for n, t in zip(names, types)]
                   for names, types in it]

    term_types = [typ(fields) for typ, fields in zip(types, term_fields)]

    @intrinsic
    def implementation(typginctx, args):
        return_type = nb.types.Tuple(term_types)
        sig = return_type(args)

        def codegen(context, builder, signature, args):
            tuple_type = signature.args[0]
            return_type = signature.return_type
            ret_type = context.get_value_type(return_type)
            ret_tuple = cgutils.get_null_value(ret_type)

            for t in range(return_type.count):
                pass

                

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
            # tfactory = term_factory(args[0], terms, term_arg_inds)

            def impl(*args):
                # terms = tfactory(args)  # noqa: F841
                pass

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


    @generated_jit(nopython=True)
    def fn(lm, uvw, chan_freq):
        init = PhaseTerm.initialiser(lm, uvw, chan_freq)
        sampler = PhaseTerm.sampler()
        init = register_jitable(inline="always")(init)
        sampler = register_jitable(inline="always")(sampler)


        def impl(lm, uvw, chan_freq):
            state = init(lm, uvw, chan_freq)
            nsrc, _ = lm.shape
            nrow, _ = uvw.shape
            nchan, = chan_freq.shape

            result = np.zeros((nrow, nchan), dtype=np.complex64)

            for s in range(nsrc):
                for r in range(nrow):
                    for c in range(nchan):
                        result[r, c] += sampler(state, s, r, 0, 0, 0, c)

            return result

        return impl

    out = rime(lm=lm, uvw=uvw, chan_freq=chan_freq, stokes=stokes)

    print(fn(lm, uvw, chan_freq))