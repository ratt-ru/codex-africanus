import abc
from numba import njit, generated_jit
from numba.core import compiler, cgutils, typing, types  # noqa
from numba.core.typed_passes import type_inference_stage
from numba.core.errors import TypingError
from numba.extending import (
    overload,
    overload_method,
    register_jitable,
    intrinsic)
from numba.experimental import jitclass, structref
from numba.np.numpy_support import as_dtype
import numba as nb
import numpy as np


class BaseStructRef(types.StructRef):
    def preprocess_fields(self, fields):
        """ Disallow literal types in field definitions """
        return tuple((n, types.unliteral(t)) for n, t in fields)    


class Term(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def term_type(cls, *args):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def initialiser(cls, *args):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def sampler(cls):
        raise NotImplementedError

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
class GeneralStateType(BaseStructRef):
    pass

@structref.register
class PhaseType(BaseStructRef):
    pass

class PhaseTerm(Term):
    term_args = ["lm", "uvw", "chan_freq"]
    abstract_type = PhaseType

    @classmethod
    def term_type(cls, *args):
        assert len(cls.term_args) == len(args)
        lm, uvw, chan_freq = args
        phase_dot = cls.result_type(lm, uvw, chan_freq)
        return cls.abstract_type([
            ("lm", lm),
            ("uvw", uvw),
            ("chan_freq", chan_freq),
            ("phase_dot", phase_dot[:, :])
        ])

    @classmethod
    def initialiser(cls, *args):
        struct_type = cls.term_type(*args)
        dot_dtype = struct_type.field_dict["phase_dot"].dtype

        def phase(lm, uvw, chan_freq):
            nsrc, _ = lm.shape
            nrow, _ = uvw.shape
            nchan, = chan_freq.shape

            state = structref.new(struct_type)
            state.lm = lm
            state.uvw = uvw
            state.chan_freq = chan_freq
            state.phase_dot = np.empty((nsrc, nrow), dtype=dot_dtype)

            zero = lm.dtype.type(0.0)
            one = lm.dtype.type(1.0)
            C = dot_dtype(-2*np.pi/3e8)

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

        return phase

    @classmethod
    def sampler(cls):
        def phase_sample(state, s, r, t, a1, a2, c):
            return np.exp(state.phase_dot[s, r] * state.chan_freq[c])

        return phase_sample

@structref.register
class BrightnessType(BaseStructRef):
    pass

class BrightnessTerm(Term):
    term_args = ["stokes", "chan_freq"]
    abstract_type = BrightnessType

    @classmethod
    def term_type(cls, *args):
        assert len(cls.term_args) == len(args)
        stokes, chan_freq = args

        return cls.abstract_type([
            ("stokes", stokes),
            ("chan_freq", chan_freq)
        ])

    @classmethod
    def initialiser(cls, *args):
        struct_type = cls.term_type(*args)

        def brightness(stokes, chan_freq):
            state = structref.new(struct_type)
            return state

        return brightness

    @classmethod
    def sampler(cls):
        def brightness_sampler(state, s, r, t, a1, a2, c):
            return 0

        return brightness_sampler

# class PhaseProxy(structref.StructRefProxy):
#     pass
# class BrightnessProxy(structref.StructRefProxy):
#     pass
#structref.define_proxy(PhaseProxy, PhaseType, ["lm", "uvw", "chan_freq"])
#structref.define_proxy(BrightnessProxy, BrightnessType, ["stokes", "chan_freq"])


def term_factory(args, terms, term_arg_inds):
    term_arg_types = tuple(tuple(args[j] for j in idx)
                           for idx in term_arg_inds)
    assert len(terms) == len(term_arg_inds)

    term_state_types = [term.term_type(*arg_types)
                        for term, arg_types
                        in zip(terms, term_arg_types)]

    constructors = [term.initialiser(*arg_types)
                    for term, arg_types
                    in zip(terms, term_arg_types)]

    samplers = [term.sampler() for term in terms]

    @intrinsic
    def construct_terms(typginctx, args):
        if not isinstance(args, nb.core.types.Tuple):
            raise TypingError("args must be a Tuple")
        
        general_state_type = GeneralStateType([
            ("source", nb.int64),
            ("row", nb.int64),
            ("chan", nb.int64),
            ("corr", nb.int64)
        ])

        return_type = nb.types.Tuple(term_state_types)

        sig = return_type(args)

        def codegen(context, builder, signature, args):
            return_type = signature.return_type

            if not isinstance(return_type, nb.types.Tuple):
                raise TypingError("signature.return_type should be a Tuple")

            llvm_ret_type = context.get_value_type(return_type)
            ret_tuple = cgutils.get_null_value(llvm_ret_type)

            if not len(args) == 1:
                raise TypingError("args must contain a single value")

            # Our single argument is a tuple of arguments, but we
            # need to extract those arguments necessary to construct
            # the term StructRef
            constructor_args = [[builder.extract_value(args[0], j)
                                 for j in idx]
                                for idx in term_arg_inds]

            # Sanity
            assert all(len(ca) == len(at) for ca, at
                       in zip(constructor_args, term_arg_types))

            for ti in range(return_type.count):
                constructor_sig = return_type[ti](*term_arg_types[ti])
                data = context.compile_internal(builder,
                                                constructors[ti],
                                                constructor_sig,
                                                constructor_args[ti])

                ret_tuple = builder.insert_value(ret_tuple, data, ti)

            return ret_tuple

        return sig, codegen


    @intrinsic
    def apply_terms(typingctx, state):
        if not isinstance(state, nb.types.Tuple):
            raise TypingError(f"{state} must be a Tuple")

        if not len(state) == len(term_state_types):
            raise TypingError(f"State length does not equal the number of terms")

        if not all(st == tst for st, tst in zip(state, term_state_types)):
            raise TypingError(f"State types don't match")

        sampler_ir = list(map(compiler.run_frontend, samplers))

        ir_args = [(typ,) + (nb.int64,)*6 for typ in term_state_types]

        type_infer = [type_inference_stage(typingctx, ir, args, None)
                      for ir, args in zip(sampler_ir, ir_args)]

        sig = nb.types.none(state)

        def codegen(context, builder, signature, args):
            ret_type = context.get_value_type(signature.return_type)
            return cgutils.get_null_value(ret_type)

        return sig, codegen

    return construct_terms, apply_terms



class rime_factory:
    def __init__(self):
        terms = [PhaseTerm, BrightnessTerm]
        args = list(sorted(set(a for t in terms for a in t.term_args)))
        arg_map = {a: i for i, a in enumerate(args)}
        term_arg_inds = tuple(tuple(arg_map[a] for a in t.term_args) for t in terms)

        try:
            lm_i = arg_map["lm"]
            uvw_i = arg_map["uvw"]
            chan_freq_i = arg_map["chan_freq"]
            stokes_i = arg_map["stokes"]
        except KeyError as e:
            raise ValueError(f"'{str(e)}' is a required argument")

        @generated_jit(nopython=True, nogil=True, cache=True)
        def rime(*args):
            assert len(args) == 1
            state_factory, apply_terms = term_factory(args[0], terms, term_arg_inds)

            def impl(*args):
                term_state = state_factory(args)  # noqa: F841

                nsrc, _ = args[lm_i].shape
                nrow, _ = args[uvw_i].shape
                nchan, = args[chan_freq_i].shape
                _, ncorr = args[stokes_i].shape

                apply_terms(term_state)

                vis = np.zeros((nrow, nchan, ncorr), np.complex128)

                for s in range(nsrc):
                    for r in range(nrow):
                        for f in range(nchan):
                            for c in range(ncorr):
                                vis[r, f, c] += 1

                return vis

            return impl

        self.terms = terms
        self.args = args
        self.arg_map = arg_map
        self.impl = rime

    def __call__(self, **kwargs):
        try:
            args = tuple(kwargs[a] for a in self.args)
        except KeyError as e:
            raise ValueError(f"{e} is a required kwarg")
        else:
            return self.impl(*args)


if __name__ == "__main__":
    rime = rime_factory()
    lm = np.random.random(size=(10, 2))
    uvw = np.random.random(size=(5, 3))
    chan_freq = np.linspace(.856e9, 2*.859e9, 4)
    stokes = np.random.random(size=(10, 4))


    @generated_jit(nopython=True, cache=True)
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


    print(out)

    # import pdb;  pdb.set_trace()
    res = fn(lm, uvw, chan_freq)
    print(res)

    from pprint import pprint

    print(fn.inspect_types(signature=fn.signatures[0]))
