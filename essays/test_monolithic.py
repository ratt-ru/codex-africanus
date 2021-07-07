import abc
from numba import njit, generated_jit
from numba.core import cgutils, typing  # noqa
from numba.extending import overload, register_jitable, intrinsic  # noqa
from numba.experimental import jitclass
from numba.np.numpy_support import as_dtype
import numba as nb
import numpy as np


class Term:
    @abc.abstractclassmethod
    def from_args(cls, index, *args):
        pass

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


class Brightness(Term):
    args = ["stokes", "chan_freq"]
    output = ["brightness"]

    @classmethod
    def from_args(cls, index, *args):
        spec = [(n, args[i]) for i, n in zip(index, cls.args)]
        assert len(spec) == len(cls.args)
        brightness_type = cls.result_type(*(a for _, a in spec))
        spec.append(("brightness", brightness_type[:, :, :]))

        @jitclass(spec=spec)
        class klass:
            def __init__(self, stokes, chan_freq):
                nsrc, ncorr = stokes.shape
                nchan, = chan_freq.shape
                self.stokes = stokes
                self.chan_freq = chan_freq
                self.brightness = np.empty(
                    (nsrc, nchan, ncorr), brightness_type)

            def setup(self):
                nsrc, nchan, ncorr = self.brightness.shape
                stokes = self.stokes
                chan_freq = self.chan_freq
                brightness = self.brightness
                two = stokes.dtype.type(2)
                C = 2.0*np.pi / 3e8
                assert ncorr == 4

                for s in range(nsrc):
                    XX = (stokes[s, 0] + stokes[s, 3]) / two
                    XY = (stokes[s, 1] + stokes[s, 2]) / two
                    YX = (stokes[s, 1] - stokes[s, 2]) / two
                    YY = (stokes[s, 0] - stokes[s, 3]) / two

                    for f in range(nchan):
                        brightness[s, f, 0] = XX * C * chan_freq[f]
                        brightness[s, f, 1] = XY * C * chan_freq[f]
                        brightness[s, f, 2] = YX * C * chan_freq[f]
                        brightness[s, f, 3] = YY * C * chan_freq[f]

        return klass


class Phase(Term):
    args = ["lm",  "uvw", "chan_freq"]
    output = ["phase"]

    @classmethod
    def from_args(cls, index, *args):
        spec = list((n, args[i]) for i, n in zip(index, cls.args))
        assert len(spec) == len(cls.args)
        phase_type = cls.result_type(*(a for _, a in spec))

        spec.append(("phase", phase_type[:, :, :]))

        @jitclass(spec=spec)
        class klass:
            def __init__(self, lm, uvw, chan_freq):
                nsrc, _ = lm.shape
                nrow, _ = uvw.shape
                nchan, = chan_freq.shape
                self.lm = lm
                self.uvw = uvw
                self.chan_freq = chan_freq
                self.phase = np.empty((nsrc, nrow, nchan), dtype=phase_type)

            def setup(self):
                lm = self.lm
                uvw = self.uvw

                nsrc = lm.shape[0]
                nrow = uvw.shape[0]
                one = lm.dtype.type(1)
                zero = lm.dtype.type(0)
                real_phase = self.phase

                for source in range(nsrc):
                    l = lm[source, 0]  # noqa
                    m = lm[source, 1]
                    n = one - l**2 - m**2
                    n = np.sqrt(n) if n >= zero else zero

                    for row in range(nrow):
                        u = uvw[row, 0]
                        v = uvw[row, 1]
                        w = uvw[row, 2]

                        real_phase[source, row] = l*u + m*v + (n - one)*w

            def sampler(self):
                def sample(r, t, a1, a2, c, args):
                    pass

                return sample, (self.lm, self.uvw, self.chan_freq)

        return klass


def klass_factory(terms, args, term_arg_inds):
    klasses = tuple(T.from_args(i, *args)
                    for T, i in zip(terms, term_arg_inds))

    @intrinsic
    def term_factory(typingctx, args):
        # We return a Tuple of Term classes
        cls_types = [T.class_type for T in klasses]
        obj_types = [T.instance_type for T in cls_types]
        return_type = nb.types.Tuple(obj_types)
        sig = return_type(args)

        def codegen(context, builder, signature, args):
            tuple_type = signature.args[0]
            return_type = signature.return_type

            # Allocate an empty output tuple
            llvm_tuple_type = context.get_value_type(return_type)
            tup = cgutils.get_null_value(llvm_tuple_type)

            for i in range(return_type.count):
                cls = cls_types[i]
                idx = term_arg_inds[i]

                sig_args = tuple(tuple_type[j] for j in idx)
                call_sig = cls.get_call_type(
                    context.typing_context, sig_args, {})
                fn = context.get_function(cls, call_sig)

                fn_args = tuple(builder.extract_value(args[0], j) for j in idx)
                instance = fn(builder, fn_args)
                tup = builder.insert_value(tup, instance, i)

                cls_method = cls.jit_methods["setup"]
                disp_type = nb.types.Dispatcher(cls_method)
                call_sig = disp_type.get_call_type(
                    context.typing_context, (cls.instance_type,), {})
                setup_fn = context.get_function(disp_type, call_sig)
                empty = setup_fn(builder, (instance, ))  # noqa: F841

            # Return the tuple
            return tup

        return sig, codegen

    return term_factory


class rime_factory:
    def __init__(self):
        terms = [Phase, Brightness]
        args = list(sorted(set(a for t in terms for a in t.args)))
        arg_map = {a: i for i, a in enumerate(args)}
        term_arg_inds = tuple(tuple(arg_map[a] for a in t.args) for t in terms)

        @generated_jit(nopython=True, nogil=True, cache=True)
        def function(*args):
            term_factory = klass_factory(terms, args[0], term_arg_inds)

            def impl(*args):
                terms = term_factory(args)  # noqa: F841

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
            self.impl(*args)


if __name__ == "__main__":
    @jitclass(spec=[("a", nb.int32)])
    class A:
        def __init__(self, a):
            self.a = a

        def sampler(self):
            def impl(r, t, a1, a2, c):
                return r

            return impl

    @intrinsic
    def test(typingctx, args):
        if not isinstance(args, nb.types.BaseTuple):
            raise nb.core.errors.TypingError(f"{args} is not a Tuple")

        sig = A.class_type.instance_type(args)
        # sig = nb.types.none(args)
        print(sig)

        def codegen(context, builder, signature, args):
            sig_args = tuple(a for a in signature.args[0])
            cls = A.class_type
            call_sig = cls.get_call_type(context.typing_context, sig_args, {})

            fn = context.get_function(cls, call_sig)

            print(f"call_sig {call_sig}")

            return fn(builder, (builder.extract_value(args[0], 0),))

            import pdb
            pdb.set_trace()

            ret_type = context.get_value_type(signature.return_type)
            return cgutils.get_null_value(ret_type)

        return sig, codegen

    @njit
    def fn(i):
        # a = A(*(1,))
        # print(a)
        a = test((i,))
        impl = a.sampler()
        return impl(10, 0, 0, 0, 0)

    import pdb
    pdb.set_trace()
    print(fn(11))

    rime = rime_factory()
    lm = np.random.random(size=(10, 2))
    uvw = np.random.random(size=(5, 3))
    chan_freq = np.linspace(.856e9, 2*.859e9, 4)
    stokes = np.random.random(size=(10, 4))

    out = rime(lm=lm, uvw=uvw, chan_freq=chan_freq, stokes=stokes)
    print(out)
