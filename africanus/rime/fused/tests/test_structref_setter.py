import numba
from numba.experimental import structref
from numba.extending import intrinsic
from numba.core import types, errors

import numpy as np


@structref.register
class StateStructRef(types.StructRef):
    def preprocess_fields(self, fields):
        """ Disallow literal types in field definitions """
        return tuple((n, types.unliteral(t)) for n, t in fields)


def test_structref_setter():
    @intrinsic
    def constructor(typingctx, arg_tuple):
        if not isinstance(arg_tuple, types.BaseTuple):
            raise errors.TypingError(f"{arg_tuple} must be a Tuple")

        # Determine the fields for the StructRef
        names = [f"arg_{i}" for i in range(len(arg_tuple))]
        fields = list((name, typ) for name, typ in zip(names, arg_tuple))
        state_type = StateStructRef(fields)

        sig = state_type(arg_tuple)

        def codegen(context, builder, signature, args):
            def make_struct():
                """ Allocate the structure """
                return structref.new(state_type)

            state = context.compile_internal(builder, make_struct,
                                             state_type(), [])

            # Now assign each argument
            U = structref._Utils(context, builder, state_type)
            data_struct = U.get_data_struct(state)

            for i, name in enumerate(names):
                value = builder.extract_value(args[0], i)
                value_type = signature.args[0][i]
                field_type = state_type.field_dict[name]
                casted = context.cast(builder, value,
                                      value_type, field_type)
                old_value = getattr(data_struct, name)
                context.nrt.incref(builder, value_type, casted)
                context.nrt.decref(builder, value_type, old_value)
                setattr(data_struct, name, casted)

            return state

        return sig, codegen

    @numba.njit
    def fn(*args):
        s = constructor(args)
        print(s.arg_0)
        print(s.arg_1)
        print(s.arg_2)
        return s.arg_2

    args = (2, "b", np.arange(10))
    assert np.array_equal(fn(*args), args[2])
