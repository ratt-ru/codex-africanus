import abc

from numba.core import types, typing


class ArgumentTransformer:
    @abc.abstractmethod
    def dask_schema(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def fields(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def initialiser(self, *args, **kwargs):
        raise NotImplementedError


class LMTransformer(ArgumentTransformer):
    def fields(self, radec, phase_centre):
        if not isinstance(radec, types.Array) or radec.ndim != 2:
            raise ValueError(f"{radec} must be a (source, radec) array")

        if not isinstance(phase_centre, types.Array) or radec.ndim != 1:
            raise ValueError(f"{phase_centre} must be a 1D array")

        ctx = typing.Context()
        dt = ctx.unify_types(radec.dtype, phase_centre.dtype)

        return [("lm", types.Array(dt, radec.ndim, radec.layout))]

    def initialiser(self, radec, phase_centre):
        pass
