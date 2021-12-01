from collections import OrderedDict
import warnings

from numpy import ndarray


def freeze(arg):
    if isinstance(arg, set):
        return tuple(map(freeze, sorted(arg)))
    elif isinstance(arg, (tuple, list)):
        return tuple(map(freeze, arg))
    elif isinstance(arg, (dict, OrderedDict)):
        return frozenset((k, freeze(v)) for k, v in sorted(arg.items()))
    elif isinstance(arg, ndarray):
        warnings.warn(f"freezing ndarray of size {arg.nbytes} "
                      f"is probably inefficient")
        return freeze(arg.tolist())
    else:
        return arg
