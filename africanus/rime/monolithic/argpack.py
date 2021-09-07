from collections import OrderedDict
from collections.abc import Mapping

from numba.core import types


class ArgumentPack(Mapping):
    def __init__(self, names, types, index):
        self.pack = OrderedDict((n, (t, i)) for n, t, i
                                in zip(names, types, index))

    def copy(self):
        names, types, index = zip(*((k, t, i) for k, (t, i) in self.pack.items()))
        return ArgumentPack(names, types, index)

    def pop(self, key):
        return self.pack.pop(key)

    def type(self, key):
        return self.pack[key][0]

    def index(self, key):
        return self.pack[key][1]

    def types(self, *keys):
        return tuple(self.pack[k][0] for k in keys)

    def indices(self, *keys):
        return tuple(self.pack[k][1] for k in keys)

    def type_dict(self, *keys):
        return {k: self.pack[k][0] for k in keys}

    def index_dict(self, *keys):
        return {k: self.pack[k][1] for k in keys}

    def __getitem__(self, key):
        return self.pack[key]

    def __iter__(self):
        return iter(self.pack)

    def __len__(self):
        return len(self.pack)


def pack_arguments(terms, args):
    expected_args = set(a for t in terms for a in t.ARGS)
    expected_args = list(sorted(expected_args))
    potential_kwargs = {k: v for t in terms for
                        k, v in t.KWARGS.items()}

    n = len(expected_args)
    starargs = args[:n]
    kwargpairs = args[n:]

    if len(starargs) < n:
        raise ValueError(f"Insufficient arguments supplied to RIME "
                         f"Given the term configuration, the following "
                         f"arguments are required: {expected_args}")

    if len(kwargpairs) % 2 != 0:
        raise ValueError(f"len(kwargs) {len(kwargpairs)} is not "
                         f"divisible by 2")

    arg_map = {a: i for i, a in enumerate(expected_args)}

    names = []
    arg_types = []
    index = []

    for name, typ in zip(expected_args, args):
        names.append(name)
        arg_types.append(typ)
        index.append(arg_map[name])

    it = zip(kwargpairs[::2], kwargpairs[1::2])
    missing_kwargs = potential_kwargs.copy()

    for i, (name, typ) in enumerate(it):
        if not isinstance(name, types.StringLiteral):
            raise ValueError(f"{name} must be a StringLiteral")

        name = name.literal_value

        try:
            del missing_kwargs[name]
        except KeyError:
            continue

        names.append(name)
        arg_types.append(typ)
        index.append(2*i + 1 + n)

    for name, default in missing_kwargs.items():
        names.append(name)
        arg_types.append(types.Omitted(default))
        index.append(-1)

    return ArgumentPack(names, arg_types, index)
