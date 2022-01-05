from collections import OrderedDict, defaultdict
from collections.abc import Mapping


class ArgumentPack(Mapping):
    def __init__(self, names, types, index):
        self.pack = OrderedDict((n, (t, i)) for n, t, i
                                in zip(names, types, index))

    def copy(self):
        names, types, index = zip(*((k, t, i) for k, (t, i)
                                    in self.pack.items()))
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


class ArgumentDependencies:
    REQUIRED_ARGS = ("time", "antenna1", "antenna2", "feed1", "feed2")
    KEY_ARGS = ("utime", "time_index",
                "uantenna", "antenna1_index", "antenna2_index",
                "ufeed", "feed1_index", "feed2_index")

    def __init__(self, arg_names, terms, transformers):
        if not set(self.REQUIRED_ARGS).issubset(arg_names):
            raise ValueError(
                f"{set(self.REQUIRED_ARGS) - set(arg_names)} "
                f"missing from arg_names")

        self.names = arg_names
        self.terms = terms
        self.transformers = transformers

        self.desired = desired = defaultdict(list)
        self.optional = optional = defaultdict(list)
        self.maybe_create = maybe_create = defaultdict(list)

        for term in terms:
            for a in term.ARGS:
                desired[a].append(term)
            for k, d in term.KWARGS.items():
                optional[k].append((term, d))

        for transformer in transformers:
            for o in transformer.OUTPUTS:
                maybe_create[o].append(transformer)

        od, cc = self._resolve_arg_dependencies()
        self.optional_defaults = od
        self.can_create = cc

        # Determine a canonical set of valid inputs
        # We start with the desired and required arguments
        self.valid_inputs = (set(desired.keys()) |
                             set(self.REQUIRED_ARGS) |
                             set(optional.keys()))

        # Then, for each argument than can be created
        # we add the transformer arguments and remove
        # the arguments to create
        for arg, transformer in cc.items():
            self.valid_inputs.update(transformer.ARGS)
            self.valid_inputs.remove(arg)

    def _resolve_arg_dependencies(self):
        # KEY_ARGS will be created
        supplied_args = set(self.names) | set(self.KEY_ARGS)
        missing = set(self.desired.keys()) - supplied_args
        available_args = set(self.names) | supplied_args
        failed_transforms = defaultdict(list)
        can_create = {}

        # Try create missing argument with transformers
        for arg in list(missing):
            # We already know how to create it
            if arg in can_create:
                continue

            # We don't know how to create
            if arg not in self.maybe_create:
                continue

            for transformer in self.maybe_create[arg]:
                # We didn't have the arguments, make a note of this
                if not set(transformer.ARGS).issubset(available_args):
                    failed_transforms[arg].append(
                        (transformer, set(transformer.ARGS)))
                    continue

            # The transformer can create arg
            if arg not in failed_transforms:
                can_create[arg] = transformer
                missing.remove(arg)

        # Fail if required arguments are missing
        for arg in missing:
            terms_wanting = self.desired[arg]
            err_msgs = []
            err_msgs.append(f"{terms_wanting} need(s) '{arg}'.")

            if arg in failed_transforms:
                for transformer, needed in failed_transforms[arg]:
                    err_msgs.append(f"{transformer} can create {arg} "
                                    f"but needs {needed}, of which "
                                    f"{needed - supplied_args} is missing "
                                    f"from the input arguments.")

            raise ValueError("\n".join(err_msgs))

        opt_defaults = {}

        for transformer in can_create.values():
            for k, d in transformer.KWARGS.items():
                self.optional[k].append((transformer, d))

        for k, v in self.optional.items():
            _, defaults = zip(*v)
            defaults = set(defaults)

            if len(defaults) != 1:
                raise ValueError(f"Multiple terms: {self.terms} have "
                                 f"contradicting definitions for "
                                 f"{k}: {defaults}")

            opt_defaults[k] = defaults.pop()

        for name in supplied_args:
            opt_defaults.pop(name, None)

        return opt_defaults, can_create
