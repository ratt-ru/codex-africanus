from africanus.experimental.rime.fused.terms.core import Term


class FeedRotation(Term):
    def __init__(self, configuration, feed_type):
        if configuration not in {"left", "right"}:
            raise ValueError(f"FeedRotation configuration must "
                             f"be either 'left' or 'right'. "
                             f"Got {configuration}")

        super().__init__(configuration)
        self.feed_type = feed_type

    def init_fields(self, typingctx, parangle_sincos):
        def dummy(parangle_sincos):
            pass

        return [], dummy

    def dask_schema(self, parangle_sincos):
        return {}

    def sampler(self):
        left = self.configuration == "left"
        ra = 0 if left else 1

        if self.feed_type == "linear":
            def feed_rotation(state, s, r, t, f1, f2, a1, a2, c):
                a = a1 if left else a2
                f = f1 if left else f2
                sin = state.parangle_sincos[t, f, a, ra, 0]
                cos = state.parangle_sincos[t, f, a, ra, 1]

                return sin, cos, -sin, cos
        elif self.feed_type == "circular":
            def feed_rotation(state, s, r, t, f1, f2, a1, a2, c):
                a = a1 if left else a2
                f = f1 if left else f2
                sin = state.parangle_sincos[t, f, a, ra, 0]
                cos = state.parangle_sincos[t, f, a, ra, 1]

                return cos - sin*1j, 0 + 0*1j, 0 + 0*1j, cos + sin*1j
        else:
            raise ValueError(f"Invalid feed_type {self.feed_type}")

        return feed_rotation
