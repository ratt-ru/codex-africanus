from africanus.experimental.rime.fused.terms.core import Term


class FeedRotation(Term):
    def __init__(self, configuration, feed_type):
        if configuration not in {"left", "right"}:
            raise ValueError(f"FeedRotation configuration must "
                             f"be either 'left' or 'right'. "
                             f"Got {configuration}")

        if feed_type not in {"linear", "circular"}:
            raise ValueError(f"FeedRotation feed_type must be "
                             f"either 'linear' or 'circular'. "
                             f"Got {feed_type}")

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
        linear = self.feed_type == "linear"

        def feed_rotation(state, s, r, t, f1, f2, a1, a2, c):
            a = a1 if left else a2
            f = f1 if left else f2
            sin = state.parangle_sincos[t, f, a, ra, 0]
            cos = state.parangle_sincos[t, f, a, ra, 1]

            return ((sin, cos, -sin, cos) if linear else
                    (cos - sin*1j, 0 + 0*1j, 0 + 0*1j, cos + sin*1j))

        return feed_rotation
