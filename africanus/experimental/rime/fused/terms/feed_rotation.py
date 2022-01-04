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
        linear = self.feed_type == "linear"

        def feed_rotation(state, s, r, t, f1, f2, a1, a2, c):
            a = a1 if left else a2
            f = f1 if left else f2
            sin_0 = state.parangle_sincos[t, f, a, 0, 0]
            cos_0 = state.parangle_sincos[t, f, a, 0, 1]
            sin_1 = state.parangle_sincos[t, f, a, 1, 0]
            cos_1 = state.parangle_sincos[t, f, a, 1, 1]

            # https://github.com/ska-sa/codex-africanus/issues/191#issuecomment-963089540
            if linear:
                return (cos_0, -sin_0, sin_1, cos_1)
            else:
                # e^{ix} = cos(x) + i.sin(x)
                cos_0 = 0.5 * cos_0
                sin_0j = 0.5 * sin_0*1j
                cos_1 = 0.5 * cos_1
                sin_1j = 0.5 * sin_1*1j

                return (
                    cos_0 + sin_0j + cos_1 + sin_1j,
                    -cos_0 - sin_0j - (-cos_1 - sin_1j),
                    cos_0 + sin_0j - (cos_1 + sin_1j),
                    -cos_0 - sin_0j + (-cos_1 - sin_1j))

        return feed_rotation
