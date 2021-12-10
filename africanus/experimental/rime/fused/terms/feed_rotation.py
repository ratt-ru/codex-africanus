from africanus.experimental.rime.fused.terms.core import Term


class FeedRotation(Term):
    def __init__(self, configuration, feed_type):
        super().__init__(configuration)
        self.feed_type = feed_type

    def init_fields(self, typingctx, parangle_sin_cos):
        def dummy(parangle_sin_cos):
            pass
        
        return [], dummy

    def dask_schema(self, parangle_sin_cos):
        return {}, {}

    def sampler(self):
        if self.configuration not in {"left", "right"}:
            raise ValueError(f"configuration {self.configuration} "
                             f"is not supported")

        left = self.configuration == "left"

        if self.feed_type == "linear":
            def feed_rotation(state, s, r, t, a1, a2, c):
                sin = state.parangle_sin_cos[t, a1 if left else a2, 0]
                cos = state.parangle_sin_cos[t, a1 if left else a2, 1]

                return sin, cos, -sin, cos
        elif self.feed_type == "circular":
            def feed_rotation(state, s, r, t, a1, a2, c):
                sin = state.parangle_sin_cos[t, a1 if left else a2, 0]
                cos = state.parangle_sin_cos[t, a1 if left else a2, 1]

                return cos - sin*1j, 0 + 0*1j, 0 + 0*1j, cos + sin*1j  
        else:
            raise ValueError(f"Invalid feed_type {self.feed_type}")

        return feed_rotation