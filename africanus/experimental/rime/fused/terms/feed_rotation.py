from africanus.experimental.rime.fused.terms.core import Term


class FeedRotation(Term):
    """Feed Rotation Term"""
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

    def init_fields(self, typingctx, feed_parangle):
        def dummy(feed_parangle):
            pass

        return [], dummy

    def dask_schema(self, feed_parangle):
        return {}

    def sampler(self):
        left = self.configuration == "left"
        linear = self.feed_type == "linear"

        def feed_rotation(state, s, r, t, f1, f2, a1, a2, c):
            a = state.antenna1_index[r] if left else state.antenna2_index[r]
            f = state.feed1_index[r] if left else state.feed2_index[r]
            sin_a = state.feed_parangle[t, f, a, 0, 0]
            cos_a = state.feed_parangle[t, f, a, 0, 1]
            sin_b = state.feed_parangle[t, f, a, 1, 0]
            cos_b = state.feed_parangle[t, f, a, 1, 1]

            # https://casa.nrao.edu/aips2_docs/notes/185/node6.html
            if linear:
                return cos_a, sin_a, -sin_b, cos_b
            else:
                # e^{ix} = cos(x) + i.sin(x)
                return (
                    0.5*((cos_a + cos_b) - (sin_a + sin_b)*1j),
                    0.5*((cos_a - cos_b) + (sin_a - sin_b)*1j),
                    0.5*((cos_a - cos_b) - (sin_a - sin_b)*1j),
                    0.5*((cos_a + cos_b) + (sin_a + sin_b)*1j))

        return feed_rotation
