from donfig import Config


class AfricanusConfig(Config):
    def numba_parallel(self, key):
        value = self.get(key, False)

        if value is False:
            return {'parallel': False}
        elif value is True:
            return {'parallel': True}
        elif isinstance(value, dict):
            value['parallel'] = True
            return value
        else:
            raise TypeError("key %s (%s) is not a bool or a dict",
                            key, value)





config = AfricanusConfig("africanus")
