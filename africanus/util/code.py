# -*- coding: utf-8 -*-


from functools import wraps

try:
    from dask.utils import SerializableLock as Lock
except ImportError:
    from threading import Lock


class SingletonMixin(object):
    __singleton_lock = Lock()
    __singleton_instance = None

    @classmethod
    def instance(cls):
        if not cls.__singleton_instance:
            with cls.__singleton_lock:
                if not cls.__singleton_instance:
                    cls.__singleton_instance = cls()

        return cls.__singleton_instance


def format_code(code):
    """
    Formats some code with line numbers

    Parameters
    ----------
    code : str
        Code

    Returns
    -------
    str
        Code prefixed with line numbers
    """
    lines = ['']
    lines.extend(["%-5d %s" % (i, l) for i, l
                  in enumerate(code.split('\n'), 1)])
    return '\n'.join(lines)


class memoize_on_key(object):
    """
    Memoize based on a key function supplied by the user.
    The key function should return a custom key
    for memoizing the decorated function, based on the arguments
    passed to it.

    In the following example, the arguments required to generate
    the `_generate_phase_delay_kernel` function are the types of
    the `lm`, `uvw` and `frequency` arrays, as well as the number
    of correlations, `ncorr`.

    The supplied ``key_fn`` produces a unique key based on these types
    and the number of correlations, which is used to cache the
    generated function.

    .. code-block:: python

        def key_fn(lm, uvw, frequency, ncorrs=4):
            '''
            Produce a unique key for the arguments of
             _generate_phase_delay_kernel
            '''
            return (lm.dtype, uvw.dtype, frequency.dtype, ncorrs)

        _code_template = jinja2.Template('''
        #define ncorrs {{ncorrs}}

        __global__ void phase_delay(
            const {{lm_type}} * lm,
            const {{uvw_type}} * uvw,
            const {{freq_type}} * frequency,
            {{out_type}} * out)
        {
            ...
        }
        ''')

        _type_map = {
            np.float32: 'float',
            np.float64: 'double'
        }

        @memoize_on_key(key_fn)
        def _generate_phase_delay_kernel(lm, uvw, frequency, ncorrs=4):
            ''' Generate the phase delay kernel '''
            out_dtype = np.result_type(lm.dtype, uvw.dtype, frequency.dtype)
            code = _code_template.render(lm_type=_type_map[lm.dtype],
                                         uvw_type=_type_map[uvw.dtype],
                                         freq_type=_type_map[frequency.dtype],
                                         ncorrs=ncorrs)
            return cp.RawKernel(code, "phase_delay")
    """

    def __init__(self, key_fn):
        self._key_fn = key_fn
        self._lock = Lock()
        self._cache = {}

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = self._key_fn(*args, **kwargs)

            with self._lock:
                try:
                    return self._cache[key]
                except KeyError:
                    self._cache[key] = entry = fn(*args, **kwargs)
                    return entry

        return wrapper
