# -*- coding: utf-8 -*-


# NOTE(sjperkins)
# Non standard library imports should be avoided,
# or should fail gracefully as functionality
# in these modules is called by setup.py
import os

# requirements
on_rtd = os.environ.get('READTHEDOCS') == 'True'

# Basic requirements containing no C extensions.
# This is necessary for building on RTD
requirements = ['appdirs >= 1.4.3',
                'decorator']

if not on_rtd:
    requirements += [
        # astropy breaks with numpy 1.15.3
        # https://github.com/astropy/astropy/issues/7943
        'numpy >= 1.14.0, != 1.15.3',
        'numba >= 0.46.0']

extras_require = {
    'cuda': ['cupy >= 5.0.0', 'jinja2 >= 2.10'],
    'dask': ['dask[array] >= 1.1.0'],
    'jax': ['jax == 0.1.27', 'jaxlib == 0.1.14'],
    'scipy': ['scipy >= 1.0.0'],
    'astropy': ['astropy >= 2.0.0, < 3.0; python_version <= "2.7"',
                'astropy >= 3.0; python_version >= "3.0"'],
    'python-casacore': ['python-casacore == 3.0.0'],
    'testing': ['pytest', 'flaky', 'pytest-flake8']
}

_non_cuda_extras = [er for n, er in extras_require.items() if n != "cuda"]
_all_extras = extras_require.values()

extras_require['complete'] = sorted(set(sum(_non_cuda_extras, [])))
extras_require['complete-cuda'] = sorted(set(sum(_all_extras, [])))

setup_requirements = []
test_requirements = (extras_require['testing'] +
                     extras_require['astropy'] +
                     extras_require['python-casacore'] +
                     extras_require['dask'] +
                     extras_require['scipy'])
