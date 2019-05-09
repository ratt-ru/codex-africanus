#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import os
import sys
from setuptools import setup, find_packages

PY2 = sys.version_info[0] == 2

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

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
        'numba >= 0.43.0']

extras_require = {
    'cuda': ['cupy >= 5.0.0', 'jinja2 >= 2.10'],
    'dask': ['dask[array] >= 1.1.0'],
    'jax': ['jax == 0.1.27', 'jaxlib == 0.1.14'],
    'scipy': ['scipy >= 1.0.0'],
    'astropy': ['astropy >= 2.0.0, < 3.0.0' if PY2 else 'astropy >= 3.0.0'],
    'python-casacore': ['python-casacore >= 2.2.1'],
    'testing': ['pytest', 'pytest-runner']
}

_non_cuda_extras = [er for n, er in extras_require.items() if n != "cuda"]
_all_extras = extras_require.values()

extras_require['complete'] = sorted(set(sum(_non_cuda_extras, [])))
extras_require['complete-cuda'] = sorted(set(sum(_all_extras, [])))

setup_requirements = ['pytest-runner', ]
test_requirements = (['pytest'] +
                     extras_require['astropy'] +
                     extras_require['python-casacore'] +
                     extras_require['dask'] +
                     extras_require['scipy'])

setup(
    author="Simon Perkins",
    author_email='sperkins@ska.ac.za',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="Radio Astronomy Building Blocks",
    entry_points={
        'console_scripts': [
            'plot-filter=africanus.filters.plot_filter:main',
            'plot-taper=africanus.filters.plot_taper:main'],
    },
    extras_require=extras_require,
    install_requires=requirements,
    license="GNU General Public License v2",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='codex-africanus',
    name='codex-africanus',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ska-sa/codex-africanus',
    version='0.1.7',
    zip_safe=False,
)
