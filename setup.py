#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""The setup script."""

import os
from setuptools import setup, find_packages

# Import requirements
from africanus.install.requirements import (requirements,
                                            extras_require,
                                            setup_requirements,
                                            test_requirements)

with open('README.rst') as readme_file:
    readme = readme_file.read()

setup(
    author="Simon Perkins",
    author_email='sperkins@ska.ac.za',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Natural Language :: English',
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
    long_description=readme,
    long_description_content_type='text/x-rst',
    include_package_data=True,
    keywords='codex-africanus',
    name='codex-africanus',
    packages=find_packages(),
    python_requires=">=3.5",
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ska-sa/codex-africanus',
    version='0.2.0',
    zip_safe=False,
)
