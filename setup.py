#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import os
from pathlib import Path
from setuptools import setup, find_packages

# Import requirements
# requirements
on_rtd = os.environ.get("READTHEDOCS") == "True"

# Basic requirements containing no C extensions.
# This is necessary for building on RTD
requirements = ["appdirs >= 1.4.3", "decorator"]

if not on_rtd:
    requirements += [
        # astropy breaks with numpy 1.15.3
        # https://github.com/astropy/astropy/issues/7943
        "numpy >= 1.14.0, != 1.15.3",
        "numba >= 0.53.1"
    ]

extras_require = {
    "cuda": ["cupy >= 9.0.0", "jinja2 >= 2.10"],
    "dask": ["dask[array] >= 2.2.0"],
    "jax": ["jax >= 0.2.11", "jaxlib >= 0.1.65"],
    "scipy": ["scipy >= 1.4.0"],
    "astropy": ["astropy >= 4.0"],
    "python-casacore": ["python-casacore >= 3.4.0, != 3.5.0"],
    "ducc0": ["ducc0 >= 0.9.0"],
    "testing": ["pytest", "flaky", "pytest-flake8 >= 1.0.6", "flake8 >= 4.0.0, < 5.0.0"],
}

with open(str(Path("africanus", "install", "extras_require.py")), "w") as f:
    f.write("# flake8: noqa")
    f.write("extras_require = {\n")
    for k, v in extras_require.items():
        f.write("   '%s': %s,\n" % (k, v))
    f.write("}\n")

_non_cuda_extras = [er for n, er in extras_require.items() if n != "cuda"]
_all_extras = extras_require.values()

extras_require["complete"] = sorted(set(sum(_non_cuda_extras, [])))
extras_require["complete-cuda"] = sorted(set(sum(_all_extras, [])))


setup_requirements = []
test_requirements = (
    extras_require["testing"]
    + extras_require["astropy"]
    + extras_require["python-casacore"]
    + extras_require["dask"]
    + extras_require["scipy"]
    + extras_require["ducc0"]
)


with open("README.rst") as readme_file:
    readme = readme_file.read()

setup(
    author="Simon Perkins",
    author_email="sperkins@ska.ac.za",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="Radio Astronomy Building Blocks",
    extras_require=extras_require,
    install_requires=requirements,
    license="BSD-3-Clause",
    long_description=readme,
    long_description_content_type="text/x-rst",
    include_package_data=True,
    keywords="codex-africanus",
    name="codex-africanus",
    packages=find_packages(),
    python_requires=">=3.8",
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/ska-sa/codex-africanus",
    version="0.3.3",
    zip_safe=False,
)
