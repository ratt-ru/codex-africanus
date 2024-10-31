#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from pathlib import Path
from setuptools import setup, find_packages

# Basic requirements containing no C extensions.
# This is necessary for building on RTD
requirements = [
    "appdirs >= 1.4.3",
    "decorator",
    "numpy >= 2.0.0",
    "numba >= 0.53.1",
    "setuptools",
]

extras_require = {
    "cuda": ["cupy >= 9.0.0", "jinja2 >= 2.10"],
    "dask": ["dask[array] >= 2.2.0"],
    "jax": ["jax >= 0.2.11", "jaxlib >= 0.1.65"],
    "scipy": ["scipy >= 1.4.0"],
    "astropy": ["astropy >= 4.0"],
    "python-casacore": ["python-casacore >= 3.6.1"],
    "ducc0": ["ducc0 >= 0.9.0"],
    "testing": ["pytest", "flaky"],
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


setup_requirements = ["setuptools"]
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
    author_email="sperkins@sarao.ac.za",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
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
    python_requires=">=3.10",
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/ska-sa/codex-africanus",
    version="0.3.7",
    zip_safe=False,
)
