.. highlight:: shell

============
Installation
============


Stable release
--------------

To install Codex Africanus, run this command in your terminal:

.. code-block:: console

    $ pip install codex-africanus

This is the preferred method to install Codex Africanus,
as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_
can guide you through the process.

By default, Codex Africanus will install with a minimal set of
dependencies, numpy and numba.

Further functionality can be enabled by installing extra requirements
as follows:

.. code-block:: console

    $ pip install codex-africanus[dask]
    $ pip install codex-africanus[scipy]
    $ pip install codex-africanus[astropy]
    $ pip install codex-africanus[python-casacore]


To install the complete set of dependencies for the CPU:

.. code-block:: console

    $ pip install codex-africanus[complete]

To install the complete set of dependencies including CUDA:

.. code-block:: console

    $ pip install codex-africanus[complete-cuda]


.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for Codex Africanus can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/ska-sa/codex-africanus

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/ska-sa/codex-africanus/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/ska-sa/codex-africanus
.. _tarball: https://github.com/ska-sa/codex-africanus/tarball/master
