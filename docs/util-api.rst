---------
Utilities
---------

Command Line
~~~~~~~~~~~~

.. currentmodule:: africanus.util.cmdline

.. autosummary::
    parse_python_assigns

.. autofunction:: parse_python_assigns


Requirements Handling
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: africanus.util.requirements

.. autosummary::
    requires_optional

.. autofunction:: requires_optional

Shapes
~~~~~~

.. currentmodule:: africanus.util.shapes

.. autosummary::
    aggregate_chunks
    corr_shape

.. autofunction:: aggregate_chunks
.. autofunction:: corr_shape


Beams
~~~~~

.. currentmodule:: africanus.util.beams

.. autosummary::
    beam_filenames
    beam_grids


.. autofunction:: beam_filenames
.. autofunction:: beam_grids

Code
~~~~

.. currentmodule:: africanus.util.code

.. autosummary::
    format_code
    memoize_on_key

.. autofunction:: format_code
.. autoclass:: memoize_on_key


dask
~~~~

.. currentmodule:: africanus.util.dask_util

.. autosummary::
    EstimatingProgressBar

.. autoclass:: EstimatingProgressBar
    :members:
    :no-inherited-members:
    :exclude-members: register, unregister

CUDA
~~~~

.. currentmodule:: africanus.util.cuda

.. autosummary::
    grids

.. autofunction:: grids

Patterns
~~~~~~~~

.. currentmodule:: africanus.util.patterns

.. autosummary::
    Multiton
    LazyProxy
    LazyProxyMultiton

.. autoclass:: Multiton
    :exclude-members: __call__, mro
.. autoclass:: LazyProxy
.. autoclass:: LazyProxyMultiton
