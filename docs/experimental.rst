.. _experimental-fused-rime-api-anchor:

-----------------------------------------------
Fused Radio Interferometer Measurement Equation
-----------------------------------------------

Radio Interferometer Measurement Equation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Radio Interferometer Measurement Equation (RIME)
describes the response of an interferometer to a sky model.
As described in `A full-sky Jones formalism <rime_paper_i_>`_,
a RIME could be written as follows:

.. _fused-rime-equation-anchor:

.. math::

    V_{pq} = G_{p} \left(
        \sum_{s} E_{ps} L_{p} K_{ps}
        B_{s}
        K_{qs}^H L_{q}^H E_{qs}^H
        \right) G_{q}^H

where for antenna :math:`p` and :math:`q`, and source :math:`s`:

* :math:`G_{p}` represents direction-independent effects.

* :math:`E_{ps}` represents direction-dependent effects.

* :math:`L_{p}` represents the feed rotation.

* :math:`K_{ps}` represents the phase delay term.

* :math:`B_{s}` represents the brightness matrix.

The RIME is more formally described in the following four papers:

* `I. A full-sky Jones formalism <rime_paper_i_>`_

* `II. Calibration and direction-dependent effects <rime_paper_ii_>`_

* `III. Addressing direction-dependent effects in 21cm WSRT observations of 3C147 <rime_paper_iii_>`_

* `IV. A generalized tensor formalism <rime_paper_iv_>`_

.. _rime_paper_i: https://arxiv.org/abs/1101.1764
.. _rime_paper_ii: https://arxiv.org/abs/1101.1765
.. _rime_paper_iii: https://arxiv.org/abs/1101.1768
.. _rime_paper_iv: https://arxiv.org/abs/1106.0579


The Fused RIME
~~~~~~~~~~~~~~

The RIME poses a number of implementation challenges which
focus on flexibility, speed and ease of use.

Firstly, the RIME can be composed of many terms representing
various physical effects.
It is useful for scientist to be able to specify many different
terms in the above :ref:`Equation <fused-rime-equation-anchor>`,
for example.

Secondly, the computational complexity of the RIME `O(S x V)` where S
is the number of source and V is the number of visibilities.
This is computionationally expensive relative to degridding
strategies.

Thirdly, it should be as easy as possible to define the RIME,
but not at the cost of the previous two constraints.

The Fused RIME therefore implements a "RIME Compiler" using
`Numba <https://numba.pydata.org/_>`_ for speed, which compiles
a RIME Specification defined by a number of `Terms` into
a single, optimal unit of execution.

.. _experimental-fused-rime-example-anchor:

A Simple Example
~~~~~~~~~~~~~~~~

In the following example, we will define a simple RIME using the
Fused RIME API to define terms for computing:

1. the Phase Delay.
2. the Brightness Matrix.

The RIME Specification
++++++++++++++++++++++

The specification for this RIME is as follows:

.. code-block:: python

    rime_spec = RimeSpecification("(Kpq, Bpq): [I,Q,U,V] -> [XX,XY,YX,YY]",
                                  terms={"K": Phase})

``(Kpq, Bpq)`` specifies the onion including the Phase Delay and
Brightness more formally defined
:ref:`here <experimental-fused-rime-api-anchor>`, while the
the ``pq`` in both terms signifies that they are calculated per-baseline.
``[I,Q,U,V] -> [XX,XY,YX,YY]`` defines the stokes to correlation conversion
within the RIME and also identifies whether the RIME is handling linear
or circular feeds. Finally :code:`terms={"K": Phase}` indicates that the
K term is implemented as a custom Phase term, described in the next section.

Custom Phase Term
+++++++++++++++++

Within the RIME, each term is *sampled* at an individual
source, row and channel.

Therefore each term must provide a sampling function that will
provide the necessary data for multiplication within the RIME.
Consider the following Phase Term:

.. code-block:: python

    from africanus.experimental.rime.fused.terms.core import Term

    class Phase(Term):
        def sampler(self):
            def phase_sample(state, s, r, t, f1, f2, a1, a2, c):
                p = state.real_phase[s, r] * state.chan_freq[c]
                return np.cos(p) + np.sin(p)*1j

            return phase_sample

This may look simple: we compute the complex phase by multiplying
the real phase at each source and row by the channel frequency
and return the complex exponential of this value.

However, questions remain: What is the `state` object and how
do we know that the `real_phase` and `chan_freq` are members?
To answer this, we must define (and understand) a second method
defined on the `Phase` term, called `init_fields`.

.. code-block:: python

    import numba
    from africanus.experimental.rime.fused.terms.core import Term

    class Phase(Term)
        def init_fields(self, typingctx, lm, uvw, chan_freq):
            # Given the numba types of the lm, uvw and chan_freq
            # arrays, derive a unified output numba type
            numba_type = typingctx.unify_types(lm.dtype,
                                               uvw.dtype,
                                               chan_freq.dtype)

            # Define the type of new fields on the state object
            # in this case a 2D Numba array with dtype numba_type
            fields = [("real_phase", numba_type[:, :])]

            def real_phase(lm, uvw, chan_freq):
                """Compute the real_phase upfront, instead of in
                the sampling function"""
                real_phase = np.empty((lm.shape[0], uvw.shape[0]), numba_type)

                for s in range(lm.shape[0]):
                    l, m = lm[s]
                    n = 1.0 - l**2 - m**2
                    n = np.sqrt(0.0 if n <= 0.0 else n) - 1.0

                    for r in range(uvw.shape[0]):
                        u, v, w = uvw[r]
                        real_phase[s, r] = -2.0*np.pi*(l*u + m*v + n*w)/3e8

                return real_phase

            # Return the new field definition and
            # the function for creating it
            return fields, real_phase

``init_fields`` serves multiple purposes:

1. It requests input for the Phase term.
   The above definition of ``init_fields`` signifies
   that the Phase term desires the ``lm``, ``uvw`` and
   ``chan_freq`` arrays.
   Additionally, these arrays will be stored on the ``state``
   object provided to the sampling function.

2. It supports reasoning about Numba types in a manner
   similar to :func:`numba.generated_jit`.
   The ``lm``, ``uvw`` and ``chan_freq``
   arguments contain the Numba types of the variables supplied
   to the RIME, while the ``typingctx`` argument contains a Numba
   Typing Context which can be useful for reasoning about
   these types.
   For example
   :code:`typingctx.unify_types(lm.dtype, uvw.dtype, chan_freq.dtype)`
   returns a type with sufficient precision, given the input types,
   similar to :func:`numpy.result_type`.

3. It allows the user to define new fields, as
   well as a function for defining those fields
   on the ``state`` object.
   The above definition of ``init_fields`` returns
   a list of :code:`(name, type)` tuples defining
   the new field names and their types, while
   :code:`real_phase` defines the creation of
   this new field.

   This is useful for optimising the sampling function
   by pre-computing values. For example, it is wasteful to
   compute the real phase for each source, row and
   channel.

Returning to our definition of the Phase Term sampling function,
we can see that it uses the new field ``real_phase`` defined in
``init_fields``, as well as the ``chan_freq`` array requested
in ``init_fields`` to compute a complex exponential.

.. code-block:: python

    class Phase(Term):
        def sampler(self):
            def phase_sample(state, s, r, t, f1, f2, a1, a2, c):
                p = state.real_phase[s, r] * state.chan_freq[c]
                return np.cos(p) + np.sin(p)*1j

            return phase_sample

We then invoke the RIME by passing in the :class:`RimeSpecification`, as
well as a dataset containing the required arguments:

.. code-block:: python

    from africanus.experimental.rime.fused.core import rime
    import numpy as np

    dataset = {
        "lm": np.random.random((10, 2))*1e-5,
        "uvw": np.random.random((100, 3))*1e5,
        "chan_freq:" np.linspace(.856e9, 2*.856e9, 16),
        ...,
        "stokes": np.random.random((10, 4)),
        # other required data
    }

    rime_spec = RimeSpecification("(Kpq, Bpq)", terms={"K": Phase})
    model_visibilities = rime(rime_spec, dataset)

API
~~~

.. currentmodule:: africanus.experimental.rime.fused.specification

.. autoclass:: RimeSpecification
    :exclude-members: equation_bits, flatten_eqn


.. currentmodule:: africanus.experimental.rime.fused.terms.core

.. py:class:: Term

    Base class for Terms which describe parts of the Fused RIME.
    Implementors of a RIME Term should inherit from it.

    A Term is an object that defines how a term in the RIME should
    be sampled to produces the Jones Terms that make up the RIME.
    It therefore defines a sampling function, which in turn
    depends on arbitrary inputs for performing the sampling.

    A high degree of flexibility and leeway is afforded when
    implementing a Term. It might be implemented by merely indexing
    an array of Jones Matrices, or by implementing some computational
    model describing the Jones Terms.

    .. code-block:: python

        class Phase(Term):
            def __init__(self, configuration):
                super().__init__(configuration)

    .. py:method:: Term.init_fields(self, typing_ctx, arg1, ..., argn, \
                                    kwarg1=None, ..., kwargn=None)

        Requests inputs to the RIME term, ensuring that they are
        stored on a ``state`` object supplied to the sampling function
        and allows for new fields to be initialised and stored on the
        ``state`` object.

        Requested inputs :code:`arg1...argn` are required to be passed
        to the Fused RIME by the caller and are supplied to ``init_fields``
        as Numba types. :code:`kwarg1...kwargn` are optional -- if omitted
        by the caller, their default types (and values)  will be supplied.

        ``init_fields`` should return a :code:`(fields, function)` tuple.
        ``fields`` should be a list of the form :code:`[(name, numba_type)]`, while
        ``function`` should be a function of the form
        :code:`fn(arg1, ..., argn, kwarg1=None, .., kwargn=None)`
        and should return the variables of the type defined
        in ``fields``. Note that it's signature therefore matches
        that of ``init_fields`` from after the ``typingctx``
        argument. See the
        :ref:`Simple Example <experimental-fused-rime-example-anchor>`.

        :param typingctx: A Numba typing context.
        :param arg1...argn: Required RIME inputs for this Term.
        :param kwarg1...kwargn: Optional RIME inputs for this Term. \
            Types here should be simple: ints, floats, complex numbers
            and strings are ideal.

        :rtype: A :code:`(fields, function)` tuple.

        .. warning::

            The ``function`` returned by ``init_fields`` must be compileable
            in Numba's
            `nopython <https://numba.pydata.org/numba-doc/latest/user/jit.html#nopython_>`_ mode.

    .. py:method:: Term.sampler(self)

        Return a sampling function of the following form:

        .. code-block:: python

            def sampler(self):
                def sample(state, s, r, t, f1, f2, a1, a2, c):
                    ...

            return sample

        :param state: A state object containing the inputs requested by
                      all ``Term`` objects in the RIME, as well as any
                      fields created by ``Term.init_fields``.
        :param s: Source index.
        :param r: Row index.
        :param t: Time index.
        :param f1: Feed 1 index.
        :param f2: Feed 2 index.
        :param a1: Antenna 1 index.
        :param a2: Antenna2 index.
        :param c: Channel index.

        :rtype: a scalar or a tuple of two scalars or a tuple of four scalars.

        .. warning::

            The sampling function returned by ``sampler`` must be compileable
            in Numba's
            `nopython <https://numba.pydata.org/numba-doc/latest/user/jit.html#nopython_>`_ mode.


.. currentmodule:: africanus.experimental.rime.fused.transformers.core

.. py:class:: Transformer

Numpy
~~~~~

.. currentmodule:: africanus.experimental.rime.fused.core

.. autosummary::
    rime

.. autofunction:: rime

Dask
~~~~

.. currentmodule:: africanus.experimental.rime.fused.dask

.. autosummary::
    rime

.. autofunction:: rime
