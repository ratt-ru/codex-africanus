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

- :math:`G_{p}` represents direction-independent effects.
- :math:`E_{ps}` represents direction-dependent effects.
- :math:`L_{p}` represents the feed rotation.
- :math:`K_{ps}` represents the phase delay term.
- :math:`B_{s}` represents the brightness matrix.

The RIME is more formally described in the following four papers:

- `I. A full-sky Jones formalism <rime_paper_i_>`_
- `II. Calibration and direction-dependent effects <rime_paper_ii_>`_
- `III. Addressing direction-dependent effects in 21cm WSRT observations of 3C147 <rime_paper_iii_>`_
- `IV. A generalized tensor formalism <rime_paper_iv_>`_

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
an optimal unit of execution.

A Simple example
~~~~~~~~~~~~~~~~

In the following example, we will define a simple RIME using the
Fused RIME API to define terms for computing:

1. the phase delay
2. the brightness matrix

Within the RIME, each term is *sampled* at individual
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

The purpose of `init_fields` is two-fold:

1. It acts as a request from the term for inputs.
   The above definition of `init_fields` signifies
   that the Phase term desires the `lm`, `uvw` and
   `chan_freq` arrays.
   Additionally, these arrays will be stored on the `state`
   object provided to the sampling function.

2. It allows the user to define new fields, as
   well as a function for defining those fields
   on the `state` object.
   The above definition of `init_fields` returns
   a list of :code:`(name, type)` tuples defining
   the new field names and their types, while
   :code:`real_phase` defines the creation of
   this new field.

   This is useful for optimising the sampling function
   by pre-computing values. For example, it is wasteful to
   compute the real phase for each source, row and
   channel.


API
~~~

Terms
+++++

.. currentmodule:: africanus.experimental.rime.fused.terms.core

.. autoclass:: Term

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
