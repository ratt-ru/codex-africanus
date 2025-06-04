.. _rime-api-anchor:

-----------------------------------------
Radio Interferometer Measurement Equation
-----------------------------------------

Functions used to compute the terms of the
Radio Interferometer Measurement Equation (RIME).
It describes the response of an interferometer to a sky model.

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


Numpy
~~~~~

.. currentmodule:: africanus.rime

.. autosummary::
    predict_vis
    phase_delay
    parallactic_angles
    feed_rotation
    transform_sources
    beam_cube_dde
    zernike_dde
    wsclean_predict

.. autofunction:: predict_vis
.. autofunction:: phase_delay
.. autofunction:: parallactic_angles
.. autofunction:: feed_rotation
.. autofunction:: transform_sources
.. autofunction:: beam_cube_dde
.. autofunction:: zernike_dde
.. autofunction:: wsclean_predict

Cuda
~~~~

.. currentmodule:: africanus.rime.cuda

.. autosummary::
    predict_vis
    phase_delay
    feed_rotation
    beam_cube_dde

.. autofunction:: predict_vis
.. autofunction:: phase_delay
.. autofunction:: feed_rotation
.. autofunction:: beam_cube_dde


Dask
~~~~

.. currentmodule:: africanus.rime.dask

.. autosummary::
    predict_vis
    phase_delay
    parallactic_angles
    feed_rotation
    transform_sources
    beam_cube_dde
    zernike_dde
    wsclean_predict


.. autofunction:: predict_vis
.. autofunction:: phase_delay
.. autofunction:: parallactic_angles
.. autofunction:: feed_rotation
.. autofunction:: transform_sources
.. autofunction:: beam_cube_dde
.. autofunction:: zernike_dde
.. autofunction:: wsclean_predict
