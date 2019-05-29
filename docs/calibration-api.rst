-----------
Calibration
-----------

This module provides basic radio interferometry
calibration utilities. Calibration is the
process of estimating the :math:`2\times2`
Jones matrices which describe transformations
of the signal as it propagates from source to
observer. Currently, all utilities assume a
discretised form of the radio interferometer
measurement equation (RIME) as described in
:ref:`rime-api-anchor`.

Numpy
~~~~~

.. currentmodule:: africanus.calibration

.. autosummary::
    jhj_and_jhr
    phase_only_GN

.. autofunction:: jhj_and_jhr
.. autofunction:: phase_only_GN
