-----------
Calibration
-----------

This module provides basic radio interferometry
calibration utilities. Calibration is the
process of estimating the :math:`2\times 2`
Jones matrices which describe transformations
of the signal as it propagates from source to
observer. Currently, all utilities assume a
discretised form of the radio interferometer
measurement equation (RIME) as described in
:ref:`rime-api-anchor`.

Calibration is usually divided into three
phases viz.

*   First generation calibration (1GC): using
    an external calibrator to infer the gains during
    the target observation. Sometimes also refered to
    as calibrator transfer

*   Second generation calibration (2GC): using
    a partially incomplete sky model to perform
    direction independent calibration. Also known
    as direction independent self-calibration.

*   Third generation calibration (3GC): using
    a partially incomplete sky model to perform
    direction dependent calibration. Also known
    as direction dependent self-calibration.

On top of these three phases, there are usually
three possible calibration scenarios. The first
is when both the Jones terms and the visibilities
are assumed to be diagonal. In this case the two
correlations can be calibrated separately and it
is refered to as :code:`diag-diag` calibration.
The second case is when the Jones matrices are
assumed to be diagonal but the visibility data
are full :math:`2\times 2` matrices. This is
refered to as :code:`diag` calibration. The final
scenario is when both the full :math:`2\times 2`
Jones matrices and the full :math:`2\times 2`
visibilities are used for calibration. This is
simply refered to as calibration. The specific
scenario is determined from the shapes of the input
gains and the input data.
Numpy
~~~~~

.. currentmodule:: africanus.calibration

.. autosummary::
    residual_vis
    correct_vis
    phase_only_gauss_newton

.. autofunction:: residual_vis
.. autofunction:: correct_vis
.. autofunction:: phase_only_gauss_newton
