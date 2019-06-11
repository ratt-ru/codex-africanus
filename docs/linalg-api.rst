--------------
Linear Algebra
--------------

This module contains specialised linear algebra
tools that are not currently available in the
:code:`python` standard scientific libraries.

Kronecker tools
---------------

A kronecker matrix is matrix that can be written
as a kronecker matrix of the individual matrices i.e.

.. math::
    K = K_0 \\otimes K_1 \\otimes K_2 \\otimes \\cdots
    
Matrices which exhibit this structure can exploit
properties of the kronecker product to avoid 
explicitly expanding the matrix :math:`K`. This
module implements some common linear algebra
operations which leverages this property for
computational gains and a reduced memory footprint.

Numpy
~~~~~

.. currentmodule:: africanus.linalg

.. autosummary::
    kron_matvec
    kron_cholesky

.. autofunction:: kron_matvec
.. autofunction:: kron_cholesky
