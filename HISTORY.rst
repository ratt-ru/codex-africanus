=======
History
=======

0.1.3 (2018-03-28)
------------------
* requires_optional accepts ImportError's for a
  better debugging experience (:pr:`62`)
* cupy implementation of the `phase_delay` kernel (:pr:`59`)
* Correct `phase_delay` argument ordering (:pr:`57`)
* Support dask for `radec_to_lmn` and `lmn_to_radec`. Also add support
  for `radec_to_lm` and `lm_to_radec` (:pr:`56`)
* Added test for dft to test if image space covariance
  is symmetric(:pr:`55`)
* Correct Parallactic Angle Computation (:pr:`49`)
* Enhance visibility predict (:pr:`50`)
* Fix Kaiser Bessel filter and taper (:pr:`48`)
* Stokes/Correlation conversion (:pr:`41`)
* Fix gridding examples (:pr:`43`)
* Add simple dask gridder example (:pr:`42`)
* Implement Kaiser Bessel filter (:pr:`38`)
* Implement W-stacking gridder/degridder (:pr:`38`)
* Use 2D filters by default (:pr:`37`)
* Fixed bug in im_to_vis. Added more tests for im_to_vis.
  Removed division by :math:`n` since it is trivial to reinstate
  after the fact. (:pr:`34`)
* Move numba implementations out of API functions. (:pr:`33`)
* Zernike Polynomial Direction Dependent Effects (:pr:`18`, :pr:`30`)
* Added division by :math:`n` to DFT.
  Fixed dask chunking issue.
  Updated test_vis_to_im_dask (:pr:`29`).
* Implement RIME visibility predict (:pr:`24`, :pr:`25`)
* Direct Fourier Transform (:pr:`19`)
* Parallactic Angle computation (:pr:`15`)
* Implement Feed Rotation term (:pr:`14`)
* Swap gridding correlation dimensions (:pr:`13`)
* Implement Direction Dependent Effect beam cubes (:pr:`12`)
* Implement Brightness Matrix Calculation (:pr:`9`)
* Implement RIME Phase Delay term (:pr:`8`)
* Support user supplied grids (:pr:`7`)
* Add dask wrappers to the gridder and degridder (:pr:`4`)
* Add weights to gridder/degridder and remove PSF function (:pr:`2`)

0.1.2 (2018-03-28)
------------------

* First release on PyPI.
