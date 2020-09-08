=======
History
=======

0.2.6 (2020-08-07)
------------------
* Add wrapper for ducc0.wgridder (:pr:204`)
* Correct Irregular Grid nesting in BeamAxes (:pr:`217`)

0.2.5 (2020-07-01)
------------------
* Convert WSClean Gaussian arcsecond and degree quantities to radians (:pr:`206`)
* Update classifiers and correct license in setup.py to BSD3 (:pr:`201`)

0.2.4 (2020-05-29)
------------------
* Support overriding the l and m axis sign in beam_grids (:pr:`199`)
* Upgrade to python-casacore 3.3.1 (:pr:`197`)
* Upgrade to jax 0.1.68 and jaxlib 0.1.47 (:pr:`197`)
* Upgrade to scipy 1.4.0 (:pr:`197`)
* Use github workflows (:pr:`196`, :pr:`197`, :pr:`198`, :pr:`199`)
* Make CASA parallactic angles thread-safe (:pr:`195`)
* Fix spectral model documentation (:pr:`190`), to match changes in (:pr:`189`)

0.2.3 (2020-05-14)
------------------
* Fix incorrect SPI calculation and make predict defaults MeqTree equivalent (:pr:`189`)
* Depend on pytest-flake8 >= 1.0.6 (:pr:`187`, :pr:`188`)
* MeqTrees Comparison Script Updates (:pr:`160`)
* Improve requirements handling (:pr:`187`)
* Use python-casacore wheels for travis testing, instead of kernsuite packages (:pr:`185`)

0.2.2 (2020-04-09)
------------------
* Add a dask Estimating Progress Bar (:pr:`182`, :pr:`183`)

0.2.1 (2020-04-03)
------------------
* Update trove to latest master commit (:pr:`178`)
* Added Cubic Spline support (:pr:`174`)
* Depend on python-casacore >= 3.2.0 (:pr:`172`)
* Drop Python 3.5 support and test Python 3.7 (:pr:`168`)
* Implement optimised WSClean predict (:pr:`166`, :pr:`167`, :pr:`177`, :pr:`179`, :pr:`180`, :pr:`181`)
* Simplify dask predict_vis code (:pr:`164`, :pr:`165`)
* Document and check weight shapes in simple gridder and degridder
  (:pr:`162`, :pr:`163`)
* Restructuring calibration module (:pr:`127`)
* Upgrade to numba 0.46.0, using new inlining functionality
  in the RIME and averaging code.
* Modified predict to be compatible with eidos fits headers (:pr:`158`)

0.2.0 (2019-09-30)
------------------
* Added standalone SPI fitter (:pr:`153`)
* Fail earlier and explain duplicate averaging rows (:pr:`155`)
* CUDA Beam Implementation (:pr:`152`)
* Fix documentation package versions (:pr:`151`)
* Deprecate experimental w-stacking gridder in favour of nifty gridder (:pr:`148`)
* Expand travis build matrix (:pr:`147`)
* Drop Python 2 support (:pr:`146`, :pr:`149`, :pr:`150`)
* Support the beam in the predict example (:pr:`145`)
* Fix weight indexing in averaging (:pr:`144`)
* Support EFFECTIVE_BW and RESOLUTION in averaging (:pr:`144`)
* Optimise predict_vis jones coherency summation (:pr:`143`)
* Remove use of @wraps (:pr:`141`, :pr:`142`)
* Set row chunks to nan in dask averaging code. (:pr:`139`)
* predict_vis documentation improvements (:pr:`135`, :pr:`140`)
* Upgrade to dask-ms in the examples (:pr:`134`, :pr:`138`)
* Explain how to obtain predict_vis time_index argument (:pr:`130`)
* Update RIME predict example to support Tigger LSM's and Gaussians (:pr:`129`)
* Add dask wrappers for the nifty gridder (:pr:`116`, :pr:`136`, :pr:`146`)
* Testing and requirement updates. (:pr:`124`)
* Upgraded DFT kernels to have a correlation axis and added flags
  for vis_to_im. Added predict_from_fits example. (:pr:`122`)
* Fixed segfault when using `_unique_internal` on empty ndarrays (:pr:`123`)
* Removed `apply_gains`. Use `africanus.calibration.utils.correct_vis`
  instead (:pr:`118`)
* Add streams parameter to dask `predict_vis` (:pr:`118`)
* Implement the beam in numba (:pr:`112`)
* Add residual_vis, correct_vis, phase_only_GN (:pr:`113`)

0.1.8 (2019-05-28)
------------------

* Use environment markers in setup.py (:pr:`110`)
* Add `apply_gains`, a wrapper around `predict_vis` (:pr:`108`)
* Fix testing extras_require (:pr:`107`)
* Fix WEIGHT_SPECTRUM averaging and add more averaging tests (:pr:`106`)

0.1.7 (2019-05-09)
------------------

* Even more support for automated travis deploys.

0.1.6 (2019-05-09)
------------------

* Support automated travis deploys.

0.1.5 (2019-05-09)
------------------
* Predict script enhancements (:pr:`103`) and
  dask channel chunking fix (:issue:`104`).
* Directly jit DFT functions (:pr:`100`, :pr:`101`)
* Spectral Models (:pr:`86`)
* Fix radec sign conversion in wsclean sky model (:pr:`96`)
* Full Time and Channel Averaging Implementation (:pr:`80`, :pr:`97`, :pr:`98`)
* Support integer seconds in wsclean ra and dec columns (:pr:`91`, :pr:`93`)
* Fix ratio computation in Gaussian Shape (:pr:`89`, :pr:`90`)

0.1.4 (2019-03-11)
------------------
* Support `complete` and `complete-cuda` to support non-GPU installs (:pr:`87`)
* Gaussian Shape Parameter Implementation (:pr:`82`, :pr:`83`)
* WSClean Spectral Model (:pr:`81`)
* Compare predict versus MeqTrees (:pr:`79`)
* Time and channel averaging (:pr:`75`)
* cupy implementation of `predict_vis` (:pr:`73`)
* Introduce transpose in second antenna term of predict (:pr:`72`)
* cupy implementation of `feed_rotation` (:pr:`67`)
* cupy implementation of `stokes_convert` kernel (:pr:`65`)
* Add a basic RIME example (:pr:`64`)
* requires_optional accepts ImportError's for a
  better debugging experience (:pr:`62`, :pr:`63`)
* Added `fit_component_spi` function (:pr:`61`)
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
