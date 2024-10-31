import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from africanus.model.shape import shapelet, shapelet_1d
from africanus.constants import c as lightspeed

Fs = np.fft.fftshift
iFs = np.fft.ifftshift

fft = np.fft.fft
ifft = np.fft.ifft
fft2 = np.fft.fft2
ifft2 = np.fft.ifft2


def test_1d_shapelet():
  # set signal space coords
  beta = 1.0
  npix = 513
  coeffs = np.ones(1, dtype=np.float64)
  l_min = -15.0 * beta
  l_max = 15.0 * beta
  delta_l = (l_max - l_min) / (npix - 1)
  if npix % 2:
    l_coords = l_min + np.arange(npix) * delta_l
  else:
    l_coords = l_min + np.arange(-0.5, npix - 0.5) * delta_l
  img_shape = shapelet_1d(l_coords, coeffs, False, beta=beta)

  # get Fourier space coords and take fft
  u = Fs(np.fft.fftfreq(npix, d=delta_l))
  fft_shape = Fs(fft(iFs(img_shape)))

  # get uv space
  uv_shape = shapelet_1d(u, coeffs, True, delta_x=delta_l, beta=beta)

  assert np.allclose(uv_shape, fft_shape)


def test_2d_shapelet(gf_shapelets):
  # Define all respective values for nrow, ncoeff, etc
  beta = [0.01, 0.01]
  nchan = 1
  ncoeffs = [1, 1]
  nsrc = 1

  # Define the range of uv values
  u_range = [-2 * np.sqrt(2) * (beta[0] ** (-1)), 2 * np.sqrt(2) * (beta[0] ** (-1))]
  v_range = [-2 * np.sqrt(2) * (beta[1] ** (-1)), 2 * np.sqrt(2) * (beta[1] ** (-1))]

  # Create an lm grid from the regular uv grid
  max_u = u_range[1]
  max_v = v_range[1]
  delta_x = 1 / (2 * max_u) if max_u > max_v else 1 / (2 * max_v)
  x_range = [-2 * np.sqrt(2) * beta[0], 2 * np.sqrt(2) * beta[0]]
  y_range = [-2 * np.sqrt(2) * beta[1], 2 * np.sqrt(2) * beta[1]]
  npix_x = int((x_range[1] - x_range[0]) / delta_x)
  npix_y = int((y_range[1] - y_range[0]) / delta_x)
  l_vals = np.linspace(x_range[0], x_range[1], npix_x)
  m_vals = np.linspace(y_range[0], y_range[1], npix_y)
  ll, mm = np.meshgrid(l_vals, m_vals)
  lm = np.vstack((ll.flatten(), mm.flatten())).T
  nrow = lm.shape[0]

  # Create input arrays
  img_coords = np.zeros((nrow, 3))
  img_coeffs = np.random.randn(nsrc, ncoeffs[0], ncoeffs[1])
  img_beta = np.zeros((nsrc, 2))
  frequency = np.empty((nchan), dtype=float)

  # Assign values to input arrays
  img_coords[:, :2], img_coords[:, 2] = lm[:, :], 0
  img_beta[0, :] = beta[:]
  frequency[:] = 1
  img_coeffs[:, :, :] = 1

  l_shapelets = shapelet_1d(
    img_coords[:, 0], img_coeffs[0, 0, :], False, beta=img_beta[0, 0]
  )
  m_shapelets = shapelet_1d(
    img_coords[:, 1], img_coeffs[0, 0, :], False, beta=img_beta[0, 1]
  )
  ca_shapelets = l_shapelets * m_shapelets

  # Compare griffinfoster (gf) shapelets to codex-africanus (ca) shapelets
  assert np.allclose(gf_shapelets, ca_shapelets)


def test_fourier_space_shapelets():
  # set overall scale
  beta_l = 1.0
  beta_m = 1.0

  # only taking the zeroth order with
  ncoeffs_l = 1
  ncoeffs_m = 1
  nsrc = 1
  coeffs_l = np.ones((nsrc, ncoeffs_l), dtype=np.float64)
  coeffs_m = np.ones((nsrc, ncoeffs_m), dtype=np.float64)

  # Define the range of lm values (these give 3 standard deviations for the
  # 0th order shapelet in image space)
  scale_fact = 10.0
  l_min = -3 * np.sqrt(2) * beta_l * scale_fact
  l_max = 3 * np.sqrt(2) * beta_l * scale_fact
  m_min = -3 * np.sqrt(2) * beta_m * scale_fact
  m_max = 3 * np.sqrt(2) * beta_m * scale_fact

  # set number of pixels
  npix = 257

  # create image space coordinate grid
  delta_l = (l_max - l_min) / (npix - 1)
  delta_m = (m_max - m_min) / (npix - 1)
  lvals = l_min + np.arange(npix) * delta_l
  mvals = m_min + np.arange(npix) * delta_m
  assert lvals[-1] == l_max
  assert mvals[-1] == m_max
  ll, mm = np.meshgrid(lvals, mvals)
  lm = np.vstack((ll.flatten(), mm.flatten())).T

  l_shapelets = shapelet_1d(lm[:, 0], coeffs_l[0, :], False, beta=beta_l)
  m_shapelets = shapelet_1d(lm[:, 1], coeffs_m[0, :], False, beta=beta_m)
  img_space_shape = l_shapelets * m_shapelets

  # next take FFT
  fft_shapelet = Fs(fft2(iFs(img_space_shape.reshape(npix, npix))))
  fft_shapelet_max = fft_shapelet.real.max()
  fft_shapelet /= fft_shapelet_max

  # get freq space coords
  freq = Fs(np.fft.fftfreq(npix, d=(delta_l)))

  # Create uv grid
  uu, vv = np.meshgrid(freq, freq)
  nrows = uu.size
  assert nrows == npix**2
  uv = np.hstack((uu.reshape(nrows, 1), vv.reshape(nrows, 1)))
  uvw = np.zeros((nrows, 3), dtype=np.float64)
  uvw[:, 0:2] = uv

  # Define other parameters for shapelet call
  nchan = 1
  frequency = np.ones(nchan, dtype=np.float64) * lightspeed / (2 * np.pi)
  beta = np.zeros((nsrc, 2), dtype=np.float64)
  beta[0, 0] = beta_l
  beta[0, 1] = beta_m

  # Call the shapelet implementation
  coeffs_l = coeffs_l.reshape(coeffs_l.shape + (1,))
  uv_space_shapelet = shapelet(
    uvw, frequency, coeffs_l, beta, (delta_l, delta_l)
  ).reshape(npix, npix)
  uv_space_shapelet_max = uv_space_shapelet.real.max()
  uv_space_shapelet /= uv_space_shapelet_max

  assert np.allclose(fft_shapelet, uv_space_shapelet)


def test_dask_shapelets():
  da = pytest.importorskip("dask.array")
  from africanus.model.shape.dask import shapelet as da_shapelet
  from africanus.model.shape import shapelet as nb_shapelet

  row_chunks = (2, 2)
  source_chunks = (5, 10, 5, 5)

  row = sum(row_chunks)
  source = sum(source_chunks)
  nmax = [5, 5]
  beta_vals = [1.0, 1.0]
  nchan = 1

  np_coords = np.random.randn(row, 3)
  np_coeffs = np.random.randn(source, nmax[0], nmax[1])
  np_frequency = np.random.randn(nchan)
  np_beta = np.empty((source, 2))
  np_beta[:, 0], np_beta[:, 1] = beta_vals[0], beta_vals[1]
  np_delta_lm = np.array(
    [1 / (10 * np.max(np_coords[:, 0])), 1 / (10 * np.max(np_coords[:, 1]))]
  )

  da_coords = da.from_array(np_coords, chunks=(row_chunks, 3))
  da_coeffs = da.from_array(np_coeffs, chunks=(source_chunks, nmax[0], nmax[1]))
  da_frequency = da.from_array(np_frequency, chunks=(nchan,))
  da_beta = da.from_array(np_beta, chunks=(source_chunks, 2))
  delta_lm = da.from_array(np_delta_lm, chunks=(2))

  np_shapelets = nb_shapelet(np_coords, np_frequency, np_coeffs, np_beta, np_delta_lm)
  da_shapelets = da_shapelet(
    da_coords, da_frequency, da_coeffs, da_beta, delta_lm
  ).compute()
  assert_array_almost_equal(da_shapelets, np_shapelets)


@pytest.fixture
def gf_shapelets():
  return np.array(
    [
      0.018926452033215378,
      0.03118821286615593,
      0.04971075420435816,
      0.07663881765058349,
      0.1142841022428579,
      0.16483955267999187,
      0.22997234561316462,
      0.3103333028594875,
      0.40506035000019336,
      0.5113869779741611,
      0.624479487346518,
      0.7376074103159139,
      0.8426960266006467,
      0.9312262345416489,
      0.9953550860502569,
      1.0290570650923165,
      1.0290570650923165,
      0.9953550860502569,
      0.9312262345416489,
      0.8426960266006465,
      0.7376074103159138,
      0.624479487346518,
      0.5113869779741611,
      0.40506035000019325,
      0.31033330285948746,
      0.22997234561316438,
      0.16483955267999187,
      0.1142841022428579,
      0.07663881765058349,
      0.04971075420435811,
      0.03118821286615593,
      0.018926452033215378,
      0.03118821286615593,
      0.05139392317575348,
      0.08191654627828761,
      0.12629032396046155,
      0.1883246211023866,
      0.2716331116219307,
      0.3789630753680171,
      0.5113869779741613,
      0.6674842383176157,
      0.8426960266006469,
      1.029057065092317,
      1.2154764603642687,
      1.3886481741511985,
      1.5345338882566935,
      1.6402094933940174,
      1.6957457605469852,
      1.6957457605469852,
      1.6402094933940174,
      1.5345338882566935,
      1.388648174151198,
      1.2154764603642685,
      1.029057065092317,
      0.8426960266006469,
      0.6674842383176155,
      0.5113869779741612,
      0.37896307536801666,
      0.2716331116219307,
      0.1883246211023866,
      0.12629032396046155,
      0.08191654627828754,
      0.05139392317575348,
      0.03118821286615593,
      0.04971075420435816,
      0.08191654627828761,
      0.130566419909516,
      0.201293587411668,
      0.3001697785771043,
      0.43295481224112503,
      0.6040275655739769,
      0.815097436793798,
      1.0639001679476456,
      1.3431694604339557,
      1.6402094933940163,
      1.937342540967424,
      2.2133601677597268,
      2.4458867606410766,
      2.6143226391225554,
      2.702841649099706,
      2.702841649099706,
      2.6143226391225554,
      2.4458867606410766,
      2.213360167759726,
      1.9373425409674239,
      1.6402094933940163,
      1.3431694604339557,
      1.0639001679476452,
      0.8150974367937979,
      0.6040275655739762,
      0.43295481224112503,
      0.3001697785771043,
      0.201293587411668,
      0.13056641990951587,
      0.08191654627828761,
      0.04971075420435816,
      0.07663881765058349,
      0.12629032396046155,
      0.201293587411668,
      0.31033330285948735,
      0.462770225332246,
      0.6674842383176153,
      0.9312262345416487,
      1.2566315845680993,
      1.6402094933940163,
      2.0707575453161375,
      2.528702657702978,
      2.9867911702474954,
      3.412326145659869,
      3.770811214654918,
      4.030487954293403,
      4.166957263054249,
      4.166957263054249,
      4.030487954293403,
      3.770811214654918,
      3.412326145659868,
      2.986791170247495,
      2.528702657702978,
      2.0707575453161375,
      1.640209493394016,
      1.256631584568099,
      0.9312262345416477,
      0.6674842383176153,
      0.462770225332246,
      0.31033330285948735,
      0.2012935874116678,
      0.12629032396046155,
      0.07663881765058349,
      0.1142841022428579,
      0.1883246211023866,
      0.3001697785771043,
      0.462770225332246,
      0.6900847555862331,
      0.9953550860502568,
      1.3886481741511976,
      1.8738938946990888,
      2.445886760641078,
      3.0879216862145458,
      3.7708112146549193,
      4.453914581966821,
      5.088473988397974,
      5.6230483142227445,
      6.010279275929925,
      6.2137828386615945,
      6.2137828386615945,
      6.010279275929925,
      5.6230483142227445,
      5.088473988397973,
      4.45391458196682,
      3.7708112146549193,
      3.0879216862145458,
      2.445886760641077,
      1.8738938946990886,
      1.3886481741511962,
      0.9953550860502568,
      0.6900847555862331,
      0.462770225332246,
      0.30016977857710403,
      0.1883246211023866,
      0.1142841022428579,
      0.16483955267999187,
      0.2716331116219307,
      0.43295481224112503,
      0.6674842383176153,
      0.9953550860502568,
      1.4356667631130016,
      2.002939510961386,
      2.702841649099707,
      3.527864957745564,
      4.4539145819668216,
      5.438891513917972,
      6.424176879879071,
      7.3394440662346385,
      8.110496128715798,
      8.669025068952923,
      8.962551776440094,
      8.962551776440094,
      8.669025068952923,
      8.110496128715798,
      7.339444066234637,
      6.42417687987907,
      5.438891513917972,
      4.4539145819668216,
      3.527864957745563,
      2.7028416490997067,
      2.002939510961384,
      1.4356667631130016,
      0.9953550860502568,
      0.6674842383176153,
      0.4329548122411246,
      0.2716331116219307,
      0.16483955267999187,
      0.22997234561316462,
      0.3789630753680171,
      0.6040275655739769,
      0.9312262345416487,
      1.3886481741511976,
      2.002939510961386,
      2.794357846573947,
      3.7708112146549193,
      4.921824684359944,
      6.213782838661596,
      7.587952155023488,
      8.962551776440092,
      10.239467045177655,
      11.315183695191251,
      12.094403296257042,
      12.503910749556066,
      12.503910749556066,
      12.094403296257042,
      11.315183695191251,
      10.239467045177651,
      8.96255177644009,
      7.587952155023488,
      6.213782838661596,
      4.921824684359942,
      3.7708112146549184,
      2.7943578465739445,
      2.002939510961386,
      1.3886481741511976,
      0.9312262345416487,
      0.6040275655739763,
      0.3789630753680171,
      0.22997234561316462,
      0.3103333028594875,
      0.5113869779741613,
      0.815097436793798,
      1.2566315845680993,
      1.8738938946990888,
      2.702841649099707,
      3.7708112146549193,
      5.088473988397975,
      6.641694706032257,
      8.385111463867542,
      10.239467045177655,
      12.094403296257038,
      13.817520620483494,
      15.269132987394766,
      16.320641123326727,
      16.873245829727512,
      16.873245829727512,
      16.320641123326727,
      15.269132987394766,
      13.81752062048349,
      12.094403296257036,
      10.239467045177655,
      8.385111463867542,
      6.641694706032255,
      5.088473988397973,
      3.7708112146549153,
      2.702841649099707,
      1.8738938946990888,
      1.2566315845680993,
      0.8150974367937973,
      0.5113869779741613,
      0.3103333028594875,
      0.40506035000019336,
      0.6674842383176157,
      1.0639001679476456,
      1.6402094933940163,
      2.445886760641078,
      3.527864957745564,
      4.921824684359944,
      6.641694706032257,
      8.669025068952925,
      10.944607468965954,
      13.36499198416051,
      15.786134414467007,
      18.035221122246448,
      19.929927903593466,
      21.302401465547916,
      22.023684852551362,
      22.023684852551362,
      21.302401465547916,
      19.929927903593466,
      18.03522112224644,
      15.786134414467005,
      13.36499198416051,
      10.944607468965954,
      8.669025068952921,
      6.641694706032256,
      4.9218246843599385,
      3.527864957745564,
      2.445886760641078,
      1.6402094933940163,
      1.0639001679476445,
      0.6674842383176157,
      0.40506035000019336,
      0.5113869779741611,
      0.8426960266006469,
      1.3431694604339557,
      2.0707575453161375,
      3.0879216862145458,
      4.4539145819668216,
      6.213782838661596,
      8.385111463867542,
      10.944607468965954,
      13.81752062048349,
      16.873245829727516,
      19.92992790359346,
      22.7693903557753,
      25.16144965029706,
      26.894191715021453,
      27.804808939201713,
      27.804808939201713,
      26.894191715021453,
      25.16144965029706,
      22.769390355775293,
      19.929927903593455,
      16.873245829727516,
      13.81752062048349,
      10.94460746896595,
      8.38511146386754,
      6.213782838661589,
      4.4539145819668216,
      3.0879216862145458,
      2.0707575453161375,
      1.3431694604339544,
      0.8426960266006469,
      0.5113869779741611,
      0.624479487346518,
      1.029057065092317,
      1.6402094933940163,
      2.528702657702978,
      3.7708112146549193,
      5.438891513917972,
      7.587952155023488,
      10.239467045177655,
      13.36499198416051,
      16.873245829727516,
      20.60474036191124,
      24.337403367979306,
      27.80480893920172,
      30.725868775068133,
      32.84180430508374,
      33.95380324486458,
      33.95380324486458,
      32.84180430508374,
      30.725868775068133,
      27.804808939201713,
      24.337403367979302,
      20.60474036191124,
      16.873245829727516,
      13.364991984160506,
      10.239467045177653,
      7.587952155023479,
      5.438891513917972,
      3.7708112146549193,
      2.528702657702978,
      1.6402094933940148,
      1.029057065092317,
      0.624479487346518,
      0.7376074103159139,
      1.2154764603642687,
      1.937342540967424,
      2.9867911702474954,
      4.453914581966821,
      6.424176879879071,
      8.962551776440092,
      12.094403296257038,
      15.786134414467007,
      19.92992790359346,
      24.337403367979306,
      28.746258981774883,
      32.841804305083734,
      36.292030332629274,
      38.79127932048947,
      40.1047230362006,
      40.1047230362006,
      38.79127932048947,
      36.292030332629274,
      32.84180430508372,
      28.74625898177488,
      24.337403367979306,
      19.92992790359346,
      15.786134414467003,
      12.094403296257036,
      8.962551776440083,
      6.424176879879071,
      4.453914581966821,
      2.9867911702474954,
      1.9373425409674223,
      1.2154764603642687,
      0.7376074103159139,
      0.8426960266006467,
      1.3886481741511985,
      2.2133601677597268,
      3.412326145659869,
      5.088473988397974,
      7.3394440662346385,
      10.239467045177655,
      13.817520620483494,
      18.035221122246448,
      22.7693903557753,
      27.80480893920172,
      32.841804305083734,
      37.52085134616084,
      41.462638974136944,
      44.317961686599176,
      45.81853473523395,
      45.81853473523395,
      44.317961686599176,
      41.462638974136944,
      37.52085134616083,
      32.84180430508373,
      27.80480893920172,
      22.7693903557753,
      18.03522112224644,
      13.81752062048349,
      10.239467045177644,
      7.3394440662346385,
      5.088473988397974,
      3.412326145659869,
      2.2133601677597246,
      1.3886481741511985,
      0.8426960266006467,
      0.9312262345416489,
      1.5345338882566935,
      2.4458867606410766,
      3.770811214654918,
      5.6230483142227445,
      8.110496128715798,
      11.315183695191251,
      15.269132987394766,
      19.929927903593466,
      25.16144965029706,
      30.725868775068133,
      36.292030332629274,
      41.462638974136944,
      45.818534735233946,
      48.9738260075251,
      50.632043141135796,
      50.632043141135796,
      48.9738260075251,
      45.818534735233946,
      41.46263897413693,
      36.29203033262927,
      30.725868775068133,
      25.16144965029706,
      19.92992790359346,
      15.269132987394762,
      11.315183695191239,
      8.110496128715798,
      5.6230483142227445,
      3.770811214654918,
      2.4458867606410744,
      1.5345338882566935,
      0.9312262345416489,
      0.9953550860502569,
      1.6402094933940174,
      2.6143226391225554,
      4.030487954293403,
      6.010279275929925,
      8.669025068952923,
      12.094403296257042,
      16.320641123326727,
      21.302401465547916,
      26.894191715021453,
      32.84180430508374,
      38.79127932048947,
      44.317961686599176,
      48.9738260075251,
      52.34640626713388,
      54.11881644684437,
      54.11881644684437,
      52.34640626713388,
      48.9738260075251,
      44.31796168659916,
      38.791279320489465,
      32.84180430508374,
      26.894191715021453,
      21.30240146554791,
      16.320641123326723,
      12.09440329625703,
      8.669025068952923,
      6.010279275929925,
      4.030487954293403,
      2.6143226391225527,
      1.6402094933940174,
      0.9953550860502569,
      1.0290570650923165,
      1.6957457605469852,
      2.702841649099706,
      4.166957263054249,
      6.2137828386615945,
      8.962551776440094,
      12.503910749556066,
      16.873245829727512,
      22.023684852551362,
      27.804808939201713,
      33.95380324486458,
      40.1047230362006,
      45.81853473523395,
      50.632043141135796,
      54.11881644684437,
      55.9512391101074,
      55.9512391101074,
      54.11881644684437,
      50.632043141135796,
      45.81853473523394,
      40.104723036200596,
      33.95380324486458,
      27.804808939201713,
      22.023684852551355,
      16.873245829727512,
      12.503910749556052,
      8.962551776440094,
      6.2137828386615945,
      4.166957263054249,
      2.7028416490997036,
      1.6957457605469852,
      1.0290570650923165,
      1.0290570650923165,
      1.6957457605469852,
      2.702841649099706,
      4.166957263054249,
      6.2137828386615945,
      8.962551776440094,
      12.503910749556066,
      16.873245829727512,
      22.023684852551362,
      27.804808939201713,
      33.95380324486458,
      40.1047230362006,
      45.81853473523395,
      50.632043141135796,
      54.11881644684437,
      55.9512391101074,
      55.9512391101074,
      54.11881644684437,
      50.632043141135796,
      45.81853473523394,
      40.104723036200596,
      33.95380324486458,
      27.804808939201713,
      22.023684852551355,
      16.873245829727512,
      12.503910749556052,
      8.962551776440094,
      6.2137828386615945,
      4.166957263054249,
      2.7028416490997036,
      1.6957457605469852,
      1.0290570650923165,
      0.9953550860502569,
      1.6402094933940174,
      2.6143226391225554,
      4.030487954293403,
      6.010279275929925,
      8.669025068952923,
      12.094403296257042,
      16.320641123326727,
      21.302401465547916,
      26.894191715021453,
      32.84180430508374,
      38.79127932048947,
      44.317961686599176,
      48.9738260075251,
      52.34640626713388,
      54.11881644684437,
      54.11881644684437,
      52.34640626713388,
      48.9738260075251,
      44.31796168659916,
      38.791279320489465,
      32.84180430508374,
      26.894191715021453,
      21.30240146554791,
      16.320641123326723,
      12.09440329625703,
      8.669025068952923,
      6.010279275929925,
      4.030487954293403,
      2.6143226391225527,
      1.6402094933940174,
      0.9953550860502569,
      0.9312262345416489,
      1.5345338882566935,
      2.4458867606410766,
      3.770811214654918,
      5.6230483142227445,
      8.110496128715798,
      11.315183695191251,
      15.269132987394766,
      19.929927903593466,
      25.16144965029706,
      30.725868775068133,
      36.292030332629274,
      41.462638974136944,
      45.818534735233946,
      48.9738260075251,
      50.632043141135796,
      50.632043141135796,
      48.9738260075251,
      45.818534735233946,
      41.46263897413693,
      36.29203033262927,
      30.725868775068133,
      25.16144965029706,
      19.92992790359346,
      15.269132987394762,
      11.315183695191239,
      8.110496128715798,
      5.6230483142227445,
      3.770811214654918,
      2.4458867606410744,
      1.5345338882566935,
      0.9312262345416489,
      0.8426960266006465,
      1.388648174151198,
      2.213360167759726,
      3.412326145659868,
      5.088473988397973,
      7.339444066234637,
      10.239467045177651,
      13.81752062048349,
      18.03522112224644,
      22.769390355775293,
      27.804808939201713,
      32.84180430508372,
      37.52085134616083,
      41.46263897413693,
      44.31796168659916,
      45.81853473523394,
      45.81853473523394,
      44.31796168659916,
      41.46263897413693,
      37.52085134616082,
      32.84180430508372,
      27.804808939201713,
      22.769390355775293,
      18.035221122246437,
      13.817520620483487,
      10.23946704517764,
      7.339444066234637,
      5.088473988397973,
      3.412326145659868,
      2.213360167759724,
      1.388648174151198,
      0.8426960266006465,
      0.7376074103159138,
      1.2154764603642685,
      1.9373425409674239,
      2.986791170247495,
      4.45391458196682,
      6.42417687987907,
      8.96255177644009,
      12.094403296257036,
      15.786134414467005,
      19.929927903593455,
      24.337403367979302,
      28.74625898177488,
      32.84180430508373,
      36.29203033262927,
      38.791279320489465,
      40.104723036200596,
      40.104723036200596,
      38.791279320489465,
      36.29203033262927,
      32.84180430508372,
      28.746258981774876,
      24.337403367979302,
      19.929927903593455,
      15.786134414467,
      12.094403296257035,
      8.962551776440081,
      6.42417687987907,
      4.45391458196682,
      2.986791170247495,
      1.937342540967422,
      1.2154764603642685,
      0.7376074103159138,
      0.624479487346518,
      1.029057065092317,
      1.6402094933940163,
      2.528702657702978,
      3.7708112146549193,
      5.438891513917972,
      7.587952155023488,
      10.239467045177655,
      13.36499198416051,
      16.873245829727516,
      20.60474036191124,
      24.337403367979306,
      27.80480893920172,
      30.725868775068133,
      32.84180430508374,
      33.95380324486458,
      33.95380324486458,
      32.84180430508374,
      30.725868775068133,
      27.804808939201713,
      24.337403367979302,
      20.60474036191124,
      16.873245829727516,
      13.364991984160506,
      10.239467045177653,
      7.587952155023479,
      5.438891513917972,
      3.7708112146549193,
      2.528702657702978,
      1.6402094933940148,
      1.029057065092317,
      0.624479487346518,
      0.5113869779741611,
      0.8426960266006469,
      1.3431694604339557,
      2.0707575453161375,
      3.0879216862145458,
      4.4539145819668216,
      6.213782838661596,
      8.385111463867542,
      10.944607468965954,
      13.81752062048349,
      16.873245829727516,
      19.92992790359346,
      22.7693903557753,
      25.16144965029706,
      26.894191715021453,
      27.804808939201713,
      27.804808939201713,
      26.894191715021453,
      25.16144965029706,
      22.769390355775293,
      19.929927903593455,
      16.873245829727516,
      13.81752062048349,
      10.94460746896595,
      8.38511146386754,
      6.213782838661589,
      4.4539145819668216,
      3.0879216862145458,
      2.0707575453161375,
      1.3431694604339544,
      0.8426960266006469,
      0.5113869779741611,
      0.40506035000019325,
      0.6674842383176155,
      1.0639001679476452,
      1.640209493394016,
      2.445886760641077,
      3.527864957745563,
      4.921824684359942,
      6.641694706032255,
      8.669025068952921,
      10.94460746896595,
      13.364991984160506,
      15.786134414467003,
      18.03522112224644,
      19.92992790359346,
      21.30240146554791,
      22.023684852551355,
      22.023684852551355,
      21.30240146554791,
      19.92992790359346,
      18.035221122246437,
      15.786134414467,
      13.364991984160506,
      10.94460746896595,
      8.66902506895292,
      6.641694706032253,
      4.921824684359937,
      3.527864957745563,
      2.445886760641077,
      1.640209493394016,
      1.0639001679476443,
      0.6674842383176155,
      0.40506035000019325,
      0.31033330285948746,
      0.5113869779741612,
      0.8150974367937979,
      1.256631584568099,
      1.8738938946990886,
      2.7028416490997067,
      3.7708112146549184,
      5.088473988397973,
      6.641694706032256,
      8.38511146386754,
      10.239467045177653,
      12.094403296257036,
      13.81752062048349,
      15.269132987394762,
      16.320641123326723,
      16.873245829727512,
      16.873245829727512,
      16.320641123326723,
      15.269132987394762,
      13.817520620483487,
      12.094403296257035,
      10.239467045177653,
      8.38511146386754,
      6.641694706032253,
      5.088473988397972,
      3.7708112146549144,
      2.7028416490997067,
      1.8738938946990886,
      1.256631584568099,
      0.8150974367937971,
      0.5113869779741612,
      0.31033330285948746,
      0.22997234561316438,
      0.37896307536801666,
      0.6040275655739762,
      0.9312262345416477,
      1.3886481741511962,
      2.002939510961384,
      2.7943578465739445,
      3.7708112146549153,
      4.9218246843599385,
      6.213782838661589,
      7.587952155023479,
      8.962551776440083,
      10.239467045177644,
      11.315183695191239,
      12.09440329625703,
      12.503910749556052,
      12.503910749556052,
      12.09440329625703,
      11.315183695191239,
      10.23946704517764,
      8.962551776440081,
      7.587952155023479,
      6.213782838661589,
      4.921824684359937,
      3.7708112146549144,
      2.7943578465739414,
      2.002939510961384,
      1.3886481741511962,
      0.9312262345416477,
      0.6040275655739756,
      0.37896307536801666,
      0.22997234561316438,
      0.16483955267999187,
      0.2716331116219307,
      0.43295481224112503,
      0.6674842383176153,
      0.9953550860502568,
      1.4356667631130016,
      2.002939510961386,
      2.702841649099707,
      3.527864957745564,
      4.4539145819668216,
      5.438891513917972,
      6.424176879879071,
      7.3394440662346385,
      8.110496128715798,
      8.669025068952923,
      8.962551776440094,
      8.962551776440094,
      8.669025068952923,
      8.110496128715798,
      7.339444066234637,
      6.42417687987907,
      5.438891513917972,
      4.4539145819668216,
      3.527864957745563,
      2.7028416490997067,
      2.002939510961384,
      1.4356667631130016,
      0.9953550860502568,
      0.6674842383176153,
      0.4329548122411246,
      0.2716331116219307,
      0.16483955267999187,
      0.1142841022428579,
      0.1883246211023866,
      0.3001697785771043,
      0.462770225332246,
      0.6900847555862331,
      0.9953550860502568,
      1.3886481741511976,
      1.8738938946990888,
      2.445886760641078,
      3.0879216862145458,
      3.7708112146549193,
      4.453914581966821,
      5.088473988397974,
      5.6230483142227445,
      6.010279275929925,
      6.2137828386615945,
      6.2137828386615945,
      6.010279275929925,
      5.6230483142227445,
      5.088473988397973,
      4.45391458196682,
      3.7708112146549193,
      3.0879216862145458,
      2.445886760641077,
      1.8738938946990886,
      1.3886481741511962,
      0.9953550860502568,
      0.6900847555862331,
      0.462770225332246,
      0.30016977857710403,
      0.1883246211023866,
      0.1142841022428579,
      0.07663881765058349,
      0.12629032396046155,
      0.201293587411668,
      0.31033330285948735,
      0.462770225332246,
      0.6674842383176153,
      0.9312262345416487,
      1.2566315845680993,
      1.6402094933940163,
      2.0707575453161375,
      2.528702657702978,
      2.9867911702474954,
      3.412326145659869,
      3.770811214654918,
      4.030487954293403,
      4.166957263054249,
      4.166957263054249,
      4.030487954293403,
      3.770811214654918,
      3.412326145659868,
      2.986791170247495,
      2.528702657702978,
      2.0707575453161375,
      1.640209493394016,
      1.256631584568099,
      0.9312262345416477,
      0.6674842383176153,
      0.462770225332246,
      0.31033330285948735,
      0.2012935874116678,
      0.12629032396046155,
      0.07663881765058349,
      0.04971075420435811,
      0.08191654627828754,
      0.13056641990951587,
      0.2012935874116678,
      0.30016977857710403,
      0.4329548122411246,
      0.6040275655739763,
      0.8150974367937973,
      1.0639001679476445,
      1.3431694604339544,
      1.6402094933940148,
      1.9373425409674223,
      2.2133601677597246,
      2.4458867606410744,
      2.6143226391225527,
      2.7028416490997036,
      2.7028416490997036,
      2.6143226391225527,
      2.4458867606410744,
      2.213360167759724,
      1.937342540967422,
      1.6402094933940148,
      1.3431694604339544,
      1.0639001679476443,
      0.8150974367937971,
      0.6040275655739756,
      0.4329548122411246,
      0.30016977857710403,
      0.2012935874116678,
      0.13056641990951576,
      0.08191654627828754,
      0.04971075420435811,
      0.03118821286615593,
      0.05139392317575348,
      0.08191654627828761,
      0.12629032396046155,
      0.1883246211023866,
      0.2716331116219307,
      0.3789630753680171,
      0.5113869779741613,
      0.6674842383176157,
      0.8426960266006469,
      1.029057065092317,
      1.2154764603642687,
      1.3886481741511985,
      1.5345338882566935,
      1.6402094933940174,
      1.6957457605469852,
      1.6957457605469852,
      1.6402094933940174,
      1.5345338882566935,
      1.388648174151198,
      1.2154764603642685,
      1.029057065092317,
      0.8426960266006469,
      0.6674842383176155,
      0.5113869779741612,
      0.37896307536801666,
      0.2716331116219307,
      0.1883246211023866,
      0.12629032396046155,
      0.08191654627828754,
      0.05139392317575348,
      0.03118821286615593,
      0.018926452033215378,
      0.03118821286615593,
      0.04971075420435816,
      0.07663881765058349,
      0.1142841022428579,
      0.16483955267999187,
      0.22997234561316462,
      0.3103333028594875,
      0.40506035000019336,
      0.5113869779741611,
      0.624479487346518,
      0.7376074103159139,
      0.8426960266006467,
      0.9312262345416489,
      0.9953550860502569,
      1.0290570650923165,
      1.0290570650923165,
      0.9953550860502569,
      0.9312262345416489,
      0.8426960266006465,
      0.7376074103159138,
      0.624479487346518,
      0.5113869779741611,
      0.40506035000019325,
      0.31033330285948746,
      0.22997234561316438,
      0.16483955267999187,
      0.1142841022428579,
      0.07663881765058349,
      0.04971075420435811,
      0.03118821286615593,
      0.018926452033215378,
    ]
  )
