import numpy as np
import pytest


def test_zernike_func_xx_corr(coeff_xx, noll_index_xx, eidos_data_xx):
    """Tests reconstruction of xx correlation against eidos"""
    from africanus.rime import zernike_dde

    npix = 17
    nsrc = npix**2
    ntime = 1
    na = 1
    nchan = 1
    ncorr = 1
    thresh = 15
    npoly = thresh

    # Linear (l,m) grid
    nx, ny = npix, npix
    grid = (np.indices((nx, ny), dtype=float) - nx // 2) * 2 / nx
    ll, mm = grid[0], grid[1]

    lm = np.vstack((ll.flatten(), mm.flatten())).T

    # Initializing coords, coeffs, and noll_indices
    coords = np.empty((3, nsrc, ntime, na, nchan), dtype=float)
    coeffs = np.empty((na, nchan, ncorr, npoly), dtype=np.complex128)
    noll_indices = np.empty((na, nchan, ncorr, npoly))
    parallactic_angles = np.zeros((ntime, na), dtype=np.float64)
    frequency_scaling = np.ones((nchan,), dtype=np.float64)
    antenna_scaling = np.ones((na, nchan, 2), dtype=np.float64)
    pointing_errors = np.zeros((ntime, na, nchan, 2), dtype=np.float64)

    # Assign Values to coeffs and noll_indices
    coeffs[0, 0, 0, :] = coeff_xx[:thresh]
    noll_indices[0, 0, 0, :] = noll_index_xx[:thresh]

    # I left 0 as all the freq values
    coords[0, 0:nsrc, 0, 0, 0] = lm[0:nsrc, 0]
    coords[1, 0:nsrc, 0, 0, 0] = lm[0:nsrc, 1]
    coords[2, 0:nsrc, 0, 0, 0] = 0

    # Call the function, reshape accordingly, and normalise
    zernike_vals = zernike_dde(
        coords,
        coeffs,
        noll_indices,
        parallactic_angles,
        frequency_scaling,
        antenna_scaling,
        pointing_errors,
    )[:, 0, 0, 0].reshape((npix, npix))
    assert np.allclose(eidos_data_xx, zernike_vals)


def test_zernike_func_xy_corr(coeff_xy, noll_index_xy, eidos_data_xy):
    """Tests reconstruction of xy correlation against eidos"""
    from africanus.rime import zernike_dde

    npix = 17
    nsrc = npix**2
    ntime = 1
    na = 1
    nchan = 1
    ncorr = 1
    thresh = 8
    npoly = thresh

    # Linear (l,m) grid
    nx, ny = npix, npix
    grid = (np.indices((nx, ny), dtype=float) - nx // 2) * 2 / nx
    ll, mm = grid[0], grid[1]

    lm = np.vstack((ll.flatten(), mm.flatten())).T

    # Initializing coords, coeffs, and noll_indices
    coords = np.empty((3, nsrc, ntime, na, nchan), dtype=float)
    coeffs = np.empty((na, nchan, ncorr, npoly), dtype=np.complex128)
    noll_indices = np.empty((na, nchan, ncorr, npoly))
    parallactic_angles = np.zeros((ntime, na), dtype=np.float64)
    frequency_scaling = np.ones((nchan,), dtype=np.float64)
    antenna_scaling = np.ones((na, nchan, 2), dtype=np.float64)
    pointing_errors = np.zeros((ntime, na, nchan, 2), dtype=np.float64)

    # Assign Values to coeffs and noll_indices
    coeffs[0, 0, 0, :] = coeff_xy[:thresh]
    noll_indices[0, 0, 0, :] = noll_index_xy[:thresh]

    # I left 0 as all the freq values
    coords[0, 0:nsrc, 0, 0, 0] = lm[0:nsrc, 0]
    coords[1, 0:nsrc, 0, 0, 0] = lm[0:nsrc, 1]
    coords[2, 0:nsrc, 0, 0, 0] = 0

    # Call the function, reshape accordingly, and normalise
    zernike_vals = zernike_dde(
        coords,
        coeffs,
        noll_indices,
        parallactic_angles,
        frequency_scaling,
        antenna_scaling,
        pointing_errors,
    )[:, 0, 0, 0].reshape((npix, npix))
    assert np.allclose(eidos_data_xy, zernike_vals)


def test_zernike_func_yx_corr(coeff_yx, noll_index_yx, eidos_data_yx):
    """Tests reconstruction of yx correlation against eidos"""
    from africanus.rime import zernike_dde

    npix = 17
    nsrc = npix**2
    ntime = 1
    na = 1
    nchan = 1
    ncorr = 1
    thresh = 8
    npoly = thresh

    # Linear (l,m) grid
    nx, ny = npix, npix
    grid = (np.indices((nx, ny), dtype=float) - nx // 2) * 2 / nx
    ll, mm = grid[0], grid[1]

    lm = np.vstack((ll.flatten(), mm.flatten())).T

    # Initializing coords, coeffs, and noll_indices
    coords = np.empty((3, nsrc, ntime, na, nchan), dtype=float)
    coeffs = np.empty((na, nchan, ncorr, npoly), dtype=np.complex128)
    noll_indices = np.empty((na, nchan, ncorr, npoly))
    parallactic_angles = np.zeros((ntime, na), dtype=np.float64)
    frequency_scaling = np.ones((nchan,), dtype=np.float64)
    antenna_scaling = np.ones((na, nchan, 2), dtype=np.float64)
    pointing_errors = np.zeros((ntime, na, nchan, 2), dtype=np.float64)

    # Assign Values to coeffs and noll_indices
    coeffs[0, 0, 0, :] = coeff_yx[:thresh]
    noll_indices[0, 0, 0, :] = noll_index_yx[:thresh]

    # I left 0 as all the freq values
    coords[0, 0:nsrc, 0, 0, 0] = lm[0:nsrc, 0]
    coords[1, 0:nsrc, 0, 0, 0] = lm[0:nsrc, 1]
    coords[2, 0:nsrc, 0, 0, 0] = 0

    # Call the function, reshape accordingly, and normalise
    zernike_vals = zernike_dde(
        coords,
        coeffs,
        noll_indices,
        parallactic_angles,
        frequency_scaling,
        antenna_scaling,
        pointing_errors,
    )[:, 0, 0, 0].reshape((npix, npix))
    assert np.allclose(eidos_data_yx, zernike_vals)


def test_zernike_func_yy_corr(coeff_yy, noll_index_yy, eidos_data_yy):
    """Tests reconstruction of yy correlation against eidos"""
    from africanus.rime import zernike_dde

    npix = 17
    nsrc = npix**2
    ntime = 1
    na = 1
    nchan = 1
    ncorr = 1
    thresh = 15
    npoly = thresh

    # Linear (l,m) grid
    nx, ny = npix, npix
    grid = (np.indices((nx, ny), dtype=float) - nx // 2) * 2 / nx
    ll, mm = grid[0], grid[1]

    lm = np.vstack((ll.flatten(), mm.flatten())).T

    # Initializing coords, coeffs, and noll_indices
    coords = np.empty((3, nsrc, ntime, na, nchan), dtype=float)
    coeffs = np.empty((na, nchan, ncorr, npoly), dtype=np.complex128)
    noll_indices = np.empty((na, nchan, 1, npoly))
    parallactic_angles = np.zeros((ntime, na), dtype=np.float64)
    frequency_scaling = np.ones((nchan,), dtype=np.float64)
    antenna_scaling = np.ones((na, nchan, 2), dtype=np.float64)
    pointing_errors = np.zeros((ntime, na, nchan, 2), dtype=np.float64)

    # Assign Values to coeffs and noll_indices
    coeffs[0, 0, 0, :] = coeff_yy[:thresh]
    noll_indices[0, 0, 0, :] = noll_index_yy[:thresh]

    # I left 0 as all the freq values
    coords[0, 0:nsrc, 0, 0, 0] = lm[0:nsrc, 0]
    coords[1, 0:nsrc, 0, 0, 0] = lm[0:nsrc, 1]
    coords[2, 0:nsrc, 0, 0, 0] = 0

    # Call the function, reshape accordingly, and normalise
    zernike_vals = zernike_dde(
        coords,
        coeffs,
        noll_indices,
        parallactic_angles,
        frequency_scaling,
        antenna_scaling,
        pointing_errors,
    )[:, 0, 0, 0].reshape((npix, npix))
    assert np.allclose(eidos_data_yy, zernike_vals)


def test_zernike_multiple_dims(coeff_xx, noll_index_xx):
    """Tests that we can call zernike_dde with multiple dimensions"""
    from africanus.rime import zernike_dde as np_zernike_dde

    npix = 17
    nsrc = npix**2
    ntime = 10
    na = 7
    nchan = 8
    corr1 = 2
    corr2 = 2
    npoly = 17

    # Linear (l,m) grid
    nx, ny = npix, npix
    grid = (np.indices((nx, ny), dtype=float) - nx // 2) * 2 / nx
    ll, mm = grid[0], grid[1]

    lm = np.vstack((ll.flatten(), mm.flatten())).T

    # Initializing coords, coeffs, and noll_indices
    coords = np.empty((3, nsrc, ntime, na, nchan), dtype=float)
    coeffs = np.empty((na, nchan, corr1, corr2, npoly), dtype=np.complex128)
    noll_indices = np.empty((na, nchan, corr1, corr2, npoly))

    parallactic_angles = np.zeros((ntime, na))
    frequency_scaling = np.ones((nchan,))
    antenna_scaling = np.ones((na, nchan, 2))
    pointing_errors = np.zeros((ntime, na, nchan, 2))

    # Assign Values to coeffs and noll_indices
    coeffs[:] = coeff_xx[:npoly]
    noll_indices[:] = noll_index_xx[:npoly]

    # I left 0 as all the freq values
    coords[0, :, :, :, :] = lm[:, 0, None, None, None]
    coords[1, :, :, :, :] = lm[:, 1, None, None, None]
    coords[2, :, :, :, :] = 0

    vals = np_zernike_dde(
        coords,
        coeffs,
        noll_indices,
        parallactic_angles,
        frequency_scaling,
        antenna_scaling,
        pointing_errors,
    )
    assert vals.shape == (nsrc, ntime, na, nchan, corr1, corr2)


def test_dask_zernike(coeff_xx, noll_index_xx):
    """Tests that dask zernike_dde agrees with numpy zernike_dde"""
    da = pytest.importorskip("dask.array")

    from africanus.rime.dask import zernike_dde
    from africanus.rime import zernike_dde as np_zernike_dde

    npix = 17
    nsrc = npix**2
    ntime = 10
    na = 7
    nchan = 8
    corr1 = 2
    corr2 = 2
    npoly = 17

    # Linear (l,m) grid
    nx, ny = npix, npix
    grid = (np.indices((nx, ny), dtype=float) - nx // 2) * 2 / nx
    ll, mm = grid[0], grid[1]

    lm = np.vstack((ll.flatten(), mm.flatten())).T

    # Initializing coords, coeffs, and noll_indices
    coords = np.empty((3, nsrc, ntime, na, nchan), dtype=float)
    coeffs = np.empty((na, nchan, corr1, corr2, npoly), dtype=np.complex128)
    noll_indices = np.empty((na, nchan, corr1, corr2, npoly))

    parallactic_angles = np.zeros((ntime, na))
    frequency_scaling = np.ones((nchan,))
    antenna_scaling = np.ones((na, nchan, 2))
    pointing_errors = np.zeros((ntime, na, nchan, 2))

    # Assign Values to coeffs and noll_indices
    coeffs[:] = coeff_xx[:npoly]
    noll_indices[:] = noll_index_xx[:npoly]

    # I left 0 as all the freq values
    coords[0, :, :, :, :] = lm[:, 0, None, None, None]
    coords[1, :, :, :, :] = lm[:, 1, None, None, None]
    coords[2, :, :, :, :] = 0

    vals = np_zernike_dde(
        coords,
        coeffs,
        noll_indices,
        parallactic_angles,
        frequency_scaling,
        antenna_scaling,
        pointing_errors,
    )
    assert vals.shape == (nsrc, ntime, na, nchan, corr1, corr2)

    # dimension chunking strategies
    time_c = (5, 5)
    ant_c = (4, 3)
    chan_c = (2, 4, 2)

    coords = da.from_array(coords, (3, npix, time_c, ant_c, chan_c))
    coeffs = da.from_array(coeffs, (ant_c, chan_c, corr1, corr2, npoly))
    noll_indices = da.from_array(noll_indices, (ant_c, chan_c, corr1, corr2, npoly))

    parallactic_angles = da.from_array(parallactic_angles)
    frequency_scaling = da.from_array(frequency_scaling)
    antenna_scaling = da.from_array(antenna_scaling)
    pointing_errors = da.from_array(pointing_errors)

    dask_vals = zernike_dde(
        coords,
        coeffs,
        noll_indices,
        parallactic_angles,
        frequency_scaling,
        antenna_scaling,
        pointing_errors,
    )

    assert np.all(vals == dask_vals.compute())


@pytest.fixture
def coeff_xx():
    return np.array(
        [
            -1.75402394e-01 - 0.14477493j,
            9.97613164e-02 + 0.0965587j,
            2.10125186e-01 + 0.17758039j,
            -1.69924807e-01 - 0.11709054j,
            -4.30692473e-02 - 0.0349753j,
            7.74099248e-02 + 0.03703381j,
            -7.51374250e-03 + 0.01024362j,
            1.40650300e-03 + 0.02095283j,
            -1.39579628e-02 - 0.01244837j,
            -7.93278560e-04 - 0.02543059j,
            3.61356760e-03 + 0.00202427j,
            2.31464542e-03 - 0.00018854j,
            9.05646002e-03 - 0.00062068j,
            -1.70722541e-04 - 0.00577695j,
            4.06321372e-03 - 0.00489419j,
            4.70079669e-03 - 0.0042618j,
            1.21656158e-02 + 0.01113621j,
        ]
    )


@pytest.fixture
def coeff_xy():
    return np.array(
        [
            -0.00378847 + 0.00520143j,
            0.02002285 + 0.02665323j,
            -0.00843154 + 0.00852609j,
            0.00449256 - 0.00522683j,
            -0.00478961 - 0.00633869j,
            -0.01326315 - 0.01646019j,
            -0.01497431 - 0.0140809j,
            -0.00117441 + 0.00205662j,
            -0.00048141 + 0.00075124j,
        ]
    )


@pytest.fixture
def coeff_yx():
    return np.array(
        [
            -2.23911814e-03 - 0.00547617j,
            -4.75247330e-03 - 0.00745264j,
            -2.21456777e-03 + 0.00619276j,
            1.20189576e-02 + 0.01197778j,
            -2.01741060e-02 - 0.01792336j,
            7.51580997e-05 + 0.00209391j,
            -3.31077481e-04 - 0.0036083j,
            1.16293179e-02 + 0.01279112j,
        ]
    )


@pytest.fixture
def coeff_yy():
    return np.array(
        [
            -0.17742637 - 0.1378773j,
            0.09912589 + 0.09639812j,
            0.21176327 + 0.17682041j,
            -0.16836034 - 0.11677519j,
            -0.0428337 - 0.03446249j,
            0.07525696 + 0.03761065j,
            -0.00754467 + 0.00811033j,
            0.01189913 + 0.01875151j,
            0.00248063 + 0.00179074j,
            0.00160786 + 0.00614232j,
            -0.01133655 - 0.01143651j,
            0.00470805 - 0.01920698j,
            0.0038768 - 0.00601548j,
            0.00172058 - 0.00385759j,
            -0.01082336 - 0.00432746j,
            -0.0009297 + 0.00796986j,
            0.01785803 + 0.00319331j,
        ]
    )


@pytest.fixture
def noll_index_xx():
    return np.array([10, 3, 21, 36, 0, 55, 16, 28, 37, 46, 23, 6, 15, 2, 5, 7, 57])


@pytest.fixture
def noll_index_xy():
    return np.array([12, 28, 22, 4, 38, 16, 46, 15, 7])


@pytest.fixture
def noll_index_yx():
    return np.array([12, 22, 4, 15, 29, 38, 7, 45])


@pytest.fixture
def noll_index_yy():
    return np.array([10, 3, 21, 36, 0, 55, 28, 16, 11, 23, 37, 46, 6, 2, 15, 5, 29])


@pytest.fixture
def eidos_data_xx():
    return np.array(
        [
            [
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                -2.11210942e-02 + 4.87466258e-03j,
                -2.49523224e-02 - 1.44627161e-04j,
                -2.47119038e-02 - 6.38713878e-04j,
                -2.43174644e-02 - 1.88136658e-04j,
                -1.87173442e-02 + 4.70992344e-03j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                -2.27466520e-02 - 3.01845791e-04j,
                -1.49939087e-02 + 2.14318015e-03j,
                -1.35785514e-02 + 5.40438739e-03j,
                -1.57185554e-02 + 6.88675083e-03j,
                -1.71256624e-02 + 7.34214429e-03j,
                -1.68165895e-02 + 6.96200374e-03j,
                -1.53753810e-02 + 5.52753170e-03j,
                -1.62011926e-02 + 2.22592041e-03j,
                -2.02072200e-02 - 4.75883781e-04j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                -6.05935583e-03 + 1.36356031e-02j,
                -1.46743487e-02 + 2.23268666e-04j,
                -1.03045469e-02 + 2.75050716e-03j,
                -1.85311515e-02 + 2.96017104e-03j,
                -2.13722953e-02 + 5.20600217e-03j,
                -1.86334271e-02 + 8.38900733e-03j,
                -1.70396502e-02 + 9.81511741e-03j,
                -1.97473818e-02 + 8.46535135e-03j,
                -2.38377885e-02 + 5.37497282e-03j,
                -2.23334873e-02 + 3.22076116e-03j,
                -1.41970227e-02 + 3.01727495e-03j,
                -1.47086646e-02 + 2.25620484e-04j,
                6.37383923e-03 + 1.27835038e-02j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                -1.14699692e-02 + 1.91828653e-04j,
                -9.74816027e-03 - 1.05549964e-03j,
                -2.12580505e-02 - 1.74694510e-03j,
                -1.18343254e-02 + 7.50979546e-03j,
                1.09370887e-02 + 1.95554681e-02j,
                2.88350734e-02 + 2.67547760e-02j,
                3.49422063e-02 + 2.87070404e-02j,
                2.87853471e-02 + 2.67581839e-02j,
                1.00612025e-02 + 1.96154963e-02j,
                -1.45993753e-02 + 7.69929599e-03j,
                -2.62821110e-02 - 1.40262503e-03j,
                -1.52383310e-02 - 6.79235071e-04j,
                -1.15111483e-02 + 1.94650834e-04j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                -1.68111272e-02 - 3.22153781e-04j,
                -5.08381749e-03 - 3.19513446e-03j,
                -2.13247574e-02 - 7.29009448e-03j,
                -5.34693718e-03 + 8.03925439e-03j,
                3.46502413e-02 + 2.85811706e-02j,
                5.92214222e-02 + 3.36425405e-02j,
                5.94283454e-02 + 2.45359622e-02j,
                5.56738500e-02 + 1.82591227e-02j,
                6.09841467e-02 + 2.44293365e-02j,
                6.11157142e-02 + 3.35127167e-02j,
                3.49382658e-02 + 2.85614310e-02j,
                -8.35765870e-03 + 8.24559184e-03j,
                -2.76048331e-02 - 6.85969440e-03j,
                -1.09225312e-02 - 2.79498277e-03j,
                -1.23671212e-02 - 6.26720263e-04j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                -2.21714183e-03 - 9.64134357e-04j,
                -1.58425631e-02 - 1.21520530e-02j,
                -1.38059048e-02 - 9.46823895e-04j,
                3.44479636e-02 + 2.89727594e-02j,
                6.34419949e-02 + 3.37054730e-02j,
                2.74099037e-02 - 6.18315963e-03j,
                -4.21501396e-02 - 6.27517715e-02j,
                -7.56417340e-02 - 8.91064748e-02j,
                -3.88883886e-02 - 6.29753131e-02j,
                3.23731905e-02 - 6.52331462e-03j,
                6.74762505e-02 + 3.34289884e-02j,
                3.48319961e-02 + 2.89464400e-02j,
                -1.84143213e-02 - 6.30989671e-04j,
                -2.34472348e-02 - 1.16308727e-02j,
                -5.03413765e-03 - 7.71073749e-04j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                -2.01120010e-02 - 3.18299651e-04j,
                -6.38127267e-04 - 7.32965860e-03j,
                -2.19478194e-02 - 1.56485648e-02j,
                8.55557748e-03 + 1.41873799e-02j,
                6.15343990e-02 + 4.04633132e-02j,
                3.11177013e-02 - 6.42311211e-04j,
                -1.12513032e-01 - 1.14676222e-01j,
                -2.83605497e-01 - 2.38697723e-01j,
                -3.57898069e-01 - 2.92111096e-01j,
                -2.78880279e-01 - 2.39021563e-01j,
                -1.04867743e-01 - 1.15200186e-01j,
                3.85626316e-02 - 1.15254369e-03j,
                6.53229830e-02 + 4.02036656e-02j,
                6.36586209e-03 + 1.43374504e-02j,
                -2.93442991e-02 - 1.51416528e-02j,
                -6.92703069e-03 - 6.89865351e-03j,
                -1.04970009e-02 - 9.77256175e-04j,
            ],
            [
                -1.41198353e-02 + 4.86843324e-04j,
                -4.11519319e-03 - 1.42661566e-02j,
                -2.09421413e-02 - 1.42734949e-02j,
                2.73817044e-02 + 2.72075307e-02j,
                6.48211451e-02 + 3.83114725e-02j,
                -3.46093268e-02 - 5.44161715e-02j,
                -2.79881088e-01 - 2.40274349e-01j,
                -5.40296479e-01 - 4.26004740e-01j,
                -6.49620813e-01 - 5.03622409e-01j,
                -5.34595208e-01 - 4.26395472e-01j,
                -2.70430651e-01 - 2.40922027e-01j,
                -2.48240737e-02 - 5.50867962e-02j,
                7.10443507e-02 + 3.78849700e-02j,
                2.71330731e-02 + 2.72245704e-02j,
                -2.76258696e-02 - 1.38154308e-02j,
                -1.18014323e-02 - 1.37393862e-02j,
                -9.04097130e-03 + 1.38767345e-04j,
            ],
            [
                -1.13641209e-02 + 8.37287731e-04j,
                -5.57008400e-03 - 1.86115032e-02j,
                -1.93297709e-02 - 1.37301597e-02j,
                3.38992125e-02 + 3.32851772e-02j,
                6.12116693e-02 + 3.62251129e-02j,
                -6.89229675e-02 - 8.22376479e-02j,
                -3.56190661e-01 - 3.03164911e-01j,
                -6.52454459e-01 - 5.18739447e-01j,
                -7.75692875e-01 - 6.08013674e-01j,
                -6.46411508e-01 - 5.19153596e-01j,
                -3.46105852e-01 - 3.03856066e-01j,
                -5.83086204e-02 - 8.29650939e-02j,
                6.83117305e-02 + 3.57385157e-02j,
                3.43792532e-02 + 3.32522780e-02j,
                -2.56779427e-02 - 1.32950927e-02j,
                -1.36078001e-02 - 1.80606446e-02j,
                -7.66667196e-03 + 5.83885955e-04j,
            ],
            [
                -1.42690718e-02 + 1.49656856e-03j,
                -3.35458707e-03 - 1.96098160e-02j,
                -2.02780995e-02 - 1.76949864e-02j,
                2.71609094e-02 + 3.10590350e-02j,
                6.29897261e-02 + 4.42240261e-02j,
                -3.85221788e-02 - 5.66076645e-02j,
                -2.85874863e-01 - 2.57606475e-01j,
                -5.47826861e-01 - 4.57587344e-01j,
                -6.57716941e-01 - 5.40974654e-01j,
                -5.42125590e-01 - 4.57978076e-01j,
                -2.76424427e-01 - 2.58254153e-01j,
                -2.87369256e-02 - 5.72782892e-02j,
                6.92129317e-02 + 4.37975235e-02j,
                2.69122782e-02 + 3.10760747e-02j,
                -2.69618278e-02 - 1.72369223e-02j,
                -1.10408261e-02 - 1.90830456e-02j,
                -9.19020781e-03 + 1.14849258e-03j,
            ],
            [
                -2.11059703e-02 + 2.47171815e-03j,
                7.03031713e-04 - 1.60554144e-02j,
                -2.04239143e-02 - 2.52219264e-02j,
                8.78974990e-03 + 1.78753461e-02j,
                5.91734254e-02 + 5.29977022e-02j,
                2.52887111e-02 + 4.96012317e-03j,
                -1.21870083e-01 - 1.28891652e-01j,
                -2.95593047e-01 - 2.73361976e-01j,
                -3.70858387e-01 - 3.35311000e-01j,
                -2.90867829e-01 - 2.73685815e-01j,
                -1.14224794e-01 - 1.29415616e-01j,
                3.27336414e-02 + 4.44989069e-03j,
                6.29620094e-02 + 5.27380546e-02j,
                6.60003451e-03 + 1.80254165e-02j,
                -2.78203940e-02 - 2.47150145e-02j,
                -5.58587171e-03 - 1.56244093e-02j,
                -1.14909702e-02 + 1.81276163e-03j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                -9.40360553e-04 - 7.04582442e-03j,
                -1.33821588e-02 - 2.97361136e-02j,
                -1.22698438e-02 - 5.84149963e-03j,
                3.33895807e-02 + 4.23073620e-02j,
                5.86414592e-02 + 5.22726116e-02j,
                1.86664184e-02 + 2.22049194e-03j,
                -5.38886955e-02 - 6.93262505e-02j,
                -8.84971524e-02 - 1.02533445e-01j,
                -5.06269444e-02 - 6.95497921e-02j,
                2.36297053e-02 + 1.88033695e-03j,
                6.26757148e-02 + 5.19961270e-02j,
                3.37736133e-02 + 4.22810427e-02j,
                -1.68782603e-02 - 5.52566540e-03j,
                -2.09868305e-02 - 2.92149333e-02j,
                -3.75735637e-03 - 6.85276381e-03j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                -1.74080733e-02 + 3.71674715e-03j,
                -2.26317426e-03 - 2.21422473e-02j,
                -1.81902309e-02 - 2.78017657e-02j,
                -3.83916843e-03 + 6.92820782e-03j,
                3.32390642e-02 + 4.63606408e-02j,
                5.44994751e-02 + 5.87113186e-02j,
                5.21026692e-02 + 4.81861763e-02j,
                4.73627034e-02 + 4.00763177e-02j,
                5.36584706e-02 + 4.80795507e-02j,
                5.63937670e-02 + 5.85814947e-02j,
                3.35270886e-02 + 4.63409012e-02j,
                -6.84988995e-03 + 7.13454526e-03j,
                -2.44703066e-02 - 2.73713656e-02j,
                -8.10188802e-03 - 2.17420956e-02j,
                -1.29640673e-02 + 3.41218067e-03j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                -1.03912443e-02 - 2.47452282e-03j,
                -5.94512966e-03 - 2.77737966e-02j,
                -1.73398924e-02 - 2.73865341e-02j,
                -9.27422376e-03 - 6.47997425e-04j,
                1.15225197e-02 + 2.87753835e-02j,
                2.77310986e-02 + 4.60122974e-02j,
                3.31782348e-02 + 5.09313782e-02j,
                2.76813723e-02 + 4.60157053e-02j,
                1.06466335e-02 + 2.88354117e-02j,
                -1.20392736e-02 - 4.58496891e-04j,
                -2.23639530e-02 - 2.70422140e-02j,
                -1.14353004e-02 - 2.73975320e-02j,
                -1.04324234e-02 - 2.47170064e-03j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                -1.25267729e-02 + 1.43661818e-02j,
                -1.33798788e-02 - 2.97635310e-03j,
                -6.07358203e-03 - 2.56701621e-02j,
                -1.36103429e-02 - 3.22079502e-02j,
                -1.68005799e-02 - 2.35140829e-02j,
                -1.46491767e-02 - 1.21399419e-02j,
                -1.33213331e-02 - 7.37716445e-03j,
                -1.57631314e-02 - 1.20635979e-02j,
                -1.92660732e-02 - 2.33451122e-02j,
                -1.74126788e-02 - 3.19473600e-02j,
                -9.96605786e-03 - 2.54033943e-02j,
                -1.34141948e-02 - 2.97400128e-03j,
                -9.35778591e-05 + 1.35140825e-02j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                -2.37913075e-02 + 6.76623084e-03j,
                -1.20147524e-02 - 1.20474300e-02j,
                -8.88449501e-03 - 2.51357580e-02j,
                -1.03943125e-02 - 3.05188649e-02j,
                -1.16548016e-02 - 3.16317973e-02j,
                -1.14923467e-02 - 3.04436120e-02j,
                -1.06813246e-02 - 2.50126137e-02j,
                -1.32220363e-02 - 1.19646897e-02j,
                -2.12518755e-02 + 6.59219285e-03j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                -2.50969717e-02 + 1.60347338e-02j,
                -2.61462145e-02 + 7.93317470e-03j,
                -2.50963732e-02 + 5.02043186e-03j,
                -2.55113565e-02 + 7.88966521e-03j,
                -2.26932217e-02 + 1.58699947e-02j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
            ],
        ]
    )


@pytest.fixture
def eidos_data_xy():
    return np.array(
        [
            [
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                5.60982470e-04 + 2.30571195e-03j,
                8.90492910e-04 + 2.18818926e-03j,
                2.91099273e-04 + 2.40008326e-03j,
                -1.10666090e-04 + 2.06965268e-03j,
                2.92594213e-03 + 1.28605700e-03j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                2.11378405e-03 + 2.03420260e-03j,
                6.80148798e-03 + 1.48303825e-03j,
                6.80009892e-03 + 8.76525928e-04j,
                4.61511642e-03 + 1.00077457e-03j,
                2.21759940e-03 + 2.46363862e-03j,
                1.50604160e-04 + 4.91418638e-03j,
                -1.98111945e-03 + 6.86625088e-03j,
                -4.09320830e-03 + 5.96424724e-03j,
                -1.43143558e-03 + 1.69140910e-03j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                -5.62786683e-03 + 3.34726414e-03j,
                6.33115955e-03 + 1.02604501e-03j,
                9.70158781e-03 - 2.46394684e-03j,
                5.81877644e-03 - 7.11362110e-03j,
                1.42873009e-04 - 1.02851992e-02j,
                -4.08383571e-03 - 1.09463832e-02j,
                -6.13171523e-03 - 9.27074382e-03j,
                -6.36213165e-03 - 5.44925945e-03j,
                -5.45007915e-03 + 2.95180091e-04j,
                -4.64111635e-03 + 6.35160472e-03j,
                -5.46194827e-03 + 8.73754708e-03j,
                -5.33329782e-03 + 4.05590935e-03j,
                1.64369819e-02 + 1.81276177e-03j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                6.24755941e-03 + 6.03328088e-04j,
                9.57267968e-03 - 4.69764138e-03j,
                3.30338664e-03 - 1.18338494e-02j,
                -4.49927768e-03 - 1.61927689e-02j,
                -9.18492421e-03 - 1.65320885e-02j,
                -1.06782718e-02 - 1.47680530e-02j,
                -1.08940547e-02 - 1.31156686e-02j,
                -1.10118955e-02 - 1.17848844e-02j,
                -1.05784115e-02 - 8.89259316e-03j,
                -8.76158049e-03 - 2.70603648e-03j,
                -6.44014518e-03 + 5.21106691e-03j,
                -6.16859355e-03 + 8.92261349e-03j,
                -5.41600797e-03 + 3.63163388e-03j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                2.09105590e-03 + 1.01960366e-03j,
                8.86879042e-03 - 3.28856260e-03j,
                3.53562458e-03 - 1.10289578e-02j,
                -4.71069364e-03 - 1.56463030e-02j,
                -8.43364497e-03 - 1.36467748e-02j,
                -6.55141240e-03 - 7.09854519e-03j,
                -2.57389408e-03 - 1.00930970e-03j,
                -6.84660925e-04 + 6.20499701e-04j,
                -2.55546906e-03 - 2.49447064e-03j,
                -6.46714203e-03 - 6.78353042e-03j,
                -8.99684258e-03 - 7.33829502e-03j,
                -8.21795843e-03 - 1.94740604e-03j,
                -6.04503142e-03 + 5.73073177e-03j,
                -6.04236406e-03 + 7.47096276e-03j,
                -1.70114249e-03 + 1.10931731e-03j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                5.92332277e-03 - 4.61983943e-04j,
                5.27782435e-03 - 6.49138333e-03j,
                -1.96662684e-03 - 1.22037040e-02j,
                -6.24860902e-03 - 1.10345423e-02j,
                -3.09207506e-03 - 1.94805333e-03j,
                5.50880214e-03 + 1.05893394e-02j,
                1.39794565e-02 + 2.00142151e-02j,
                1.71783737e-02 + 2.16013844e-02j,
                1.33684475e-02 + 1.50066088e-02j,
                4.96588428e-03 + 4.42090564e-03j,
                -3.12835657e-03 - 3.73291339e-03j,
                -6.82425665e-03 - 4.70426011e-03j,
                -5.98988806e-03 + 8.64420771e-04j,
                -4.68899430e-03 + 6.11037514e-03j,
                -4.76263148e-03 + 3.65367772e-03j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                -2.79048915e-04 + 1.40002038e-04j,
                4.78777630e-03 - 1.37870199e-03j,
                1.59222484e-03 - 6.39530729e-03j,
                -3.34110999e-03 - 8.75550616e-03j,
                -3.17395108e-03 - 3.84311240e-03j,
                3.92392666e-03 + 7.80585531e-03j,
                1.48257359e-02 + 2.14381888e-02j,
                2.41138919e-02 + 3.10505503e-02j,
                2.72606996e-02 + 3.24892285e-02j,
                2.28797092e-02 + 2.54009462e-02j,
                1.32608303e-02 + 1.33707824e-02j,
                3.05919762e-03 + 2.20097468e-03j,
                -3.33532613e-03 - 3.09792541e-03j,
                -4.56422428e-03 - 1.41436651e-03j,
                -3.36129356e-03 + 3.06530092e-03j,
                -3.41092502e-03 + 3.59092393e-03j,
                1.15078006e-03 + 7.57940200e-04j,
            ],
            [
                8.37460561e-04 - 1.79208273e-04j,
                2.14549660e-03 - 7.86155382e-04j,
                -9.24840699e-05 - 3.48244921e-03j,
                -2.01510157e-03 - 4.12429347e-03j,
                -3.47756255e-04 - 2.25349351e-04j,
                5.28646139e-03 + 7.59990046e-03j,
                1.26718668e-02 + 1.64011544e-02j,
                1.85424832e-02 + 2.26071345e-02j,
                2.03111976e-02 + 2.37405936e-02j,
                1.72638688e-02 + 1.94926662e-02j,
                1.08249337e-02 + 1.18245938e-02j,
                3.82950660e-03 + 4.07370753e-03j,
                -9.34584529e-04 - 6.50595733e-04j,
                -2.32293189e-03 - 1.18629401e-03j,
                -1.64851049e-03 + 7.49842107e-04j,
                -1.46467938e-03 + 1.63114980e-03j,
                -7.39982208e-04 + 7.11438515e-04j,
            ],
            [
                2.39736377e-04 - 4.19824360e-04j,
                -5.21152012e-04 + 9.12637091e-04j,
                -4.11604795e-04 + 7.20798911e-04j,
                3.11250340e-05 - 5.45059021e-05j,
                4.60356044e-04 - 8.06171696e-04j,
                6.88216438e-04 - 1.20519893e-03j,
                6.53882126e-04 - 1.14507297e-03j,
                3.91814799e-04 - 6.86142833e-04j,
                0.00000000e00 + 0.00000000e00j,
                -3.91814799e-04 + 6.86142833e-04j,
                -6.53882126e-04 + 1.14507297e-03j,
                -6.88216438e-04 + 1.20519893e-03j,
                -4.60356044e-04 + 8.06171696e-04j,
                -3.11250340e-05 + 5.45059021e-05j,
                4.11604795e-04 - 7.20798911e-04j,
                5.21152012e-04 - 9.12637091e-04j,
                -2.39736377e-04 + 4.19824360e-04j,
            ],
            [
                -1.78850539e-04 - 9.74144147e-04j,
                -3.14222226e-03 + 2.53161311e-03j,
                -7.74239360e-04 + 5.00024811e-03j,
                1.98285990e-03 + 4.18075481e-03j,
                1.15476065e-03 - 1.18787006e-03j,
                -4.01754264e-03 - 9.82202048e-03j,
                -1.14463660e-02 - 1.85472414e-02j,
                -1.78031615e-02 - 2.39018286e-02j,
                -2.03111976e-02 - 2.37405936e-02j,
                -1.80031905e-02 - 1.81979721e-02j,
                -1.20504345e-02 - 9.67850688e-03j,
                -5.09842535e-03 - 1.85158750e-03j,
                1.27580139e-04 + 2.06381514e-03j,
                2.35517355e-03 + 1.12983268e-03j,
                2.51523392e-03 - 2.26764101e-03j,
                2.46140503e-03 - 3.37660753e-03j,
                8.13721867e-05 + 4.41913905e-04j,
            ],
            [
                1.52588982e-03 - 2.32345952e-03j,
                -5.60330016e-03 + 2.80684063e-03j,
                -2.55137548e-03 + 8.07496395e-03j,
                3.05715506e-03 + 9.25276569e-03j,
                3.66524192e-03 + 2.98276795e-03j,
                -2.95849312e-03 - 9.49651455e-03j,
                -1.38343205e-02 - 2.31743473e-02j,
                -2.35011414e-02 - 3.21235938e-02j,
                -2.72606996e-02 - 3.24892285e-02j,
                -2.34924596e-02 - 2.43279028e-02j,
                -1.42522457e-02 - 1.16346239e-02j,
                -4.02463116e-03 - 5.10315452e-04j,
                2.84403530e-03 + 3.95826986e-03j,
                4.84817921e-03 + 9.17106983e-04j,
                4.32044420e-03 - 4.74495758e-03j,
                4.22644889e-03 - 5.01906258e-03j,
                -2.39762097e-03 + 1.42551728e-03j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                -6.28862132e-03 + 1.10169175e-03j,
                -6.26397262e-03 + 8.21831803e-03j,
                1.36902287e-03 + 1.32502231e-02j,
                6.29840908e-03 + 1.09473328e-02j,
                3.61522377e-03 + 1.03191963e-03j,
                -4.86517978e-03 - 1.17164455e-02j,
                -1.35564835e-02 - 2.07549218e-02j,
                -1.71783737e-02 - 2.16013844e-02j,
                -1.37914204e-02 - 1.42659022e-02j,
                -5.60950664e-03 - 3.29379949e-03j,
                2.60520786e-03 + 4.64904709e-03j,
                6.77445659e-03 + 4.79146955e-03j,
                6.58749203e-03 - 1.91093988e-03j,
                5.67514258e-03 - 7.83730984e-03j,
                5.12793003e-03 - 4.29338553e-03j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                -1.51477213e-03 - 2.02878702e-03j,
                -9.62593520e-03 + 4.61446828e-03j,
                -4.35000370e-03 + 1.24550918e-02j,
                4.32027339e-03 + 1.63300038e-02j,
                8.47099501e-03 + 1.35813677e-02j,
                6.79705782e-03 + 6.66837296e-03j,
                2.77564518e-03 + 6.56004847e-04j,
                6.84660925e-04 - 6.20499701e-04j,
                2.35371796e-03 + 2.84777549e-03j,
                6.22149661e-03 + 7.21370265e-03j,
                8.95949254e-03 + 7.40370210e-03j,
                8.60837868e-03 + 1.26370533e-03j,
                6.85941053e-03 - 7.15686573e-03j,
                6.79950884e-03 - 8.79686844e-03j,
                1.12485872e-03 - 1.00133944e-04j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                -6.25289938e-03 - 5.93976770e-04j,
                -1.02846266e-02 + 5.94439690e-03j,
                -3.95488993e-03 + 1.29747566e-02j,
                4.14071530e-03 + 1.68206804e-02j,
                9.07134224e-03 + 1.67309923e-02j,
                1.06718235e-02 + 1.47793452e-02j,
                1.08940547e-02 + 1.31156686e-02j,
                1.10183438e-02 + 1.17735922e-02j,
                1.06919934e-02 + 8.69368935e-03j,
                9.12014287e-03 + 2.07812502e-03j,
                7.09164847e-03 - 6.35197408e-03j,
                6.88054045e-03 - 1.01693690e-02j,
                5.42134794e-03 - 3.64098519e-03j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                7.24016179e-03 - 6.17070174e-03j,
                -6.33560953e-03 - 1.01825224e-03j,
                -1.02063510e-02 + 3.34788396e-03j,
                -6.31185057e-03 + 7.97708845e-03j,
                -4.62589888e-04 + 1.08450848e-02j,
                3.93938180e-03 + 1.11993497e-02j,
                6.13171523e-03 + 9.27074382e-03j,
                6.50658556e-03 + 5.19629297e-03j,
                5.76979603e-03 - 8.55065644e-04j,
                5.13419049e-03 - 7.21507207e-03j,
                5.96671146e-03 - 9.62148420e-03j,
                5.33774780e-03 - 4.06370211e-03j,
                -1.80492769e-02 + 1.01067582e-03j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                -1.78447904e-03 - 2.61087881e-03j,
                -6.95804450e-03 - 1.20887777e-03j,
                -7.03310573e-03 - 4.68486315e-04j,
                -4.75750580e-03 - 7.51423469e-04j,
                -2.21759940e-03 - 2.46363862e-03j,
                -8.21478073e-06 - 5.16353748e-03j,
                2.21412627e-03 - 7.27429049e-03j,
                4.24976483e-03 - 6.23840772e-03j,
                1.10213057e-03 - 1.11473289e-03j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                -2.49272243e-04 - 2.85157632e-03j,
                -8.08166657e-04 - 2.33235831e-03j,
                -2.91099273e-04 - 2.40008326e-03j,
                2.83398372e-05 - 1.92548363e-03j,
                -3.23765236e-03 - 7.40192633e-04j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
            ],
        ]
    )


@pytest.fixture
def eidos_data_yx():
    return np.array(
        [
            [
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                -2.92303197e-03 - 2.13190285e-04j,
                -1.06480305e-03 + 5.19881548e-04j,
                4.72733362e-19 - 3.38537894e-19j,
                1.06480305e-03 - 5.19881548e-04j,
                2.92303197e-03 + 2.13190285e-04j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                -3.78602439e-03 + 1.93154709e-03j,
                -1.57713870e-03 + 3.89804882e-03j,
                -5.61537788e-04 + 3.51761763e-03j,
                -8.70733043e-05 + 2.05679809e-03j,
                4.91466284e-21 - 9.27127125e-19j,
                8.70733043e-05 - 2.05679809e-03j,
                5.61537788e-04 - 3.51761763e-03j,
                1.57713870e-03 - 3.89804882e-03j,
                3.78602439e-03 - 1.93154709e-03j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                -1.09594235e-02 - 6.41539529e-03j,
                -3.04802552e-03 + 4.37368661e-03j,
                -9.96314506e-04 + 6.22172653e-03j,
                8.78961883e-04 + 6.69631275e-03j,
                1.67793810e-03 + 5.62198517e-03j,
                1.18073464e-03 + 3.14118831e-03j,
                -4.73365637e-19 - 1.18993549e-18j,
                -1.18073464e-03 - 3.14118831e-03j,
                -1.67793810e-03 - 5.62198517e-03j,
                -8.78961883e-04 - 6.69631275e-03j,
                9.96314506e-04 - 6.22172653e-03j,
                3.04802552e-03 - 4.37368661e-03j,
                1.09594235e-02 + 6.41539529e-03j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                -3.29614029e-03 + 4.49277526e-03j,
                -8.26126276e-04 + 6.98111300e-03j,
                2.68355437e-03 + 9.11201081e-03j,
                4.38581398e-03 + 8.98927252e-03j,
                3.54128616e-03 + 6.33756768e-03j,
                1.70389541e-03 + 2.97340008e-03j,
                -4.96307468e-19 - 8.68953090e-19j,
                -1.70389541e-03 - 2.97340008e-03j,
                -3.54128616e-03 - 6.33756768e-03j,
                -4.38581398e-03 - 8.98927252e-03j,
                -2.68355437e-03 - 9.11201081e-03j,
                8.26126276e-04 - 6.98111300e-03j,
                3.29614029e-03 - 4.49277526e-03j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                -4.14130674e-03 + 2.60331713e-03j,
                -1.82649340e-03 + 5.73811622e-03j,
                3.08836790e-03 + 9.30809644e-03j,
                6.50372026e-03 + 1.09300264e-02j,
                5.08899066e-03 + 7.71876613e-03j,
                1.49184566e-03 + 2.69504881e-03j,
                -4.55916052e-04 - 8.18401999e-05j,
                2.42129930e-19 + 1.72172365e-19j,
                4.55916052e-04 + 8.18401999e-05j,
                -1.49184566e-03 - 2.69504881e-03j,
                -5.08899066e-03 - 7.71876613e-03j,
                -6.50372026e-03 - 1.09300264e-02j,
                -3.08836790e-03 - 9.30809644e-03j,
                1.82649340e-03 - 5.73811622e-03j,
                4.14130674e-03 - 2.60331713e-03j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                -2.99858833e-03 + 3.80418326e-03j,
                7.77133774e-04 + 6.37472259e-03j,
                7.09458631e-03 + 1.09135155e-02j,
                7.08638891e-03 + 9.23051163e-03j,
                6.07398387e-04 + 1.29328563e-03j,
                -5.49400694e-03 - 5.72863545e-03j,
                -5.54312564e-03 - 5.94018326e-03j,
                1.22698245e-18 + 1.31972407e-18j,
                5.54312564e-03 + 5.94018326e-03j,
                5.49400694e-03 + 5.72863545e-03j,
                -6.07398387e-04 - 1.29328563e-03j,
                -7.08638891e-03 - 9.23051163e-03j,
                -7.09458631e-03 - 1.09135155e-02j,
                -7.77133774e-04 - 6.37472259e-03j,
                2.99858833e-03 - 3.80418326e-03j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                -3.56129685e-03 + 6.63347893e-04j,
                -2.69508762e-03 + 2.41047073e-03j,
                3.81000866e-03 + 6.79467072e-03j,
                9.21922956e-03 + 1.06203256e-02j,
                4.23741513e-03 + 4.50106222e-03j,
                -7.32748588e-03 - 7.96144401e-03j,
                -1.49861518e-02 - 1.60474016e-02j,
                -1.18835426e-02 - 1.26774566e-02j,
                1.66691615e-18 + 1.77668263e-18j,
                1.18835426e-02 + 1.26774566e-02j,
                1.49861518e-02 + 1.60474016e-02j,
                7.32748588e-03 + 7.96144401e-03j,
                -4.23741513e-03 - 4.50106222e-03j,
                -9.21922956e-03 - 1.06203256e-02j,
                -3.81000866e-03 - 6.79467072e-03j,
                2.69508762e-03 - 2.41047073e-03j,
                3.56129685e-03 - 6.63347893e-04j,
            ],
            [
                -1.89379518e-03 + 2.08734496e-03j,
                -2.25070410e-03 + 5.25501368e-04j,
                5.89276460e-03 + 6.23275007e-03j,
                9.74677690e-03 + 9.08217823e-03j,
                7.58759781e-04 - 1.98093327e-04j,
                -1.43491022e-02 - 1.54805207e-02j,
                -2.27223032e-02 - 2.38551450e-02j,
                -1.68861239e-02 - 1.76253544e-02j,
                1.16532525e-18 + 1.21488455e-18j,
                1.68861239e-02 + 1.76253544e-02j,
                2.27223032e-02 + 2.38551450e-02j,
                1.43491022e-02 + 1.54805207e-02j,
                -7.58759781e-04 + 1.98093327e-04j,
                -9.74677690e-03 - 9.08217823e-03j,
                -5.89276460e-03 - 6.23275007e-03j,
                2.25070410e-03 - 5.25501368e-04j,
                1.89379518e-03 - 2.08734496e-03j,
            ],
            [
                -1.14744144e-03 + 1.71422065e-03j,
                -2.21960396e-03 - 1.72880313e-03j,
                6.49135711e-03 + 4.39896738e-03j,
                9.98699125e-03 + 7.55872748e-03j,
                -1.73297283e-04 - 1.90592397e-03j,
                -1.63763593e-02 - 1.74507814e-02j,
                -2.50094301e-02 - 2.56897046e-02j,
                -1.83800007e-02 - 1.87444263e-02j,
                0.00000000e00 + 0.00000000e00j,
                1.83800007e-02 + 1.87444263e-02j,
                2.50094301e-02 + 2.56897046e-02j,
                1.63763593e-02 + 1.74507814e-02j,
                1.73297283e-04 + 1.90592397e-03j,
                -9.98699125e-03 - 7.55872748e-03j,
                -6.49135711e-03 - 4.39896738e-03j,
                2.21960396e-03 + 1.72880313e-03j,
                1.14744144e-03 - 1.71422065e-03j,
            ],
            [
                -1.04397988e-06 + 1.49542856e-03j,
                -2.79776776e-03 - 4.09852705e-03j,
                5.41610730e-03 + 1.18699815e-03j,
                1.03604268e-02 + 6.18976715e-03j,
                2.48037577e-03 - 1.11915012e-04j,
                -1.20688275e-02 - 1.31404917e-02j,
                -2.06327391e-02 - 2.08556086e-02j,
                -1.56506255e-02 - 1.56163353e-02j,
                1.08557565e-18 + 1.08064561e-18j,
                1.56506255e-02 + 1.56163353e-02j,
                2.06327391e-02 + 2.08556086e-02j,
                1.20688275e-02 + 1.31404917e-02j,
                -2.48037577e-03 + 1.11915012e-04j,
                -1.03604268e-02 - 6.18976715e-03j,
                -5.41610730e-03 - 1.18699815e-03j,
                2.79776776e-03 + 4.09852705e-03j,
                1.04397988e-06 - 1.49542856e-03j,
            ],
            [
                1.85925717e-03 + 1.67408725e-03j,
                -3.27885190e-03 - 5.51048207e-03j,
                2.58620303e-03 - 3.27661407e-03j,
                9.70724844e-03 + 3.65553410e-03j,
                6.74486275e-03 + 2.72299142e-03j,
                -3.67338779e-03 - 5.43540739e-03j,
                -1.15100940e-02 - 1.18506663e-02j,
                -9.79397856e-03 - 9.67792018e-03j,
                1.39585571e-18 + 1.36939883e-18j,
                9.79397856e-03 + 9.67792018e-03j,
                1.15100940e-02 + 1.18506663e-02j,
                3.67338779e-03 + 5.43540739e-03j,
                -6.74486275e-03 - 2.72299142e-03j,
                -9.70724844e-03 - 3.65553410e-03j,
                -2.58620303e-03 + 3.27661407e-03j,
                3.27885190e-03 + 5.51048207e-03j,
                -1.85925717e-03 - 1.67408725e-03j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                -1.97648540e-03 - 4.13271271e-03j,
                -1.18444621e-03 - 7.66108323e-03j,
                6.44927535e-03 - 1.29230044e-03j,
                8.89279709e-03 + 2.86345233e-03j,
                4.11561122e-03 + 7.09503395e-04j,
                -1.83990884e-03 - 3.20259882e-03j,
                -3.26285091e-03 - 3.60015420e-03j,
                7.78543151e-19 + 8.17380291e-19j,
                3.26285091e-03 + 3.60015420e-03j,
                1.83990884e-03 + 3.20259882e-03j,
                -4.11561122e-03 - 7.09503395e-04j,
                -8.89279709e-03 - 2.86345233e-03j,
                -6.44927535e-03 + 1.29230044e-03j,
                1.18444621e-03 + 7.66108323e-03j,
                1.97648540e-03 + 4.13271271e-03j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                2.48332247e-03 + 5.31609706e-04j,
                -3.15457994e-03 - 8.63977809e-03j,
                9.59767429e-04 - 7.34724019e-03j,
                6.32807109e-03 - 1.51985482e-03j,
                6.89539884e-03 + 1.35170684e-03j,
                3.99929328e-03 + 9.16978010e-04j,
                1.26569994e-03 + 4.33811522e-06j,
                -2.20907134e-19 + 6.12360038e-20j,
                -1.26569994e-03 - 4.33811522e-06j,
                -3.99929328e-03 - 9.16978010e-04j,
                -6.89539884e-03 - 1.35170684e-03j,
                -6.32807109e-03 + 1.51985482e-03j,
                -9.59767429e-04 + 7.34724019e-03j,
                3.15457994e-03 + 8.63977809e-03j,
                -2.48332247e-03 - 5.31609706e-04j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                3.18763077e-04 - 3.06371154e-03j,
                -2.77992505e-03 - 9.53327420e-03j,
                5.54953896e-04 - 7.54332582e-03j,
                3.74050302e-03 - 3.21654346e-03j,
                4.02930504e-03 - 6.27223803e-04j,
                2.31754534e-03 + 8.09890001e-05j,
                -7.26746219e-19 - 5.67240505e-20j,
                -2.31754534e-03 - 8.09890001e-05j,
                -4.02930504e-03 + 6.27223803e-04j,
                -3.74050302e-03 + 3.21654346e-03j,
                -5.54953896e-04 + 7.54332582e-03j,
                2.77992505e-03 + 9.53327420e-03j,
                -3.18763077e-04 + 3.06371154e-03j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                6.86078592e-03 + 2.43118373e-03j,
                5.66877844e-04 - 3.18280018e-03j,
                -2.32440105e-03 - 8.15616778e-03j,
                -1.08261810e-03 - 7.33949307e-03j,
                4.54132466e-04 - 4.44929962e-03j,
                7.04077342e-04 - 1.90456361e-03j,
                -3.21596334e-19 + 6.51217358e-19j,
                -7.04077342e-04 + 1.90456361e-03j,
                -4.54132466e-04 + 4.44929962e-03j,
                1.08261810e-03 + 7.33949307e-03j,
                2.32440105e-03 + 8.15616778e-03j,
                -5.66877844e-04 + 3.18280018e-03j,
                -6.86078592e-03 - 2.43118373e-03j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                2.83860481e-03 - 1.40160329e-04j,
                -5.55035762e-04 - 4.03884715e-03j,
                -1.14530207e-03 - 4.40333516e-03j,
                -6.34136961e-04 - 2.56723033e-03j,
                2.66908425e-19 + 1.13884445e-18j,
                6.34136961e-04 + 2.56723033e-03j,
                1.14530207e-03 + 4.40333516e-03j,
                5.55035762e-04 + 4.03884715e-03j,
                -2.83860481e-03 + 1.40160329e-04j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
            ],
            [
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                2.49752205e-03 + 7.97549070e-04j,
                8.27948153e-04 - 7.20348576e-05j,
                -3.32212314e-19 + 1.28606411e-19j,
                -8.27948153e-04 + 7.20348576e-05j,
                -2.49752205e-03 - 7.97549070e-04j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
                0.00000000e00 + 0.00000000e00j,
            ],
        ]
    )


@pytest.fixture
def eidos_data_yy():
    return np.array(
        [
            [
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                -0.02773569 + 4.61166965e-03j,
                -0.02896486 - 1.71081542e-03j,
                -0.02820611 - 2.67007775e-03j,
                -0.02972357 - 2.01417028e-03j,
                -0.03060841 + 3.46308326e-03j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
            ],
            [
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                -0.02521604 + 1.07379979e-03j,
                -0.01608335 - 5.28883615e-04j,
                -0.01245676 + 1.13729228e-03j,
                -0.012125 + 2.29457258e-03j,
                -0.01174045 + 2.88698477e-03j,
                -0.01081274 + 2.81924739e-03j,
                -0.01030938 + 1.99587322e-03j,
                -0.01464053 + 4.79941238e-05j,
                -0.02825091 - 1.39619664e-04j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
            ],
            [
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                -0.0070217 + 2.48121842e-02j,
                -0.01749212 + 6.90314413e-04j,
                -0.01266861 - 1.16186264e-03j,
                -0.01691482 - 1.55101508e-03j,
                -0.01533663 + 1.36250032e-03j,
                -0.00945929 + 5.18238878e-03j,
                -0.00613894 + 6.94740025e-03j,
                -0.00812801 + 5.71467092e-03j,
                -0.01239012 + 2.54058953e-03j,
                -0.01237066 + 2.65859033e-04j,
                -0.00801672 + 6.98083164e-04j,
                -0.01745111 + 7.06711643e-04j,
                -0.02188059 + 1.88712176e-02j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
            ],
            [
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                -0.01510183 + 2.05520538e-03j,
                -0.01443594 - 4.84074607e-03j,
                -0.02068452 - 5.60035766e-03j,
                -0.00564054 + 4.93953653e-03j,
                0.02012797 + 1.74781096e-02j,
                0.03862003 + 2.43142901e-02j,
                0.04450001 + 2.59504306e-02j,
                0.03867946 + 2.43380508e-02j,
                0.02117474 + 1.78966352e-02j,
                -0.00233604 + 6.26076320e-03j,
                -0.01468027 - 3.19970556e-03j,
                -0.00787465 - 2.21737203e-03j,
                -0.01505262 + 2.07488205e-03j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
            ],
            [
                0.0 + 0.00000000e00j,
                -0.01767809 + 6.99252637e-03j,
                -0.01241805 - 5.85125205e-03j,
                -0.02400167 - 1.08819467e-02j,
                -0.00133762 + 6.12803683e-03j,
                0.04160948 + 2.70601792e-02j,
                0.0648333 + 3.03234509e-02j,
                0.06155612 + 1.85955190e-02j,
                0.05483913 + 1.08771232e-02j,
                0.05969678 + 1.78521088e-02j,
                0.06256943 + 2.94182994e-02j,
                0.04126526 + 2.69225522e-02j,
                0.00226049 + 7.56665305e-03j,
                -0.01649636 - 7.88113155e-03j,
                -0.00544021 - 3.06133335e-03j,
                -0.02298911 + 4.86904233e-03j,
                0.0 + 0.00000000e00j,
            ],
            [
                0.0 + 0.00000000e00j,
                -0.00912497 + 3.59046284e-04j,
                -0.023367 - 1.56449933e-02j,
                -0.01403719 - 3.06768175e-03j,
                0.03906802 + 2.83091750e-02j,
                0.06720729 + 3.16431829e-02j,
                0.02609488 - 1.23349670e-02j,
                -0.04992965 - 7.34371720e-02j,
                -0.08866124 - 1.02184669e-01j,
                -0.05382777 - 7.49957379e-02j,
                0.02016326 - 1.47065796e-02j,
                0.06238596 + 2.97154903e-02j,
                0.03860907 + 2.81256723e-02j,
                -0.00852968 - 8.65637292e-04j,
                -0.01427867 - 1.20112451e-02j,
                -0.00575838 + 1.70509434e-03j,
                0.0 + 0.00000000e00j,
            ],
            [
                -0.01509845 + 1.35713670e-02j,
                -0.01070455 - 8.47650182e-03j,
                -0.02864361 - 1.87161114e-02j,
                0.00931525 + 1.32434039e-02j,
                0.06519588 + 4.06924695e-02j,
                0.03220498 - 2.30345593e-03j,
                -0.11723294 - 1.20785995e-01j,
                -0.29519943 - 2.49588157e-01j,
                -0.3758397 - 3.05663635e-01j,
                -0.30084653 - 2.51846013e-01j,
                -0.12636981 - 1.24439152e-01j,
                0.02330756 - 5.86087475e-03j,
                0.06066815 + 3.88821665e-02j,
                0.01193218 + 1.42897179e-02j,
                -0.01980409 - 1.51818438e-02j,
                -0.00318869 - 5.47146852e-03j,
                -0.02658932 + 8.97702144e-03j,
            ],
            [
                -0.0130474 + 1.03203164e-02j,
                -0.01557349 - 1.61303513e-02j,
                -0.02739254 - 1.69481097e-02j,
                0.02810887 + 2.69569762e-02j,
                0.06832847 + 4.00579397e-02j,
                -0.03231383 - 5.27869250e-02j,
                -0.28093131 - 2.40836572e-01j,
                -0.54617049 - 4.29520923e-01j,
                -0.66170145 - 5.09223433e-01j,
                -0.55298407 - 4.32245167e-01j,
                -0.29222551 - 2.45352284e-01j,
                -0.04400817 - 5.74626228e-02j,
                0.06089113 + 3.70842989e-02j,
                0.02840601 + 2.70757800e-02j,
                -0.01940483 - 1.37544168e-02j,
                -0.00638768 - 1.24576276e-02j,
                -0.01911714 + 7.89347753e-03j,
            ],
            [
                -0.01118647 + 9.47127324e-03j,
                -0.0177522 - 2.04199721e-02j,
                -0.0264801 - 1.64282957e-02j,
                0.03411185 + 3.31210240e-02j,
                0.06583748 + 3.97144748e-02j,
                -0.06212695 - 7.55079861e-02j,
                -0.34857632 - 2.94552702e-01j,
                -0.64635166 - 5.09796877e-01j,
                -0.77476652 - 5.99944164e-01j,
                -0.65357358 - 5.12684386e-01j,
                -0.36062865 - 2.99371537e-01j,
                -0.07481213 - 8.05798508e-02j,
                0.05735221 + 3.63218451e-02j,
                0.03353815 + 3.28916457e-02j,
                -0.01889341 - 1.33949421e-02j,
                -0.00814634 - 1.65793018e-02j,
                -0.01560529 + 7.70451735e-03j,
            ],
            [
                -0.01140681 + 1.11668462e-02j,
                -0.01571235 - 2.08093126e-02j,
                -0.0290053 - 2.07237826e-02j,
                0.02607466 + 3.00038764e-02j,
                0.06842071 + 4.79727808e-02j,
                -0.02743893 - 4.65244472e-02j,
                -0.27009462 - 2.41591339e-01j,
                -0.53044579 - 4.37836663e-01j,
                -0.6440967 - 5.20731556e-01j,
                -0.53725937 - 4.40560908e-01j,
                -0.28138882 - 2.46107051e-01j,
                -0.03913327 - 5.12001449e-02j,
                0.06098337 + 4.49991400e-02j,
                0.0263718 + 3.01226801e-02j,
                -0.02101759 - 1.75300897e-02j,
                -0.00652654 - 1.71365890e-02j,
                -0.01747655 + 8.74000725e-03j,
            ],
            [
                -0.01061734 + 1.55974112e-02j,
                -0.01035648 - 1.58842005e-02j,
                -0.03120563 - 2.83426219e-02j,
                0.00506473 + 1.50836734e-02j,
                0.06317782 + 5.43025245e-02j,
                0.03698239 + 1.34749627e-02j,
                -0.10330731 - 1.12416230e-01j,
                -0.27352604 - 2.51097691e-01j,
                -0.35114724 - 3.11581138e-01j,
                -0.27917314 - 2.53355547e-01j,
                -0.11244418 - 1.16069387e-01j,
                0.02808497 + 9.91754390e-03j,
                0.05865009 + 5.24922215e-02j,
                0.00768165 + 1.61299874e-02j,
                -0.02236611 - 2.48083543e-02j,
                -0.00284063 - 1.28791672e-02j,
                -0.02210821 + 1.10030657e-02j,
            ],
            [
                0.0 + 0.00000000e00j,
                -0.00703829 - 4.44362148e-03j,
                -0.02545409 - 3.19189081e-02j,
                -0.01971525 - 9.93450179e-03j,
                0.03321532 + 3.95531034e-02j,
                0.06622445 + 5.45371373e-02j,
                0.03326099 + 1.13326609e-02j,
                -0.03530496 - 5.46497385e-02j,
                -0.07104319 - 8.61833398e-02j,
                -0.03920307 - 5.62083044e-02j,
                0.02732937 + 8.96104841e-03j,
                0.06140311 + 5.26094447e-02j,
                0.03275637 + 3.93696008e-02j,
                -0.01420774 - 7.73245734e-03j,
                -0.01636577 - 2.82851599e-02j,
                -0.00367171 - 3.09757342e-03j,
                0.0 + 0.00000000e00j,
            ],
            [
                0.0 + 0.00000000e00j,
                -0.01111574 + 1.03786452e-02j,
                -0.01213413 - 2.21064582e-02j,
                -0.02866068 - 3.10793072e-02j,
                -0.00945819 + 1.76094361e-03j,
                0.03380588 + 4.20520838e-02j,
                0.06079717 + 5.75435608e-02j,
                0.06192507 + 5.02548835e-02j,
                0.05711849 + 4.32593664e-02j,
                0.06006573 + 4.95114733e-02j,
                0.05853331 + 5.66384093e-02j,
                0.03346166 + 4.19144568e-02j,
                -0.00586009 + 3.19955982e-03j,
                -0.02115538 - 2.80784920e-02j,
                -0.0051563 - 1.93165395e-02j,
                -0.01642677 + 8.25516120e-03j,
                0.0 + 0.00000000e00j,
            ],
            [
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                -0.00986135 + 3.31816801e-04j,
                -0.01513025 - 2.82355528e-02j,
                -0.02650829 - 3.08470583e-02j,
                -0.01510398 - 6.50516355e-03j,
                0.00950166 + 2.20787834e-02j,
                0.02844901 + 3.95487907e-02j,
                0.03474551 + 4.46903114e-02j,
                0.02850844 + 3.95725515e-02j,
                0.01054843 + 2.24973090e-02j,
                -0.01179947 - 5.18393688e-03j,
                -0.02050404 - 2.84464062e-02j,
                -0.00856896 - 2.56121788e-02j,
                -0.00981214 + 3.51493478e-04j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
            ],
            [
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                0.01315989 + 2.26621535e-02j,
                -0.01120354 - 1.37775188e-03j,
                -0.01224273 - 2.55446719e-02j,
                -0.02108901 - 3.40988446e-02j,
                -0.0230227 - 2.75170311e-02j,
                -0.01913585 - 1.74716484e-02j,
                -0.01642114 - 1.30105072e-02j,
                -0.01780456 - 1.69393662e-02j,
                -0.0200762 - 2.63389419e-02j,
                -0.01654485 - 3.22819705e-02j,
                -0.00759084 - 2.36847261e-02j,
                -0.01116253 - 1.36135465e-03j,
                -0.001699 + 1.67211870e-02j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
            ],
            [
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                -0.01373193 + 6.99950781e-03j,
                -0.01121444 - 1.17351084e-02j,
                -0.01123853 - 2.47896530e-02j,
                -0.01309703 - 3.04581569e-02j,
                -0.01346485 - 3.15908126e-02j,
                -0.01178477 - 2.99334821e-02j,
                -0.00909115 - 2.39310720e-02j,
                -0.00977161 - 1.11582307e-02j,
                -0.0167668 + 5.78608836e-03j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
            ],
            [
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                -0.00981126 + 1.27158466e-02j,
                -0.01584017 + 5.06142233e-03j,
                -0.01641213 + 2.37201721e-03j,
                -0.01659889 + 4.75806746e-03j,
                -0.01268397 + 1.15672602e-02j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
                0.0 + 0.00000000e00j,
            ],
        ]
    )
