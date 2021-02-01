import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest


from africanus.rime.fast_beam_cubes import beam_cube_dde, freq_grid_interp


def rf(*a, **kw):
    return np.random.random(*a, **kw)


def rc(*a, **kw):
    return rf(*a, **kw) + 1j*rf(*a, **kw)


@pytest.fixture
def beam_freq_map():
    return np.array([.5, .56,  .7, .91, 1.0])


@pytest.fixture
def beam_freq_map_montblanc():
    """ Montblanc doesn't handle values outside the cube in the same way """
    return np.array([.4, .56,  .7, .91, 1.1])


@pytest.fixture
def freqs():
    """
    Related to the beam_freq_map fixture.
    Explanation of frequency test values:

    1. One value (0.4) below the beam freq range
    2. One value (0.5) on the first beam freq
    3. One value (1.0) on the last beam freq
    4. One value (1.1) above the beam freq range
    """

    return np.array([.4, .5, .6, .7, .8, .9, 1.0, 1.1])


def test_fast_beam_small():
    """ Small beam test, interpolation of one soure at [0.1, 0.1] """
    np.random.seed(42)

    # One frequency, to the lower side of the beam frequency map
    freq = np.asarray([.3])
    beam_freq_map = np.asarray([0.0, 1.0])

    beam_lw = 2
    beam_mh = 2
    beam_nud = beam_freq_map.shape[0]

    time = 1
    ants = 1
    chans = freq.shape[0]

    beam_extents = np.asarray([[-1.0, 1.0], [-1.0, 1.0]])
    lm = np.asarray([[0.1, 0.1]])

    # Set all of the following to identity
    parangles = np.zeros((time, ants))
    point_errors = np.zeros((time, ants, chans, 2))
    antenna_scaling = np.ones((ants, chans, 2))

    beam = rc((beam_lw, beam_mh, beam_nud, 1))

    ddes = beam_cube_dde(beam, beam_extents, beam_freq_map,
                         lm, parangles, point_errors, antenna_scaling,
                         freq)

    # Pen and paper the expected value
    lower_l = beam_extents[0, 0]
    lower_m = beam_extents[1, 0]

    ld = (lm[0, 0] - lower_l) / (beam_extents[0, 1] - lower_l)
    md = (lm[0, 1] - lower_m) / (beam_extents[1, 1] - lower_m)
    chd = freq[0]  # Hard-coded

    corr_sum = np.zeros((1,), dtype=beam.dtype)
    abs_sum = np.zeros((1,), dtype=beam.real.dtype)

    # Weights of the sample at each grid point
    wt0 = (1 - ld)*(1 - md)*(1 - chd)
    wt1 = ld*(1 - md)*(1 - chd)
    wt2 = (1 - ld)*md*(1 - chd)
    wt3 = ld*md*(1 - chd)

    wt4 = (1 - ld)*(1 - md)*chd
    wt5 = ld*(1 - md)*chd
    wt6 = (1 - ld)*md*chd
    wt7 = ld*md*chd

    # Sum lower channel correlations
    corr_sum[:] += wt0 * beam[0, 0, 0, 0]
    corr_sum[:] += wt1 * beam[1, 0, 0, 0]
    corr_sum[:] += wt2 * beam[0, 1, 0, 0]
    corr_sum[:] += wt3 * beam[1, 1, 0, 0]

    abs_sum[:] += wt0 * np.abs(beam[0, 0, 0, 0])
    abs_sum[:] += wt1 * np.abs(beam[1, 0, 0, 0])
    abs_sum[:] += wt2 * np.abs(beam[0, 1, 0, 0])
    abs_sum[:] += wt3 * np.abs(beam[1, 1, 0, 0])

    # Sum upper channel correlations
    corr_sum[:] += wt4 * beam[0, 0, 1, 0]
    corr_sum[:] += wt5 * beam[1, 0, 1, 0]
    corr_sum[:] += wt6 * beam[0, 1, 1, 0]
    corr_sum[:] += wt7 * beam[1, 1, 1, 0]

    abs_sum[:] += wt4 * np.abs(beam[0, 0, 1, 0])
    abs_sum[:] += wt5 * np.abs(beam[1, 0, 1, 0])
    abs_sum[:] += wt6 * np.abs(beam[0, 1, 1, 0])
    abs_sum[:] += wt7 * np.abs(beam[1, 1, 1, 0])

    corr_sum *= abs_sum / np.abs(corr_sum)

    assert_array_almost_equal([[[[[0.470255+0.4786j]]]]], ddes)
    assert_array_almost_equal(ddes.squeeze(), corr_sum.squeeze())


def test_grid_interpolate(freqs, beam_freq_map):
    freq_data = freq_grid_interp(freqs, beam_freq_map)

    freq_scale = freq_data[:, 0]
    fgrid_diff = freq_data[:, 1]
    grid_pos = np.int32(freq_data[:, 2])

    # Frequencies (first -- 0.8 and last -- 1.1)
    # outside the beam result in scaling,
    assert_array_almost_equal(freq_scale,
                              [0.8, 1.,  1.,  1.,  1.,  1.,  1.,  1.1])
    # Frequencies outside the beam are snapped to 0 if below
    # and beam_nud - 2 if above.
    # Frequencies on the edges are similarly snapped
    assert_array_equal(grid_pos, [0, 0, 1, 2, 2, 2, 3, 3])
    # Frequency less grid position frequency.
    # frequency is snapped to the first and last beam freq value
    # if outside (first and last values)
    # Third frequency value is also exactly on a grid point
    exp_diff = [1.,
                1.,
                0.71428571,
                1.,
                0.52380952,
                0.04761905,
                0.,
                0.]

    assert_array_almost_equal(fgrid_diff, exp_diff)


def test_dask_fast_beams(freqs, beam_freq_map):
    da = pytest.importorskip("dask.array")
    from africanus.rime.dask import beam_cube_dde as dask_beam_cube_dde

    beam_lw = 10
    beam_mh = 10
    beam_nud = beam_freq_map.shape[0]

    src_c = (2, 3, 5)
    time_c = (3, 2)
    ants_c = (2, 1, 1)
    chan_c = da.core.normalize_chunks(3, shape=freqs.shape)[0]

    src = sum(src_c)
    time = sum(time_c)
    ants = sum(ants_c)
    chans = sum(chan_c)

    # Source Transform variables
    lm = np.random.random(size=(src, 2))
    parangles = np.random.random(size=(time, ants))
    point_errors = np.random.random(size=(time, ants, chans, 2))
    antenna_scaling = np.random.random(size=(ants, chans, 2))

    # Make random values more representative
    lm = (lm - 0.5)*0.0001        # Shift lm to around the centre
    parangles *= np.pi / 12        # parangles to 15 degrees max
    point_errors *= 0.001         # Pointing errors
    antenna_scaling *= 0.0001     # Antenna scaling

    # Beam variables
    beam = rc((beam_lw, beam_mh, beam_nud, 2, 2))
    beam_lm_extents = np.asarray([[-1.0, 1.0], [-1.0, 1.0]])

    # Compute numba ddes
    ddes = beam_cube_dde(beam, beam_lm_extents, beam_freq_map,
                         lm, parangles, point_errors, antenna_scaling,
                         freqs)

    # Create dask arrays
    da_beam = da.from_array(beam, chunks=beam.shape)
    da_beam_freq_map = da.from_array(beam_freq_map, chunks=beam_freq_map.shape)
    da_lm = da.from_array(lm, chunks=(src_c, 2))
    da_parangles = da.from_array(parangles, chunks=(time_c, ants_c))
    da_point_errors = da.from_array(point_errors,
                                    chunks=(time_c, ants_c, chan_c, 2))
    da_ant_scale = da.from_array(antenna_scaling, chunks=(ants_c, chan_c, 2))
    da_extents = da.from_array(beam_lm_extents, chunks=beam_lm_extents.shape)
    da_freqs = da.from_array(freqs, chunks=chan_c)

    # dask ddes
    da_ddes = dask_beam_cube_dde(da_beam, da_extents, da_beam_freq_map,
                                 da_lm, da_parangles, da_point_errors,
                                 da_ant_scale, da_freqs)

    # Should be strictly equal
    assert_array_equal(da_ddes.compute(), ddes)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fast_beams_vs_montblanc(freqs, beam_freq_map_montblanc, dtype):
    """ Test that the numba beam matches montblanc implementation """
    mb_tf_mod = pytest.importorskip("montblanc.impl.rime.tensorflow")
    tf = pytest.importorskip("tensorflow")

    freqs = freqs.astype(dtype)
    beam_freq_map = beam_freq_map_montblanc.astype(dtype)

    beam_lw = 10
    beam_mh = 10
    beam_nud = beam_freq_map.shape[0]

    src = 10
    time = 5
    ants = 4
    chans = freqs.shape[0]

    ctype = np.result_type(dtype, np.complex64)

    # Source Transform variables
    lm = np.random.random(size=(src, 2)).astype(dtype)
    parangles = np.random.random(size=(time, ants)).astype(dtype)
    point_errors = np.random.random(size=(time, ants, chans, 2)).astype(dtype)
    antenna_scaling = np.random.random(size=(ants, chans, 2)).astype(dtype)

    # Make random values more representative
    lm = (lm - 0.5)*0.0001        # Shift lm to around the centre
    parangles *= np.pi / 12        # parangles to 15 degrees max
    point_errors *= 0.001         # Pointing errors
    antenna_scaling *= 0.0001     # Antenna scaling

    # Beam variables
    beam = rc((beam_lw, beam_mh, beam_nud, 2, 2)).astype(ctype)
    beam_lm_extents = np.asarray([[-1.0, 1.0], [-1.0, 1.0]]).astype(dtype)

    ddes = beam_cube_dde(beam, beam_lm_extents, beam_freq_map,
                         lm, parangles, point_errors, antenna_scaling,
                         freqs)

    assert ddes.shape == (src, time, ants, chans, 2, 2)

    rime = mb_tf_mod.load_tf_lib()

    # Montblanc beam extent format is different
    mb_beam_extents = np.array([beam_lm_extents[0, 0],
                                beam_lm_extents[1, 0],
                                beam_freq_map[0],
                                beam_lm_extents[0, 1],
                                beam_lm_extents[1, 1],
                                beam_freq_map[-1]], dtype=dtype)

    # Montblanc wants flattened correlations
    mb_beam = beam.reshape(beam.shape[:3] + (-1,))

    np_args = [lm, freqs, point_errors, antenna_scaling,
               np.sin(parangles), np.cos(parangles),
               mb_beam_extents, beam_freq_map, mb_beam]
    # Argument string name list
    arg_names = ["lm", "frequency", "point_errors", "antenna_scaling",
                 "parallactic_angle_sin", "parallactic_angle_cos",
                 "beam_extents", "beam_freq_map", "e_beam"]

    # Constructor tensorflow variables
    tf_args = [tf.Variable(v, name=n) for v, n in zip(np_args, arg_names)]

    with tf.device("/cpu:0"):
        tf_dde = rime.e_beam(*tf_args)

    init_op = tf.global_variables_initializer()

    with tf.Session() as S:
        S.run(init_op)
        res = S.run(tf_dde)

    assert_array_almost_equal(res, ddes.reshape(ddes.shape[:4] + (-1,)))
