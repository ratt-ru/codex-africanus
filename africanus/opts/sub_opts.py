import numpy as np
import dask.array as da


def proj_l2ball(x, eps, y):
    # projection of x onto the l2 ball centered in y with radius eps
    p = x-y
    p = p * np.minimum(eps/np.linalg.norm(p), np.ones_like(y))
    p = p+y
    return p


def da_proj_l2ball(x, eps, y):
    # projection of x onto the l2 ball centered in y with radius eps
    p = x-y
    p = p * da.minimum(eps/da.linalg.norm(p), da.ones_like(y))
    p = p+y
    return p


def proj_l1_plus_pos(x, tau):
    return np.where(x < tau, 0.0, x-tau)


def da_proj_l1_plus_pos(x, tau):
    return da.where(x < tau, 0.0, x-tau)


def power_dask(L, LT, im_size, tol=1e-2, max_iter=100):
    np.random.seed(123)
    x = da.random.random((im_size[0], im_size[1]), chunks=([im_size[0], im_size[1]]))
    x /= da.linalg.norm(x, 'fro').compute()
    x_chunks = x.chunks
    init_val = 1

    for i in range(max_iter):
        y = L(x)
        # x = LT(y)
        x = da.from_array(LT(y), chunks=x_chunks)
        val = da.linalg.norm(x, 'fro').compute()
        rel_var = np.abs(val - init_val) / init_val
        if rel_var < tol:
            break
        print("Iter {0}: {1}".format(i, rel_var))
        init_val = val
        x /= val
    print('Spectral norm=', np.sqrt(val))
    return np.sqrt(val)


def pow_method(L, LT, im_size, tol=1e-6, max_iter=100):
    """
    @author: mjiang
    ming.jiang@epfl.ch
    Computes the spectral radius (maximum eigen value) of the operator A

    @param A: function handle of direct operator

    @param At: function handle of adjoint operator

    @param im_size: size of the image

    @param tol: tolerance of the error, stopping criterion

    @param max_iter: max iteration

    @return: spectral radius of the operator
    """
    np.random.seed(123)
    x = np.random.randn(im_size[0], im_size[1])
    x /= np.linalg.norm(x, 'fro')
    init_val = 1

    for i in range(max_iter):
        y = L(x)
        x = LT(y)
        val = np.linalg.norm(x, 'fro')
        rel_var = np.abs(val - init_val) / init_val
        if rel_var < tol:
            break
        print("Iter {0}: {1}".format(i, rel_var))
        init_val = val
        x /= val
    print('Spectral norm=', np.sqrt(val))
    return np.sqrt(val)


def da_get_diff(x_new, x, n):

    # Get new norms
    norm2 = da.linalg.norm(x_new).compute()
    norm1 = da.linalg.norm(x_new, 1).compute()

    if norm1 == 0:
        norm1 = 1
    if norm2 == 0:
        norm2 = 1

    # get diff i.t.o. 1-norm
    diff1 = da.linalg.norm(x_new - x, 1).compute() / norm1
    # get diff i.t.o. 2-norm
    diff2 = da.linalg.norm(x_new - x).compute() / norm2

    print('L1 norm=', norm1, ' L2 norm=', norm2)
    print('Iter = %i, diff1 = %f, diff2 = %f' % (n, diff1, diff2))

    return da.maximum(diff1, diff2)


def get_diff(x_new, x, n):

    # Get new norms
    norm2 = np.linalg.norm(x_new)
    norm1 = np.linalg.norm(x_new, 1)

    if norm1 == 0:
        norm1 = 1
    if norm2 == 0:
        norm2 = 1

    # get diff i.t.o. 1-norm
    diff1 = np.linalg.norm(x_new - x, 1) / norm1
    # get diff i.t.o. 2-norm
    diff2 = np.linalg.norm(x_new - x) / norm2

    print('L1 norm=', norm1, ' L2 norm=', norm2)
    print('Iter = %i, diff1 = %f, diff2 = %f' % (n, diff1, diff2))

    return np.maximum(diff1, diff2)


# def power_method(A, im_size, maxiter=100, tol=1e-6):
#     x = np.random.randn(im_size[0], im_size[1])
#     x /= np.linalg.norm(x)
#     k = 0
#     eps = 1.0
#     lam = 1.0
#     while k < maxiter and eps > tol:
#         y = A(x)
#         x = (y / np.sqrt(y.T.dot(y)))
#         print(x.shape)
#         lamp = x.T.dot(A(x))
#         print(lamp)
#         eps = np.abs(lam - lamp)
#         lam = lamp
#         k += 1
#
#     print("Iters = ", k)
#     return lam

# def radec_to_lm(ra0, dec0, ra, dec):
#     delta_ra = ra - ra0
#     l = (np.cos(dec) * np.sin(delta_ra))
#     m = (np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(delta_ra))
#     return l, m
#
#
# data_path = "/home/antonio/Documents/Masters/Helpful_Stuff/WSCMSSSMFTestSuite_(copy)/SSMF.MS_p0"
#
# nrow = 500
# nchan = 1
#
# for ds in xarrayms.xds_from_ms(data_path):
#     Vdat = ds.DATA.data.compute()
#     uvw = -ds.UVW.data.compute()[0:nrow, :]
#     weights = ds.WEIGHT.data.compute()[0:nrow, 0:nchan]
#
# vis = Vdat[0:nrow, 0:nchan, 0]
#
# datas = fits.open('/home/antonio/Documents/Masters/Helpful_Stuff/WSCMSSSMFTestSuite_(copy)/output.fits')[0]
#
# hd = datas.header
# print(hd)
#
# # setup LM
# ra_cen = 0
# ra = 3.15126500e-05
# dec_cen = 0
# dec = -0.00551471375
# l, m = radec_to_lm(ra_cen, dec_cen, ra, dec)
#
# # l = np.linspace(-l_change, l_change, hd['NAXIS1'])
# # m = np.linspace(-m_change, m_change, hd['NAXIS2'])
# # ll, mm = np.meshgrid(l, m)
# lm = np.array([l, m]).reshape([1,2])# np.vstack((ll.flatten(), mm.flatten())).T
#
# # get frequency
# msfreq = tbl("/home/antonio/Documents/Masters/Helpful_Stuff/WSCMSSSMFTestSuite_(copy)/SSMF.MS_p0::SPECTRAL_WINDOW")
# freq = msfreq.getcol("CHAN_FREQ").squeeze()[0:nchan]
# print(freq)
# ref_freq = 1.53e9
# alpha = -.7
#
# # data = datas.data
# # data = data.reshape([data.shape[2]*data.shape[3],1]).astype(float)
# data = np.array([100]*(freq/ref_freq)**alpha).reshape([1, nchan])
#
# recovered = im_to_vis(data, uvw, lm, freq)
#
# # npix = hd['NAXIS2']
#
# print(recovered - vis)
#
# #print(vis)
#
# print(sum(abs(vis - recovered)))
#
# # plt.show()
