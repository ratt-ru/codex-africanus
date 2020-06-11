"""
Example of GPR interpolation on incomplete Euclidean grid
"""

import numpy as np
np.random.seed(420)
from africanus.gps.kernels import exponential_squared as expsq
from africanus.linalg import kronecker_tools as kt
from africanus.linalg.pcg_masked import pcg
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

if __name__=="__main__":
    # domain
    Nt = 100
    t = np.linspace(0,1,Nt)
    Nv = 90
    Ntot = Nt*Nv
    v = np.linspace(0,1,Nv)
    Nl = 2  # translates to Nl**2 directions
    Ns = Nl**2
    l = np.random.randn(Nl)
    m = np.random.randn(Nl)
    ll, mm = np.meshgrid(l, m)
    lm = np.vstack((ll.flatten(), mm.flatten())).T

    # covariances
    Kt = expsq(t, t, 1.0, 0.1)
    Kv = expsq(v, v, 1.0, 0.25)
    Ks = expsq(lm, lm, 1.0, 1.0)

    K = (Kt, Kv, Ks)
    
    
    # sample
    L = kt.kron_cholesky(K)
    u = np.random.randn(Nt*Nv*Ns)

    f = kt.kron_matvec(L, u).reshape(Nt, Nv, Ns)

    # for s in range(Ns):
    #     plt.imshow(f[:, :, s])
    #     plt.colorbar()
    #     plt.show()

    # add some diagonal noise
    sigman = 0.1
    Sigma = sigman**2 * np.exp(np.random.randn(Nt, Nv, Ns))  # variance
    y = f + np.random.randn(Nt, Nv, Ns)*np.sqrt(Sigma)
    
    # # assume some t/nu indices are flagged
    Nflag = int(0.25*Ntot)
    It = np.random.randint(0, Nt, Nflag)
    Iv = np.random.randint(0, Nv, Nflag)
    y[It, Iv, :] = np.nan
    y[:, 45:60, :] = np.nan
    mask = np.isnan(y.flatten())

    # # plot data
    # for s in range(Ns):
    #     plt.imshow(y[:, :, s])
    #     plt.colorbar()
    #     plt.show()

    # recover signal
    data = np.where(mask, 0.0, y.flatten())
    Sigma = Sigma.flatten()
    mask_ind = np.argwhere(np.isnan(data)).squeeze()
    unmask_ind = np.argwhere(~np.isnan(data)).squeeze()
    # print(data[mask_ind])
    # print(data[unmask_ind])

    # data covariance operator
    Ky = lambda x: np.where(mask, 0.0, kt.kron_matvec(K, x) + Sigma * x)
    M = lambda x: np.where(mask, 0.0, kt.kron_matvec(K, x))

    # mean func
    mbar = np.mean(data[unmask_ind])
    
    tmp = pcg(Ky, data-mbar, np.zeros(Nt*Nv*Ns, dtype=np.float64), M=M)
    fp = (mbar + kt.kron_matvec(K, tmp)).reshape(Nt, Nv, Ns)
    for s in range(Ns):
        plt.figure('f')
        plt.imshow(f[:, :, s])
        plt.colorbar()
        plt.figure('y')
        plt.imshow(y[:, :, s])
        plt.colorbar()
        plt.figure('fp')
        plt.imshow(fp[:, :, s])
        plt.colorbar()
        plt.show()





