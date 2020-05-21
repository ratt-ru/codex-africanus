"""
Example of GPR interpolation on incomplete Euclidean grid
"""

import numpy as np
np.random.seed(420)
from africanus.gps.kernels import exponential_squared as expsq
from africanus.linalg import kronecker_tools as kt
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def pcg(A, b, x0, M=None, mask_ind=None, tol=1e-3, maxit=500, verbosity=2, report_freq=10):
    
    if M is None:
        M = lambda x: x
    
    b[mask_ind] = 0.0
    r = A(x0) - b
    r[mask_ind] = 0.0
    y = M(r)
    y[mask_ind] = 0.0
    p = -y
    rnorm = np.vdot(r, y)
    if np.isnan(rnorm) or rnorm == 0.0:
        eps0 = 1.0
    else:
        eps0 = rnorm
    k = 0
    x = x0
    while rnorm/eps0 > tol and k < maxit:
        xp = x.copy()
        rp = r.copy()
        Ap = A(p)
        Ap[mask_ind] = 0.0
        rnorm = np.vdot(r, y)
        alpha = rnorm/np.vdot(p, Ap)
        x = xp + alpha*p
        r = rp + alpha*Ap
        r[mask_ind] = 0.0
        y = M(r)
        y[mask_ind] = 0.0
        rnorm_next = np.vdot(r, y)
        # while rnorm_next > rnorm:  # TODO - better line search
        #     alpha *= 0.9
        #     x = xp + alpha*p
        #     r = rp + alpha*Ap
        #     y = M(r)
        #     rnorm_next = np.vdot(r, y)

        beta = rnorm_next/rnorm
        p = beta*p - y
        rnorm = rnorm_next
        k += 1

        if not k%report_freq and verbosity > 1:
            print("At iteration %i rnorm = %f"%(k, rnorm/eps0))

    if k >= maxit:
        if verbosity > 0:
            print("CG - Maximum iterations reached. Norm of residual = %f.  "%(rnorm/eps0))
    else:
        if verbosity > 0:
            print("CG - Success, converged after %i iterations"%k)
    return x


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

    K = np.array([Kt, Kv, Ks], dtype=object)
    
    
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
    Nflag = int(0.05*Ntot)
    It = np.random.randint(0, Nt, Nflag)
    Iv = np.random.randint(0, Nv, Nflag)
    y[It, Iv, :] = np.nan
    y[:, 45:60, :] = np.nan

    # # plot data
    # for s in range(Ns):
    #     plt.imshow(y[:, :, s])
    #     plt.colorbar()
    #     plt.show()

    # recover signal
    data = y.flatten()
    Sigma = Sigma.flatten()
    mask_ind = np.argwhere(np.isnan(data)).squeeze()
    unmask_ind = np.argwhere(~np.isnan(data)).squeeze()
    # print(data[mask_ind])
    # print(data[unmask_ind])

    # data covariance operator
    Ky = lambda x: kt.kron_matvec(K, x) + Sigma * x
    M = lambda x: kt.kron_matvec(K, x)

    # mean func
    mbar = np.mean(data[unmask_ind])
    tmp = pcg(Ky, data-mbar, np.zeros(Nt*Nv*Ns, dtype=np.float64), M=M, mask_ind=mask_ind)
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





