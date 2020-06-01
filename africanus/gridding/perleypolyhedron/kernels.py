import numpy as np
from scipy.special import jn

def uspace(W, oversample):
    """
    Generates a kernel sampling of the form
    0   1   2   3   4
    |...|...|...|...|...
    |<----->|
    half-support (W / 2)
    | x W
    . x (oversample - 1) x W
    where W is odd
    """
    assert W % 2 == 1 # must be odd so that the taps can be centred at the origin
    taps = np.arange(oversample*W) / float(oversample) - W//2 # (|+.) * W centred at 0
    return taps

def sinc(W, oversample=5, a = 1.0):
    """
    Basic oversampled sinc window
    """
    u = uspace(W, oversample)
    res = np.sinc(u * a)
    return res / np.sum(res)

def kbsinc(W, b=None, oversample=5, order=15):
    """
    Modified keiser bessel windowed sinc (Jackson et al., IEEE transactions on medical imaging, 1991)
    with a modification of higher order bessels as default, as this improves the kernel at low
    number of taps.
    """
    autocoeffs = np.polyfit([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                            [1.9980,2.3934,3.3800,4.2054,4.9107,5.7567,6.6291,7.4302],
                            1)
    if b is None:
        b = np.poly1d(autocoeffs)(W)
    
    u = uspace(W, oversample)
    wnd =  jn(order, b * np.sqrt(1 - (2*u/(W+1))**2)) * 1/(W+1)
    res = sinc(W,oversample=oversample) * wnd * np.sum(wnd)
    return res / np.sum(res)

def hanningsinc(W, a=0.5, oversample=5):
    """
    Basic hanning windowed sinc
    """
    autocoeffs = np.polyfit([1.5, 2.0, 2.5, 3.0, 3.5],
                            [0.7600,0.7146,0.6185,0.5534,0.5185],
                            3)
    if a is None:
        a = np.poly1d(autocoeffs)(W)
    u = uspace(W, oversample)
    wnd = a + (1 - a) * np.cos(2*np.pi/(W+1)*u)
    res = sinc(W,oversample=oversample) * wnd 
    return res / np.sum(res)

def pack_kernel(K, W, oversample=5):
    """
    Repacks kernel to be cache coherent
    Expects sampled kernel of the form
    |...|...|...|...|...
    |<----->|
    half-support (W / 2)
    | x W
    . x (oversample - 1) x W
    """
    pkern = np.empty(oversample*W, dtype=K.dtype)
    for t in range(oversample):
        pkern[t*W:(t+1)*W] = K[t::oversample]
    return pkern

def unpack_kernel(K, W, oversample=5):
    """
    Unpacks kernel to original sampling form (non-cache coherent)
    Produces sampled kernel of the form
    |...|...|...|...|...
    |<----->|
    half-support (W / 2)
    | x W
    . x (oversample - 1) x W
    """
    upkern = np.empty(oversample*W, dtype=K.dtype)
    for t in range(oversample):
        upkern[t::oversample] = K[t*W:(t+1)*W]
    return upkern

