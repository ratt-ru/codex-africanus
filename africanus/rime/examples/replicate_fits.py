from astropy.io import fits
from africanus.rime import zernike_dde
import numpy as np
import dask
import dask.array as da
import time
import matplotlib.pyplot as plt


from util import *
from spectral import *
from spatial import *
from parallelize import *

def dct_recon_all(Co):
    F,C,I = Co['nu'], Co['dctc'], Co['dcti']
    recons = np.zeros((2,len(F),2,2,C.shape[-1]))
    for i in range(2):
        for j in range(2):
            for k in range(C.shape[-1]):
                for p in range(2):
                    recons[p,:,i,j,k] = dct_recon(C[p,:,i,j,k], list(map(int,I[p,:,i,j,k])), len(F))
    print(recons)
    return recons

def write_fits(beam, timestamp, filename):
    hdr = fits.Header()
    ctypes = ['px', 'py']
    beam = beam.compute()
    crvals = [0.0, 0.0]
    crpix = [beam.shape[0] // 2, beam.shape[1] // 2]
    cunits = ["deg", "deg"]
    for i in range(len(beam.shape)):
        ii = str(i + 1)
        hdr['CTYPE' + ii] = ctypes[i]
        hdr['CRPIX' + ii] = crpix[i]
        print(crvals[i])
        hdr['CRVAL' + ii] = crvals[i]
        hdr['CUNIT' + ii] = cunits[i]
    hdr['TELESCOP'] = 'MeerKAT'
    hdr['DATE'] = time.ctime()
    print("COMPUTING BEAM")
    hdu = fits.PrimaryHDU(beam.real, header=hdr)
    print("HDU IS ", hdu.header)
    print("BEAM DONE")
    hdu.writeto(filename, overwrite=True)

na = 1
nchan = 1
ntime = 1
npoly = 8

fits_file = fits.open("./primary_beam_mh_1070MHz_10deg_I_re.fits")

eidos_pb = fits_file[0]

cr_l = eidos_pb.header['CRVAL2']
cr_m = eidos_pb.header['CRVAL1']

del_l = eidos_pb.header['CDELT2'] * np.pi/180
del_m = eidos_pb.header['CDELT1'] * np.pi/180

npix = eidos_pb.header['NAXIS1']

l_min = cr_l - ((npix // 2) * del_l)
l_max = cr_l + ((npix // 2) * del_l)

m_min = cr_m - ((npix // 2) * del_m)
m_max = cr_m + ((npix // 2) * del_m)

l = np.arange(l_min, l_max, del_l)
m = np.arange(m_min, m_max, del_m)

ll, mm = np.meshgrid(l, m)

lm = np.vstack((ll.flatten(), mm.flatten()))

coords = np.empty((3, npix * npix, ntime, na, nchan), dtype=np.float64)
coeffs = np.zeros((na, nchan, 2, 2, npoly), dtype=np.float64)
noll_index = np.zeros((na, nchan, 2, 2, npoly), dtype=np.float64)
parallactic_angles = np.zeros((ntime, nchan,), dtype=np.float64)
antenna_scaling = np.ones((na, nchan, 2))
frequency_scaling = np.ones((nchan,), dtype=np.float64)
pointing_errors = np.zeros((ntime, na, nchan, 2), dtype=np.float64)


coords[:2, :, 0, 0, 0], coords[2, :, 0, 0, 0] = lm[:, :], 0

filename="./meerkat_beam_coeffs_ah_zp_dct.npy"
coeffs_file = np.load(filename, encoding='latin1', allow_pickle=True).item()
Cr = dct_recon_all(coeffs_file)
print(Cr)
corr_letters = [b'x', b'y']

for corr1 in range(2):
    for corr2 in range(2):
        corr_index = corr_letters[corr1] + corr_letters[corr2]
        noll_index[0,0,corr1, corr2, :] = coeffs_file[b'noll_index'][corr_index][:npoly]
        coeffs[0,0,corr1,corr2, :] = coeffs_file[b'coeff'][corr_index][:npoly]

z = zernike_dde(da.from_array(coords, chunks=coords.shape),
        da.from_array(coeffs, chunks=coeffs.shape),
        da.from_array(noll_index, chunks=noll_index.shape),
        da.from_array(parallactic_angles, chunks=parallactic_angles.shape),
        da.from_array(frequency_scaling, chunks=frequency_scaling.shape),
        da.from_array(antenna_scaling, chunks=antenna_scaling.shape),
        da.from_array(pointing_errors, chunks=pointing_errors.shape))[:, 0, 0, 0, 0, 1].reshape((npix, npix)).real
z = z - np.min(z)
z = z / np.max(np.abs(z))
        
print(z.shape, npix * npix)

write_fits(z, [0], "eidos_ca_primary_beam.fits")
write_fits(da.from_array(eidos_pb.data.real[0, 0, 0, :, :]), [0], "eidos_extracted_primary_beam.fits")