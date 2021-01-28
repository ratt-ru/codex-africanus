import numpy as np


def fac(x):
    if x < 0:
        raise ValueError("Factorial input is negative.")
    if x == 0:
        return 1
    factorial = 1
    for i in range(1, int(x + 1)):
        factorial *= i
    return factorial


def pre_fac(k, n, m):
    numerator = (-1.0) ** k * fac(n - k)
    denominator = fac(k) * fac((n + m) / 2.0 - k) * fac((n - m) / 2.0 - k)
    return numerator / denominator


def zernike_rad(m, n, rho):
    if n < 0 or m < 0 or abs(m) > n:
        raise ValueError("m and n values are incorrect.")
    radial_component = 0
    for k in range(int((n - m) / 2 + 1)):
        radial_component += pre_fac(k, n, m) * rho ** (n - 2.0 * k)
    # print(radial_component)
    return radial_component


def zernike(j, rho, phi):
    # print(rho)
    final_product = np.zeros(rho.shape)
    j += 1
    n = 0
    j1 = j - 1
    # print(j1, n)
    while j1 > n:
        n += 1
        j1 -= n
    m = (-1) ** j * ((n % 2) + 2 * int((j1 + ((n + 1) % 2)) / 2.0))
    # print(m)
    if m > 0:
        final_product = zernike_rad(m, n, rho) * np.cos(m * phi)
    elif m < 0:
        final_product = zernike_rad(-m, n, rho) * np.sin(-m * phi)
    else:
        final_product = zernike_rad(0, n, rho)

    final_product[np.where(rho > 1)] = 0
    return final_product


def _convert_coords(l, m):
    rho, phi = ((l ** 2 + m ** 2) ** 0.5), np.arctan2(l, m)
    return rho, phi


def nb_zernike_dde(
    coords,
    coeffs,
    noll_index,
    out,
    parallactic_angles,
    frequency_scaling,
    antenna_scaling,
    pointing_errors,
):
    sources, times, ants, chans, corrs = out.shape
    npoly = coeffs.shape[-1]

    for t in range(times):
        for a in range(ants):
            sin_pa = np.sin(parallactic_angles[t, a])
            cos_pa = np.cos(parallactic_angles[t, a])

            for c in range(chans):
                l, m, freq = (
                    coords[0, :, t, a, c],
                    coords[1, :, t, a, c],
                    coords[2, :, t, a, c],
                )

                l = l * frequency_scaling[c]
                m = m * frequency_scaling[c]

                l += pointing_errors[t, a, c, 0]
                m += pointing_errors[t, a, c, 1]

                vl = l * cos_pa - l * sin_pa
                vm = m * sin_pa + m * cos_pa

                vl *= antenna_scaling[a, c, 0]
                vm *= antenna_scaling[a, c, 1]

                rho, phi = _convert_coords(vl, vm)
                # print("rho, phi,l,m, sqrt(l**2 + m**2) is ", rho, phi,vl,vm,(l**2 + m**2)**0.5)

                for co in range(corrs):
                    zernike_sum = 0

                    for p in range(npoly):
                        zc = coeffs[a, c, co, p]
                        zn = noll_index[a, c, co, p]
                        zernike_sum += zc * zernike(zn, rho, phi)
                        # print(zn, rho, phi)

                    out[:, t, a, c, co] = zernike_sum

    return out


def zernike_dde(
    coords,
    coeffs,
    noll_index,
    parallactic_angles,
    frequency_scaling,
    antenna_scaling,
    pointing_errors,
):
    """ Wrapper for :func:`nb_zernike_dde` """
    _, sources, times, ants, chans = coords.shape
    # ant, chan, corr_1, ..., corr_n, poly
    corr_shape = coeffs.shape[2:-1]
    npoly = coeffs.shape[-1]

    # Flatten correlation dimensions for numba function
    fcorrs = np.product(corr_shape)
    ddes = np.empty((sources, times, ants, chans, fcorrs), coeffs.dtype)

    coeffs = coeffs.reshape((ants, chans, fcorrs, npoly))
    noll_index = noll_index.reshape((ants, chans, fcorrs, npoly))

    result = nb_zernike_dde(
        coords,
        coeffs,
        noll_index,
        ddes,
        parallactic_angles,
        frequency_scaling,
        antenna_scaling,
        pointing_errors,
    )

    # Reshape to full correlation size
    return result.reshape((sources, times, ants, chans) + corr_shape)
