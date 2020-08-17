from africanus.util.numba import jit, overload
from numpy import cos, sin


def uvw_norotate(uvw, ra0, dec0, ra, dec, policy_type):
    pass


def uvw_rotate(uvw, ra0, dec0, ra, dec, policy_type):
    '''
        Compute the following 3x3 coordinate transformation matrix:
        Z_rot(facet_new_rotation) * \\
        T(new_phase_centre_ra,new_phase_centre_dec) * \\
        transpose(T(old_phase_centre_ra,
                    old_phase_centre_dec)) * \\
        transpose(Z_rot(facet_old_rotation))
        where:
                            |      cRA             -sRA            0       |
        T (RA,D) =          |      -sDsRA          -sDcRA          cD      |
                            |      cDsRA           cDcRA           sD      |
        This is the similar to the one in Thompson, A. R.; Moran, J. M.;
        and Swenson, G. W., Jr. Interferometry and Synthesis
        in Radio Astronomy, New York: Wiley, ch. 4, but in a
        lefthanded system.
        We're not transforming between a coordinate system with w pointing
        towards the pole and one with w pointing towards the reference
        centre here, so the last rotation matrix is ignored!
        This transformation will let the image be tangent to the celestial
        sphere at the new delay centre
    '''
    d_ra = ra - ra0
    c_d_ra = cos(d_ra)
    s_d_ra = sin(d_ra)
    c_new_dec = cos(dec)
    c_old_dec = cos(dec0)
    s_new_dec = sin(dec)
    s_old_dec = sin(dec0)
    mat_11 = c_d_ra
    mat_12 = s_old_dec * s_d_ra
    mat_13 = -c_old_dec * s_d_ra
    mat_21 = -s_new_dec * s_d_ra
    mat_22 = s_new_dec * s_old_dec * c_d_ra + c_new_dec * c_old_dec
    mat_23 = -c_old_dec * s_new_dec * c_d_ra + c_new_dec * s_old_dec
    mat_31 = c_new_dec * s_d_ra
    mat_32 = -c_new_dec * s_old_dec * c_d_ra + s_new_dec * c_old_dec
    mat_33 = c_new_dec * c_old_dec * c_d_ra + s_new_dec * s_old_dec
    uvw[0] = mat_11 * uvw[0] + mat_12 * uvw[1] + mat_13 * uvw[3]
    uvw[1] = mat_21 * uvw[0] + mat_22 * uvw[1] + mat_23 * uvw[3]
    uvw[2] = mat_31 * uvw[0] + mat_32 * uvw[1] + mat_33 * uvw[3]


@jit(nopython=True, nogil=True, fastmath=True, parallel=False)
def uvw_planarwapprox(uvw, ra0, dec0, ra, dec, policy_type):
    '''
        Implements the coordinate uv transform associated with taking a planar
        approximation to w(n-1) as described in Kogan & Greisen's AIPS Memo 113
        This is essentially equivalent to rotating the facet to be tangent to
        the celestial sphere as Perley suggested to limit error, but it instead
        takes w into account in a linear approximation to the phase error near
        the facet centre. This keeps the facets parallel to the original facet
        plane. Of course this 2D taylor expansion of the first order is only
        valid over a small field of view, but that true of normal tilted
        faceting as well. Only a convolution can get rid of the (n-1)
        factor in the ME.
    '''
    d_ra = ra - ra0
    n_dec = dec
    o_dec = dec0
    c_d_ra = cos(d_ra)
    s_d_ra = sin(d_ra)
    c_new_dec = cos(n_dec)
    c_old_dec = cos(o_dec)
    s_new_dec = sin(n_dec)
    s_old_dec = sin(o_dec)
    li0 = c_new_dec * s_d_ra
    mi0 = s_new_dec * c_old_dec - c_new_dec * s_old_dec * c_d_ra
    ni0 = s_new_dec * s_old_dec + c_new_dec * c_old_dec * c_d_ra
    uvw[0] = uvw[0] - uvw[2] * li0 / ni0
    uvw[1] = uvw[1] - uvw[2] * mi0 / ni0


def policy(uvw, ra0, dec0, ra, dec, policy_type):
    pass


@overload(policy, inline="always")
def policy_impl(uvw, ra0, dec0, ra, dec, policy_type):
    if policy_type.literal_value == "None":
        return uvw_norotate
    elif policy_type.literal_value == "rotate":
        return uvw_rotate
    elif policy_type.literal_value == "wlinapprox":
        return uvw_planarwapprox
    else:
        raise ValueError("Invalid baseline transform policy type")
