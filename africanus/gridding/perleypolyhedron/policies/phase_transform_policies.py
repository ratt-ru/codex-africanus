from africanus.util.numba import overload
from numpy import pi, cos, sin, sqrt


def phase_norotate(vis,
                   uvw,
                   lambdas,
                   ra0,
                   dec0,
                   ra,
                   dec,
                   policy_type,
                   phasesign=1.0):
    pass


def phase_rotate(vis,
                 uvw,
                 lambdas,
                 ra0,
                 dec0,
                 ra,
                 dec,
                 policy_type,
                 phasesign=1.0):
    '''
        Convert ra,dec to l,m,n based on Synthesis Imaging II, Pg. 388
        The phase term (as documented in Perley & Cornwell (1992))
        calculation requires the delta l,m,n coordinates.
        Through simplification l0,m0,n0 = (0,0,1) (assume dec == dec0 and
        ra == ra0, and the simplification follows)
        l,m,n is then calculated using the new and original phase centres
        as per the relation on Pg. 388
        lambdas has the same shape as vis
    '''
    d_ra = ra - ra0
    d_dec = dec
    d_decp = dec0
    c_d_dec = cos(d_dec)
    s_d_dec = sin(d_dec)
    s_d_ra = sin(d_ra)
    c_d_ra = cos(d_ra)
    c_d_decp = cos(d_decp)
    s_d_decp = sin(d_decp)
    ll = c_d_dec * s_d_ra
    mm = (s_d_dec * c_d_decp - c_d_dec * s_d_decp * c_d_ra)
    nn = -(1 - sqrt(1 - ll * ll - mm * mm))
    for c in range(lambdas.size):
        x = phasesign * 2 * pi * (uvw[0] * ll + uvw[1] * mm +
                                  uvw[2] * nn) / lambdas[c]
        vis[c, :] *= cos(x) + 1.0j * sin(x)


def policy(vis, uvw, lambdas, ra0, dec0, ra, dec, policy_type, phasesign=1.0):
    pass


@overload(policy, inline="always")
def policy_impl(vis,
                uvw,
                lambdas,
                ra0,
                dec0,
                ra,
                dec,
                policy_type,
                phasesign=1.0):
    if policy_type.literal_value == "None" or \
       policy_type.literal_value is None:
        return phase_norotate
    elif policy_type.literal_value == "phase_rotate":
        return phase_rotate
    else:
        raise ValueError("Invalid baseline transform policy type")
