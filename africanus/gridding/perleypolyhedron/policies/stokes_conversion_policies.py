from africanus.util.numba import overload


def stokes2corr(vis_in, vis_out, policy_type):
    pass


@overload(stokes2corr, inline="always")
def stokes2corrimpl(vis_in, vis_out, policy_type):
    if policy_type.literal_value == "XXYY_FROM_I":

        def XXYY_FROM_I(vis_in, vis_out, policy_type):
            vis_out[0] += vis_in
            vis_out[1] += vis_in

        return XXYY_FROM_I
    elif policy_type.literal_value == "XXXYYXYY_FROM_I":

        def XXXYYXYY_FROM_I(vis_in, vis_out, policy_type):
            vis_out[0] += vis_in
            vis_out[1] += 0
            vis_out[2] += 0
            vis_out[3] += vis_in

        return XXXYYXYY_FROM_I
    elif policy_type.literal_value == "RRLL_FROM_I":

        def RRLL_FROM_I(vis_in, vis_out, policy_type):
            vis_out[0] += vis_in
            vis_out[1] += vis_in

        return RRLL_FROM_I
    elif policy_type.literal_value == "RRRLLRLL_FROM_I":

        def RRRLLRLL_FROM_I(vis_in, vis_out, policy_type):
            vis_out[0] += vis_in
            vis_out[1] += 0
            vis_out[2] += 0
            vis_out[3] += vis_in

        return RRRLLRLL_FROM_I
    elif policy_type.literal_value == "XXYY_FROM_Q":

        def XXYY_FROM_Q(vis_in, vis_out, policy_type):
            vis_out[0] += vis_in
            vis_out[1] += -vis_in

        return XXYY_FROM_Q
    elif policy_type.literal_value == "XXXYYXYY_FROM_Q":

        def XXXYYXYY_FROM_Q(vis_in, vis_out, policy_type):
            vis_out[0] += vis_in
            vis_out[1] += 0
            vis_out[2] += 0
            vis_out[3] += -vis_in

        return XXXYYXYY_FROM_Q
    elif policy_type.literal_value == "RLLR_FROM_Q":

        def RLLR_FROM_Q(vis_in, vis_out, policy_type):
            vis_out[0] += vis_in
            vis_out[1] += vis_in

        return RLLR_FROM_Q
    elif policy_type.literal_value == "RRRLLRLL_FROM_Q":

        def RRRLLRLL_FROM_Q(vis_in, vis_out, policy_type):
            vis_out[0] += 0
            vis_out[1] += vis_in
            vis_out[2] += vis_in
            vis_out[3] += 0

        return RRRLLRLL_FROM_Q
    elif policy_type.literal_value == "XYYX_FROM_U":

        def XYYX_FROM_U(vis_in, vis_out, policy_type):
            vis_out[0] += vis_in
            vis_out[1] += vis_in

        return XYYX_FROM_U
    elif policy_type.literal_value == "XXXYYXYY_FROM_U":

        def XXXYYXYY_FROM_U(vis_in, vis_out, policy_type):
            vis_out[0] += 0
            vis_out[1] += vis_in
            vis_out[2] += vis_in
            vis_out[3] += 0

        return XXXYYXYY_FROM_U
    elif policy_type.literal_value == "RLLR_FROM_U":

        def RLLR_FROM_U(vis_in, vis_out, policy_type):
            vis_out[0] += 1.0j * vis_in
            vis_out[1] += -1.0j * vis_in

        return RLLR_FROM_U
    elif policy_type.literal_value == "RRRLLRLL_FROM_U":

        def RRRLLRLL_FROM_U(vis_in, vis_out, policy_type):
            vis_out[0] += 0.0
            vis_out[1] += 1.0j * vis_in
            vis_out[2] += -1.0j * vis_in
            vis_out[3] += 0.0

        return RRRLLRLL_FROM_U
    elif policy_type.literal_value == "XYYX_FROM_V":

        def XYYX_FROM_V(vis_in, vis_out, policy_type):
            vis_out[0] += 1.0j * vis_in
            vis_out[1] += -1.0j * vis_in

        return XYYX_FROM_V
    elif policy_type.literal_value == "XXXYYXYY_FROM_V":

        def XXXYYXYY_FROM_V(vis_in, vis_out, policy_type):
            vis_out[0] += 0.0
            vis_out[1] += 1.0j * vis_in
            vis_out[2] += -1.0j * vis_in
            vis_out[3] += 0.0

        return XXXYYXYY_FROM_V
    elif policy_type.literal_value == "RRLL_FROM_V":

        def RRLL_FROM_V(vis_in, vis_out, policy_type):
            vis_out[0] += vis_in
            vis_out[1] += -vis_in

        return RRLL_FROM_V
    elif policy_type.literal_value == "RRRLLRLL_FROM_V":

        def RRRLLRLL_FROM_V(vis_in, vis_out, policy_type):
            vis_out[0] += vis_in
            vis_out[1] += 0
            vis_out[2] += 0
            vis_out[3] += -vis_in

        return RRRLLRLL_FROM_V
    else:
        raise ValueError("Invalid stokes conversion")


def corr2stokes(vis_in, policy_type):
    pass


@overload(corr2stokes, inline="always")
def corr2stokesimpl(vis_in, policy_type):
    if policy_type.literal_value == "I_FROM_XXYY":
        return lambda vis_in, policy_type: (vis_in[0] + vis_in[1]) * 0.5
    elif policy_type.literal_value == "I_FROM_XXXYYXYY":
        return lambda vis_in, policy_type: (vis_in[0] + vis_in[3]) * 0.5
    elif policy_type.literal_value == "I_FROM_RRLL":
        return lambda vis_in, policy_type: (vis_in[0] + vis_in[1]) * 0.5
    elif policy_type.literal_value == "I_FROM_RRRLLRLL":
        return lambda vis_in, policy_type: (vis_in[0] + vis_in[3]) * 0.5
    elif policy_type.literal_value == "Q_FROM_XXYY":
        return lambda vis_in, policy_type: (vis_in[0] - vis_in[1]) * 0.5
    elif policy_type.literal_value == "Q_FROM_XXXYYXYY":
        return lambda vis_in, policy_type: (vis_in[0] - vis_in[3]) * 0.5
    elif policy_type.literal_value == "Q_FROM_RRRLLRLL":
        return lambda vis_in, policy_type: (vis_in[1] + vis_in[2]) * 0.5
    elif policy_type.literal_value == "U_FROM_XYYX":
        return lambda vis_in, policy_type: (vis_in[0] + vis_in[1]) * 0.5
    elif policy_type.literal_value == "U_FROM_XXXYYXYY":
        return lambda vis_in, policy_type: (vis_in[1] + vis_in[2]) * 0.5
    elif policy_type.literal_value == "U_FROM_RLLR":
        return lambda vis_in, policy_type: -1.0j * (vis_in[0] - vis_in[1]
                                                    ) * 0.5
    elif policy_type.literal_value == "U_FROM_RRRLLRLL":
        return lambda vis_in, policy_type: -1.0j * (vis_in[1] - vis_in[2]
                                                    ) * 0.5
    elif policy_type.literal_value == "V_FROM_RRLL":
        return lambda vis_in, policy_type: (vis_in[0] - vis_in[1]) * 0.5
    elif policy_type.literal_value == "V_FROM_RRRLLRLL":
        return lambda vis_in, policy_type: (vis_in[0] - vis_in[3]) * 0.5
    elif policy_type.literal_value == "V_FROM_XYYX":
        return lambda vis_in, policy_type: -1.0j * (vis_in[0] - vis_in[1]
                                                    ) * 0.5
    elif policy_type.literal_value == "V_FROM_XXXYYXYY":
        return lambda vis_in, policy_type: -1.0j * (vis_in[1] - vis_in[2]
                                                    ) * 0.5
    else:
        raise ValueError("Invalid stokes conversion")


def ncorr_out(policy_type):
    pass


@overload(ncorr_out, inline="always")
def ncorr_outimpl(policy_type):
    if policy_type.literal_value == "XXYY_FROM_I":
        return lambda policy_type: 2
    elif policy_type.literal_value == "XXXYYXYY_FROM_I":
        return lambda policy_type: 4
    elif policy_type.literal_value == "RRLL_FROM_I":
        return lambda policy_type: 2
    elif policy_type.literal_value == "RRRLLRLL_FROM_I":
        return lambda policy_type: 4
    elif policy_type.literal_value == "XXYY_FROM_Q":
        return lambda policy_type: 2
    elif policy_type.literal_value == "XXXYYXYY_FROM_Q":
        return lambda policy_type: 4
    elif policy_type.literal_value == "RLLR_FROM_Q":
        return lambda policy_type: 2
    elif policy_type.literal_value == "RRRLLRLL_FROM_Q":
        return lambda policy_type: 4
    elif policy_type.literal_value == "XYYX_FROM_U":
        return lambda policy_type: 2
    elif policy_type.literal_value == "XXXYYXYY_FROM_U":
        return lambda policy_type: 4
    elif policy_type.literal_value == "RLLR_FROM_U":
        return lambda policy_type: 2
    elif policy_type.literal_value == "RRRLLRLL_FROM_U":
        return lambda policy_type: 4
    elif policy_type.literal_value == "XYYX_FROM_V":
        return lambda policy_type: 2
    elif policy_type.literal_value == "XXXYYXYY_FROM_V":
        return lambda policy_type: 4
    elif policy_type.literal_value == "RRLL_FROM_V":
        return lambda policy_type: 2
    elif policy_type.literal_value == "RRRLLRLL_FROM_V":
        return lambda policy_type: 4
    else:
        raise ValueError("Invalid stokes conversion")


def ncorr_outpy(policy_type):
    if policy_type == "XXYY_FROM_I":
        return lambda: 2
    elif policy_type == "XXXYYXYY_FROM_I":
        return lambda: 4
    elif policy_type == "RRLL_FROM_I":
        return lambda: 2
    elif policy_type == "RRRLLRLL_FROM_I":
        return lambda: 4
    elif policy_type == "XXYY_FROM_Q":
        return lambda: 2
    elif policy_type == "XXXYYXYY_FROM_Q":
        return lambda: 4
    elif policy_type == "RLLR_FROM_Q":
        return lambda: 2
    elif policy_type == "RRRLLRLL_FROM_Q":
        return lambda: 4
    elif policy_type == "XYYX_FROM_U":
        return lambda: 2
    elif policy_type == "XXXYYXYY_FROM_U":
        return lambda: 4
    elif policy_type == "RLLR_FROM_U":
        return lambda: 2
    elif policy_type == "RRRLLRLL_FROM_U":
        return lambda: 4
    elif policy_type == "XYYX_FROM_V":
        return lambda: 2
    elif policy_type == "XXXYYXYY_FROM_V":
        return lambda: 4
    elif policy_type == "RRLL_FROM_V":
        return lambda: 2
    elif policy_type == "RRRLLRLL_FROM_V":
        return lambda: 4
    else:
        raise ValueError("Invalid stokes conversion")
