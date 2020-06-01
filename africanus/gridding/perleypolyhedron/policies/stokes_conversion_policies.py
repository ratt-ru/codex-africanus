from numba import jit, float32, float64, literally
from numpy import cos, sin
from numba.extending import overload

def stokes2corr(vis_in, policy_type):
    pass

@overload(stokes2corr, inline="always")
def stokes2corrimpl(vis_in, vis_out, policy_type):
    if policy_type.literal_value == "XXYY_FROM_I":
        return lambda vis_in, vis_out, policy_type: [vis_in,vis_in]
    elif policy_type.literal_value == "XXXYYXYY_FROM_I":
        return lambda vis_in, vis_out, policy_type: [vis_in,0,0,vis_in]
    elif policy_type.literal_value == "RRLL_FROM_I":
        return lambda vis_in, vis_out, policy_type: [vis_in,vis_in]
    elif policy_type.literal_value == "RRRLLRLL_FROM_I":
        return lambda vis_in, vis_out, policy_type: [vis_in,0,0,vis_in]
    elif policy_type.literal_value == "XXYY_FROM_Q":
        return lambda vis_in, vis_out, policy_type: [vis_in,-vis_in]
    elif policy_type.literal_value == "XXXYYXYY_FROM_Q":
        return lambda vis_in, vis_out, policy_type: [vis_in,0,0,-vis_in]
    elif policy_type.literal_value == "RLLR_FROM_Q":
        return lambda vis_in, vis_out, policy_type: [vis_in,vis_in]
    elif policy_type.literal_value == "RRRLLRLL_FROM_Q":
        return lambda vis_in, vis_out, policy_type: [0,vis_in,vis_in,0]
    elif policy_type.literal_value == "XYYX_FROM_U":
        return lambda vis_in, vis_out, policy_type: [vis_in,vis_in]
    elif policy_type.literal_value == "XXXYYXYY_FROM_U":
        return lambda vis_in, vis_out, policy_type: [0,vis_in,vis_in,0]
    elif policy_type.literal_value == "RLLR_FROM_U":
        return lambda vis_in, vis_out, policy_type: [1.0j*vis_in,-1.0j*vis_in]
    elif policy_type.literal_value == "RRRLLRLL_FROM_U":
        return lambda vis_in, vis_out, policy_type: [0,1.0j*vis_in,-1.0j*vis_in,0]
    elif policy_type.literal_value == "XYYX_FROM_V":
        return lambda vis_in, vis_out, policy_type: [1.0j*vis_in,-1.0j*vis_in]
    elif policy_type.literal_value == "XXXYYXYY_FROM_V":
        return lambda vis_in, vis_out, policy_type: [0,1.0j*vis_in,-1.0j*vis_in,0]
    elif policy_type.literal_value == "RRLL_FROM_V":
        return lambda vis_in, vis_out, policy_type: [vis_in,-vis_in]
    elif policy_type.literal_value == "RRRLLRLL_FROM_V":
        return lambda vis_in, vis_out, policy_type: [vis_in,0,0,-vis_in]
    else:
        raise ValueError("Invalid stokes conversion")
    
def corr2stokes(vis_in, policy_type):
    pass

@overload(corr2stokes, inline="always")
def corr2stokesimpl(vis_in, policy_type):
    if policy_type.literal_value == "I_FROM_XXYY":
        return lambda vis_in, policy_type: (vis_in[0]+vis_in[1])*0.5
    elif policy_type.literal_value == "I_FROM_XXXYYXYY":
        return lambda vis_in, policy_type: (vis_in[0]+vis_in[3])*0.5
    elif policy_type.literal_value == "I_FROM_RRLL":
        return lambda vis_in, policy_type: (vis_in[0]+vis_in[1])*0.5
    elif policy_type.literal_value == "I_FROM_RRRLLRLL":
        return lambda vis_in, policy_type: (vis_in[0]+vis_in[3])*0.5
    elif policy_type.literal_value == "Q_FROM_XXYY":
        return lambda vis_in, policy_type: (vis_in[0]-vis_in[1])*0.5
    elif policy_type.literal_value == "Q_FROM_XXXYYXYY":
        return lambda vis_in, policy_type: (vis_in[0]-vis_in[3])*0.5
    elif policy_type.literal_value == "Q_FROM_RRRLLRLL":
        return lambda vis_in, policy_type: (vis_in[1]+vis_in[2])*0.5
    elif policy_type.literal_value == "U_FROM_XYYX":
        return lambda vis_in, policy_type: (vis_in[0]+vis_in[1])*0.5
    elif policy_type.literal_value == "U_FROM_XXXYYXYY":
        return lambda vis_in, policy_type: (vis_in[1]+vis_in[2])*0.5
    elif policy_type.literal_value == "U_FROM_RLLR":
        return lambda vis_in, policy_type: -1.0j*(vis_in[0]-vis_in[1])*0.5
    elif policy_type.literal_value == "U_FROM_RRRLLRLL":
        return lambda vis_in, policy_type: -1.0j*(vis_in[1]-vis_in[2])*0.5
    elif policy_type.literal_value == "V_FROM_RRLL":
        return lambda vis_in, policy_type: (vis_in[0]-vis_in[1])*0.5
    elif policy_type.literal_value == "V_FROM_RRRLLRLL":
        return lambda vis_in, policy_type: (vis_in[0]-vis_in[3])*0.5
    elif policy_type.literal_value == "V_FROM_XYYX":
        return lambda vis_in, policy_type: -1.0j*(vis_in[0]-vis_in[1])*0.5
    elif policy_type.literal_value == "V_FROM_XXXYYXYY":
        return lambda vis_in, policy_type: -1.0j*(vis_in[1]-vis_in[2])*0.5
    else:
        raise ValueError("Invalid stokes conversion")