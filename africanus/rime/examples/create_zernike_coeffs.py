import numpy as np

coeff_xx = np.array([-1.75402394e-01-0.14477493j,  9.97613164e-02+0.0965587j, 2.10125186e-01+0.17758039j, -1.69924807e-01-0.11709054j, -4.30692473e-02-0.0349753j,  7.74099248e-02+0.03703381j, -7.51374250e-03+0.01024362j,  1.40650300e-03+0.02095283j, -1.39579628e-02-0.01244837j, -7.93278560e-04-0.02543059j, 3.61356760e-03+0.00202427j,  2.31464542e-03-0.00018854j, 9.05646002e-03-0.00062068j, -1.70722541e-04-0.00577695j, 4.06321372e-03-0.00489419j,  4.70079669e-03-0.0042618j, 1.21656158e-02+0.01113621j])
coeff_xy = np.array([-0.00378847+0.00520143j,  0.02002285+0.02665323j, -0.00843154+0.00852609j,  0.00449256-0.00522683j, -0.00478961-0.00633869j, -0.01326315-0.01646019j, -0.01497431-0.0140809j, -0.00117441+0.00205662j, -0.00048141+0.00075124j])
coeff_yx =  np.array([-2.23911814e-03-0.00547617j, -4.75247330e-03-0.00745264j, -2.21456777e-03+0.00619276j,  1.20189576e-02+0.01197778j, -2.01741060e-02-0.01792336j,  7.51580997e-05+0.00209391j, -3.31077481e-04-0.0036083j,  1.16293179e-02+0.01279112j])
coeff_yy = np.array([-0.17742637-0.1378773j,  0.09912589+0.09639812j, 0.21176327+0.17682041j, -0.16836034-0.11677519j, -0.0428337 - 0.03446249j,  0.07525696+0.03761065j, -0.00754467+0.00811033j,  0.01189913+0.01875151j, 0.00248063+0.00179074j,  0.00160786+0.00614232j, -0.01133655-0.01143651j,  0.00470805-0.01920698j, 0.0038768 - 0.00601548j,  0.00172058-0.00385759j, -0.01082336-0.00432746j, -0.0009297 + 0.00796986j, 0.01785803+0.00319331j])

noll_index_xx = np.array([10,  3, 21, 36,  0, 55, 16, 28, 37, 46, 23,  6, 15,  2,  5,  7, 57])
noll_index_xy = np.array([12, 28, 22,  4, 38, 16, 46, 15,  7])
noll_index_yx = np.array([12, 22,  4, 15, 29, 38,  7, 45])
noll_index_yy = np.array([10,  3, 21, 36,  0, 55, 28, 16, 11, 23, 37, 46,  6,  2, 15,  5, 29])

z_dict = {'coeff' :
{'xx' : coeff_xx,
'xy' : coeff_xy,
'yx' : coeff_yx,
'yy' : coeff_yy},
'noll_index' :
{'xx' : noll_index_xx,
'xy' : noll_index_xy,
'yx' : noll_index_yx,
'yy' : noll_index_yy}}

np.save("./zernike_coeffs.npy", z_dict)