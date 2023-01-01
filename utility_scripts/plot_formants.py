# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 23:37:18 2022

@author: varya
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib.ticker import PercentFormatter


#print(np.std([64, 28, 26, 63, 23, 22]))
#print(np.std([49, 33, 28, 41, 24, 24, 22, 21, 33, 23, 26, 23]))

def plot_formants(input_file):
    id_f1 = []
    id_f2 = []
    id_f3 = []
    id_f0 = []
    f1 = {}
    f2 = {}
    f3 = {}
    mean_f1 = []
    mean_f2 =[]
    mean_f3 = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.split('\t')
            id_f0.append((line[0], line[2]))
            id_f1.append((line[0], line[3]))
            id_f2.append((line[0], line[4]))
            id_f3.append((line[0], line[5]))
    id_f0 = id_f0[1:]
    id_f1 = id_f1[1:]
    id_f2 = id_f2[1:]
    id_f3 = id_f3[1:]
    
    for item in id_f0:
        if item[0] not in f1:
            f1[item[0]] = []
        f1[item[0]].append(float(item[1]))
    for v in f1.values():
        v = list(map(float, v))
        mean = round((np.mean(v)), 2)
        mean_f1.append(mean)
        
    for item in id_f2:
       if item[0] not in f2:
           f2[item[0]] = []
       f2[item[0]].append(float(item[1]))
    for v in f2.values():
       v = list(map(float, v))
       mean1 = round((np.mean(v)), 2)
       mean_f2.append(mean1)
       
       
    for item in id_f3:
       if item[0] not in f3:
           f3[item[0]] = []
       f3[item[0]].append(float(item[1]))
    for v in f3.values():
       v = list(map(float, v))
       mean2 = np.mean(v)
       mean_f3.append(mean2)
       
    #print(mean_f1[0])
    return mean_f1, mean_f2, mean_f3

# def get_bins(x):
#     q25, q75 = np.percentile(x, [25, 75])
#     bin_width = 2 * (q75 - q25) * len(x) ** (-1/3)
#     bins = round((max(x) - min(x)) / bin_width)
#     #print("Freedman–Diaconis number of bins:", bins)
#     return bins

# fig1 = plt.figure(figsize =(40, 10))
AS1_NC = plot_formants('data_AS1_NC.Table')
print(len(AS1_NC[0]), len(AS1_NC[1]))
AS2_NC = plot_formants('data_AS2_NC.Table')
AS3_NC = plot_formants('data_AS3_NC.Table')
AS4_NC = plot_formants('data_AS4_NC.Table')
AS5_NC = plot_formants('data_AS5_NC.Table')
AS6_NC = plot_formants('data_AS6_NC.Table')


AS1_NC_df = pd.DataFrame({"F0": AS1_NC[0], "F2": AS1_NC[1], "F3": AS1_NC[2]})
AS1_NC_df.insert(0, 'Talkers', 'AS1')
AS2_NC_df = pd.DataFrame({"F0": AS2_NC[0], "F2": AS2_NC[1], "F3": AS2_NC[2]})
AS2_NC_df.insert(0, 'Talkers', 'AS2')
AS3_NC_df = pd.DataFrame({"F0": AS3_NC[0], "F2": AS3_NC[1], "F3": AS3_NC[2]})
AS3_NC_df.insert(0, 'Talkers', 'AS3')
AS4_NC_df = pd.DataFrame({"F0": AS4_NC[0], "F2": AS4_NC[1], "F3": AS4_NC[2]})
AS4_NC_df.insert(0, 'Talkers', 'AS4')
AS5_NC_df = pd.DataFrame({"F0": AS5_NC[0], "F2": AS5_NC[1], "F3": AS5_NC[2]})
AS5_NC_df.insert(0, 'Talkers', 'AS5')
AS6_NC_df = pd.DataFrame({"F0": AS6_NC[0], "F2": AS6_NC[1], "F3": AS6_NC[2]})
AS6_NC_df.insert(0, 'Talkers', 'AS6')

NC = pd.concat([AS1_NC_df, AS2_NC_df, AS3_NC_df, AS4_NC_df, AS5_NC_df, AS6_NC_df], axis=0, ignore_index=True)
NC.insert(1, 'Noise', 'NC')
#print(NC)

AS1_SSN = plot_formants('data_AS1_SSN.Table')
print(len(AS1_SSN[0]))
#print(np.mean(AS1_NC[1])) # it's a mean of F2! in no condition
AS2_SSN = plot_formants('data_AS2_SSN.Table')
AS3_SSN = plot_formants('data_AS3_SSN.Table')
AS4_SSN = plot_formants('data_AS4_SSN.Table')
AS5_SSN = plot_formants('data_AS5_SSN.Table')
AS6_SSN = plot_formants('data_AS6_SSN.Table')


AS1_SSN_df = pd.DataFrame({"F0": AS1_SSN[0], "F2": AS1_SSN[1], "F3": AS1_SSN[2]})
AS1_SSN_df.insert(0, 'Talkers', 'AS1')
AS2_SSN_df = pd.DataFrame({"F0": AS2_SSN[0], "F2": AS2_SSN[1], "F3": AS2_SSN[2]})
AS2_SSN_df.insert(0, 'Talkers', 'AS2')
AS3_SSN_df = pd.DataFrame({"F0": AS3_SSN[0], "F2": AS3_SSN[1], "F3": AS3_SSN[2]})
AS3_SSN_df.insert(0, 'Talkers', 'AS3')
AS4_SSN_df = pd.DataFrame({"F0": AS4_SSN[0], "F2": AS4_SSN[1], "F3": AS4_SSN[2]})
AS4_SSN_df.insert(0, 'Talkers', 'AS4')
AS5_SSN_df = pd.DataFrame({"F0": AS5_SSN[0], "F2": AS5_SSN[1], "F3": AS5_SSN[2]})
AS5_SSN_df.insert(0, 'Talkers', 'AS5')
AS6_SSN_df = pd.DataFrame({"F0": AS6_SSN[0], "F2": AS6_SSN[1], "F3": AS6_SSN[2]})
AS6_SSN_df.insert(0, 'Talkers', 'AS6')

SSN = pd.concat([AS1_SSN_df, AS2_SSN_df, AS3_SSN_df, AS4_SSN_df, AS5_SSN_df, AS6_SSN_df], axis=0, ignore_index=True)
SSN.insert(1, 'Noise', 'SSN')
#print(SSN)

AS1_babble = plot_formants('data_AS1_babble.Table')
print(len(AS1_babble[0]))
AS2_babble = plot_formants('data_AS2_babble.Table')
AS3_babble = plot_formants('data_AS3_babble.Table')
AS4_babble = plot_formants('data_AS4_babble.Table')
AS5_babble = plot_formants('data_AS5_babble.Table')
AS6_babble = plot_formants('data_AS6_babble.Table')

AS1_babble_df = pd.DataFrame({"F0": AS1_babble[0], "F2": AS1_babble[1], "F3": AS1_babble[2]})
AS1_babble_df.insert(0, 'Talkers', 'AS1')
AS2_babble_df = pd.DataFrame({"F0": AS2_babble[0], "F2": AS2_babble[1], "F3": AS2_babble[2]})
AS2_babble_df.insert(0, 'Talkers', 'AS2')
AS3_babble_df = pd.DataFrame({"F0": AS3_babble[0], "F2": AS3_babble[1], "F3": AS3_babble[2]})
AS3_babble_df.insert(0, 'Talkers', 'AS3')
AS4_babble_df = pd.DataFrame({"F0": AS4_babble[0], "F2": AS4_babble[1], "F3": AS4_babble[2]})
AS4_babble_df.insert(0, 'Talkers', 'AS4')
AS5_babble_df = pd.DataFrame({"F0": AS5_babble[0], "F2": AS5_babble[1], "F3": AS5_babble[2]})
AS5_babble_df.insert(0, 'Talkers', 'AS5')
AS6_babble_df = pd.DataFrame({"F0": AS6_babble[0], "F2": AS6_babble[1], "F3": AS6_babble[2]})
AS6_babble_df.insert(0, 'Talkers', 'AS6')

babble = pd.concat([AS1_babble_df, AS2_babble_df, AS3_babble_df, AS4_babble_df, AS5_babble_df, AS6_babble_df], axis=0, ignore_index=True)
babble.insert(1, 'Noise', 'babble')
#print(babble)

formants = pd.concat([NC, SSN, babble], axis=0, ignore_index=True)
#print(formants)
formants.to_csv('AS_f0_means.csv', encoding='utf-8', index=False)

#df_as_f2f3 = pd.concat([df_as1, df_as2, df_as3, df_as4, df_as5, df_as6], ignore_index=True, sort=False)

C1_NC = plot_formants('data_C1_NC.Table')
C2_NC = plot_formants('data_C2_NC.Table')
C3_NC = plot_formants('data_C3_NC.Table')
C4_NC = plot_formants('data_C4_NC.Table')
C5_NC = plot_formants('data_C5_NC.Table')
C6_NC = plot_formants('data_C6_NC.Table')

C1_NC_df = pd.DataFrame({"F0": C1_NC[0], "F2": C1_NC[1], "F3": C1_NC[2]})
C1_NC_df.insert(0, 'Talkers', 'C1')
C2_NC_df = pd.DataFrame({"F0": C2_NC[0], "F2": C2_NC[1], "F3": C2_NC[2]})
C2_NC_df.insert(0, 'Talkers', 'C2')
C3_NC_df = pd.DataFrame({"F0": C3_NC[0], "F2": C3_NC[1], "F3": C3_NC[2]})
C3_NC_df.insert(0, 'Talkers', 'C3')
C4_NC_df = pd.DataFrame({"F0": C4_NC[0], "F2": C4_NC[1], "F3": C4_NC[2]})
C4_NC_df.insert(0, 'Talkers', 'C4')
C5_NC_df = pd.DataFrame({"F0": C5_NC[0], "F2": C5_NC[1], "F3": C5_NC[2]})
C5_NC_df.insert(0, 'Talkers', 'C5')
C6_NC_df = pd.DataFrame({"F0": C6_NC[0], "F2": C6_NC[1], "F3": C6_NC[2]})
C6_NC_df.insert(0, 'Talkers', 'C6')

NC_C = pd.concat([C1_NC_df, C2_NC_df, C3_NC_df, C4_NC_df, C5_NC_df, C6_NC_df], axis=0, ignore_index=True)
NC_C.insert(1, 'Noise', 'NC')


C1_SSN = plot_formants('data_C1_SSN.Table')
C2_SSN = plot_formants('data_C2_SSN.Table')
C3_SSN = plot_formants('data_C3_SSN.Table')
C4_SSN = plot_formants('data_C4_SSN.Table')
C5_SSN = plot_formants('data_C5_SSN.Table')
C6_SSN = plot_formants('data_C6_SSN.Table')

C1_SSN_df = pd.DataFrame({"F0": C1_SSN[0], "F2": C1_SSN[1], "F3": C1_SSN[2]})
C1_SSN_df.insert(0, 'Talkers', 'C1')
C2_SSN_df = pd.DataFrame({"F0": C2_SSN[0], "F2": C2_SSN[1], "F3": C2_SSN[2]})
C2_SSN_df.insert(0, 'Talkers', 'C2')
C3_SSN_df = pd.DataFrame({"F0": C3_SSN[0], "F2": C3_SSN[1], "F3": C3_SSN[2]})
C3_SSN_df.insert(0, 'Talkers', 'C3')
C4_SSN_df = pd.DataFrame({"F0": C4_SSN[0], "F2": C4_SSN[1], "F3": C4_SSN[2]})
C4_SSN_df.insert(0, 'Talkers', 'C4')
C5_SSN_df = pd.DataFrame({"F0": C5_SSN[0], "F2": C5_SSN[1], "F3": C5_SSN[2]})
C5_SSN_df.insert(0, 'Talkers', 'C5')
C6_SSN_df = pd.DataFrame({"F0": C6_SSN[0], "F2": C6_SSN[1], "F3": C6_SSN[2]})
C6_SSN_df.insert(0, 'Talkers', 'C6')

SSN_C = pd.concat([C1_SSN_df, C2_SSN_df, C3_SSN_df, C4_SSN_df, C5_SSN_df, C6_SSN_df], axis=0, ignore_index=True)
SSN_C.insert(1, 'Noise', 'SSN')

C1_babble = plot_formants('data_C1_babble.Table')
C2_babble = plot_formants('data_C2_babble.Table')
C3_babble = plot_formants('data_C3_babble.Table')
C4_babble = plot_formants('data_C4_babble.Table')
C5_babble = plot_formants('data_C5_babble.Table')
C6_babble = plot_formants('data_C6_babble.Table')

C1_babble_df = pd.DataFrame({"F0": C1_babble[0], "F2": C1_babble[1], "F3": C1_babble[2]})
C1_babble_df.insert(0, 'Talkers', 'C1')
C2_babble_df = pd.DataFrame({"F0": C2_babble[0], "F2": C2_babble[1], "F3": C2_babble[2]})
C2_babble_df.insert(0, 'Talkers', 'C2')
C3_babble_df = pd.DataFrame({"F0": C3_babble[0], "F2": C3_babble[1], "F3": C3_babble[2]})
C3_babble_df.insert(0, 'Talkers', 'C3')
C4_babble_df = pd.DataFrame({"F0": C4_babble[0], "F2": C4_babble[1], "F3": C4_babble[2]})
C4_babble_df.insert(0, 'Talkers', 'C4')
C5_babble_df = pd.DataFrame({"F0": C5_babble[0], "F2": C5_babble[1], "F3": C5_babble[2]})
C5_babble_df.insert(0, 'Talkers', 'C5')
C6_babble_df = pd.DataFrame({"F0": C6_babble[0], "F2": C6_babble[1], "F3": C6_babble[2]})
C6_babble_df.insert(0, 'Talkers', 'C6')

babble_C = pd.concat([C1_babble_df, C2_babble_df, C3_babble_df, C4_babble_df, C5_babble_df, C6_babble_df], axis=0, ignore_index=True)
babble_C.insert(1, 'Noise', 'babble')

formants_C = pd.concat([NC_C, SSN_C, babble_C], axis=0, ignore_index=True)
print(formants_C)
#formants_C.to_csv('C_f0_means.csv', encoding='utf-8', index=False)


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    #print('x', ell_radius_x)
    #print('y', ell_radius_y)
    ax_x = ell_radius_x*2
    ax_y = ell_radius_y*2
    perimeter = math.pi * ( 3*(ax_x+ax_y) - math.sqrt( (3*ax_x + ax_y) * (ax_x + 3*ax_y) ) )
    area = ell_radius_x*ell_radius_y*math.pi
    print('Area', area)
    print('Perimeter', perimeter)
    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# fig, ax_nstd = plt.subplots(figsize=(6, 6))
# plt.scatter(C6_NC[1], C6_NC[0], marker='x', c='dodgerblue', label='no noise')
# xlist = []
# ylist = []

# # x_ave = np.average(C6_NC[1])
# # y_ave = np.average(C6_NC[0])
# #plt.scatter(xlist, ylist)
# # plt.scatter(x_ave, y_ave, c = 'red', marker = '*', s = 50)
# ax_nstd.scatter(C6_NC[1], C6_NC[0], s=0.5)
# confidence_ellipse(np.array(C6_NC[1]), np.array(C6_NC[0]), ax_nstd, n_std=3,
#                    edgecolor='royalblue')#, linestyle='--')

# plt.xlabel('F2')
# plt.ylabel('F1')

# plt.scatter(C6_SSN[1], C6_SSN[0], marker='p', c='pink', label='speech shaped noise')
# alist = []
# blist = []

# # a_ave = np.average(C6_SSN[1])
# # b_ave = np.average(C6_SSN[0])
# # plt.scatter(alist, blist)
# # plt.scatter(a_ave, b_ave, c = 'green', marker = 'o', s = 50)
# ax_nstd.scatter(C6_SSN[1], C6_SSN[0], s=0.5)
# confidence_ellipse(np.array(C6_SSN[1]), np.array(C6_SSN[0]), ax_nstd, n_std=3,
#                     edgecolor='palevioletred')#, linestyle='--')
# plt.xlabel('F2')
# plt.ylabel('F1')

# plt.scatter(C6_babble[1], C6_babble[0], marker='^', c='violet', label='babble noise')
# # dlist = []
# # flist = []
# # d_ave = np.average(C6_babble[1])
# # f_ave = np.average(C6_babble[0])
# # plt.scatter(dlist, flist)
# # plt.scatter(d_ave, f_ave, c = 'green', marker = 's', s = 50)
# ax_nstd.scatter(C6_babble[1], C6_babble[0], s=0.5)
# confidence_ellipse(np.array(C6_babble[1]), np.array(C6_babble[0]), ax_nstd, n_std=3,
#                     edgecolor='magenta')#, linestyle='--')
# plt.xlabel('F2')
# plt.ylabel('F1')
# plt.legend(scatterpoints=1,
#            loc='upper right',
#            ncol=3,
#            fontsize=8)
# plt.title('Control 6')
# plt.show()

nc = [np.mean(AS1_NC[1]), np.mean(AS2_NC[1]), np.mean(AS3_NC[1]), np.mean(AS4_NC[1]), np.mean(AS5_NC[1]), np.mean(AS6_NC[1])]
nss = [np.mean(AS1_SSN[1]), np.mean(AS2_SSN[1]), np.mean(AS3_SSN[1]), np.mean(AS4_SSN[1]), np.mean(AS5_SSN[1]), np.mean(AS6_SSN[1])]
bbl = [np.mean(AS1_babble[1]), np.mean(AS2_babble[1]), np.mean(AS3_babble[1]), np.mean(AS4_babble[1]), np.mean(AS5_babble[1]), np.mean(AS6_babble[1])]

nc_f3 = [np.mean(AS1_NC[2]), np.mean(AS2_NC[2]), np.mean(AS3_NC[2]), np.mean(AS4_NC[2]), np.mean(AS5_NC[2]), np.mean(AS6_NC[2])]
nss_f3 = [np.mean(AS1_SSN[2]), np.mean(AS2_SSN[2]), np.mean(AS3_SSN[2]), np.mean(AS4_SSN[2]), np.mean(AS5_SSN[2]), np.mean(AS6_SSN[2])]
bbl_f3 = [np.mean(AS1_babble[2]), np.mean(AS2_babble[2]), np.mean(AS3_babble[2]), np.mean(AS4_babble[2]), np.mean(AS5_babble[2]), np.mean(AS6_babble[2])]


nc_c = [np.mean(C1_NC[1]), np.mean(C2_NC[1]), np.mean(C3_NC[1]), np.mean(C4_NC[1]), np.mean(C5_NC[1]), np.mean(C6_NC[1])]
nss_c = [np.mean(C1_SSN[1]), np.mean(C2_SSN[1]), np.mean(C3_SSN[1]), np.mean(C4_SSN[1]), np.mean(C5_SSN[1]), np.mean(C6_SSN[1])]
bbl_c = [np.mean(C1_babble[1]), np.mean(C2_babble[1]), np.mean(C3_babble[1]), np.mean(C4_babble[1]), np.mean(C5_babble[1]), np.mean(C6_babble[1])]


nc_cf3 = [np.mean(C1_NC[2]), np.mean(C2_NC[2]), np.mean(C3_NC[2]), np.mean(C4_NC[2]), np.mean(C5_NC[2]), np.mean(C6_NC[2])]
nss_cf3 = [np.mean(C1_SSN[2]), np.mean(C2_SSN[2]), np.mean(C3_SSN[2]), np.mean(C4_SSN[2]), np.mean(C5_SSN[2]), np.mean(C6_SSN[2])]
bbl_cf3 = [np.mean(C1_babble[2]), np.mean(C2_babble[2]), np.mean(C3_babble[2]), np.mean(C4_babble[2]), np.mean(C5_babble[2]), np.mean(C6_babble[2])]

C = ['C1','C2', 'C3', 'C4', 'C5', 'C6']
AS = ['AS1', 'AS2', 'AS3', 'AS4', 'AS5', 'AS6']


#F2 all AS all conditions
d_f2 = {'Talkers': AS, 'F2 NC': nc, 'F2 SSN': nss, 'F2 babble': bbl}
#F3 all AS all conditions
d_f3 = {'Talkers': AS, 'F3 NC': nc_f3, 'F3 SSN': nss_f3, 'F3 babble': bbl_f3}


#F2 all Control all conditions
dc_f2 = {'Talkers': C, 'F2 NC': nc_c, 'F2 SSN': nss_c, 'F2 babble': bbl_c}
#F3 all Conrtols all conditions
dc_f3 = {'Talkers': C, 'F3 NC': nc_cf3, 'F3 SSN': nss_cf3, 'F3 babble': bbl_cf3}

# # csv for Asperger formants
# df_f2 = pd.DataFrame(d_f2)
# df_f2.to_csv('AS_f2_all.csv', encoding='utf-8', index=False)

# df_f3 = pd.DataFrame(d_f3)
# df_f3.to_csv('AS_f3_all.csv', encoding='utf-8', index=False)
# # print(df_f3)

# # csv for Controls formants
# dfc_f2 = pd.DataFrame(dc_f2)
# dfc_f2.to_csv('C_f2_all.csv', encoding='utf-8', index=False)

# dfc_f3 = pd.DataFrame(dc_f3)
# dfc_f3.to_csv('C_f3_all.csv', encoding='utf-8', index=False)
# #print(dfc_f2)


def get_percent_increase(nc, ssn, bbl):
	increase_ssn = ssn - nc
	increase_bbl = bbl - nc
	percent_ssn = (increase_ssn/nc)*100
	percent_bbl = (increase_bbl/nc)*100
	
	return round(percent_ssn, 2), round(percent_bbl, 2)

AS1_NC_mean = np.mean(AS1_NC[1])
#print(AS1_NC_mean)
AS1_SSN_mean = np.mean(AS1_SSN[1])
print(AS1_SSN_mean)
AS1_babble_mean = np.mean(AS1_babble[1])
#print(AS1_babble_mean)

AS2_NC_mean = np.mean(AS2_NC[1])
#print(AS2_NC_mean)
AS2_SSN_mean = np.mean(AS2_SSN[1])
print(AS2_SSN_mean)
AS2_babble_mean = np.mean(AS2_babble[1])

AS3_NC_mean = np.mean(AS3_NC[1])
#print(AS3_NC_mean)
AS3_SSN_mean = np.mean(AS3_SSN[1])
print(AS3_SSN_mean)
AS3_babble_mean = np.mean(AS3_babble[1])

AS4_NC_mean = np.mean(AS4_NC[1])
#print(AS4_NC_mean)
AS4_SSN_mean = np.mean(AS4_SSN[1])
print(AS4_SSN_mean)
AS4_babble_mean = np.mean(AS4_babble[1])

AS5_NC_mean = np.mean(AS5_NC[1])
#print(AS5_NC_mean)
AS5_SSN_mean = np.mean(AS5_SSN[1])
print(AS5_SSN_mean)
AS5_babble_mean = np.mean(AS5_babble[1])

AS6_NC_mean = np.mean(AS6_NC[1])
#print(AS6_NC_mean)
AS6_SSN_mean = np.mean(AS6_SSN[1])
print(AS6_SSN_mean)
AS6_babble_mean = np.mean(AS6_babble[1])


percent_AS1 = get_percent_increase(AS1_NC_mean, AS1_SSN_mean, AS1_babble_mean)
print(percent_AS1)
percent_AS2 = get_percent_increase(AS2_NC_mean, AS2_SSN_mean, AS2_babble_mean)
print(percent_AS2)
percent_AS3 = get_percent_increase(AS3_NC_mean, AS3_SSN_mean, AS3_babble_mean)
print(percent_AS3)
percent_AS4 = get_percent_increase(AS4_NC_mean, AS4_SSN_mean, AS4_babble_mean)
print(percent_AS4)
percent_AS5 = get_percent_increase(AS5_NC_mean, AS5_SSN_mean, AS5_babble_mean)
print(percent_AS5)
percent_AS6 = get_percent_increase(AS6_NC_mean, AS6_SSN_mean, AS6_babble_mean)
print(percent_AS6)
ssn_m = [percent_AS1[0], percent_AS2[0], percent_AS3[0], percent_AS4[0], percent_AS5[0], percent_AS6[0]]
bbl_m = [percent_AS1[1], percent_AS2[1], percent_AS3[1], percent_AS4[1], percent_AS5[1], percent_AS6[1]]
print(np.mean(bbl_m))


C1_NC_mean = np.mean(C1_NC[2])
C1_SSN_mean = np.mean(C1_SSN[2])
C1_babble_mean = np.mean(C1_babble[2])

C2_NC_mean = np.mean(C2_NC[2])
C2_SSN_mean = np.mean(C2_SSN[2])
C2_babble_mean = np.mean(C2_babble[2])

C3_NC_mean = np.mean(C3_NC[2])
C3_SSN_mean = np.mean(C3_SSN[2])
C3_babble_mean = np.mean(C3_babble[2])

C4_NC_mean = np.mean(C4_NC[2])
C4_SSN_mean = np.mean(C4_SSN[2])
C4_babble_mean = np.mean(C4_babble[2])

C5_NC_mean = np.mean(C5_NC[2])
C5_SSN_mean = np.mean(C5_SSN[2])
C5_babble_mean = np.mean(C5_babble[2])

C6_NC_mean = np.mean(C6_NC[2])
C6_SSN_mean = np.mean(C6_SSN[2])
C6_babble_mean = np.mean(C6_babble[2])

# percent_C1 = get_percent_increase(C1_NC_mean, C1_SSN_mean, C1_babble_mean)
# print(percent_C1)
# percent_C2 = get_percent_increase(C2_NC_mean, C2_SSN_mean, C2_babble_mean)
# print(percent_C2)
# percent_C3 = get_percent_increase(C3_NC_mean, C3_SSN_mean, C3_babble_mean)
# print(percent_C3)
# percent_C4 = get_percent_increase(C4_NC_mean, C4_SSN_mean, C4_babble_mean)
# print(percent_C4)
# percent_C5 = get_percent_increase(C5_NC_mean, C5_SSN_mean, C5_babble_mean)
# print(percent_C5)
# percent_C6 = get_percent_increase(C6_NC_mean, C6_SSN_mean, C6_babble_mean)
# print(percent_C6)



def calculate_perimeter(a,b):
     perimeter = math.pi * ( 3*(a+b) - math.sqrt( (3*a + b) * (a + 3*b) ) )
     return perimeter

calculate_perimeter(2,3)

#percent_mean_ssn = round(np.mean([percent_AS1[0], percent_AS2[0],percent_AS3[0], percent_AS4[0], percent_AS5[0], percent_AS6[0]]), 2)
#percent_mean_bbl = round(np.mean([percent_AS1[1], percent_AS2[1],percent_AS3[1], percent_AS4[1], percent_AS5

#df = pd.concat([df1, df2], ignore_index=True, sort=False).fill
#print(df) #f1
# bins_f1 = get_bins(AS1_NC[0])
# bins_f2 = get_bins(AS1_NC[1])
# bins_f3 = get_bins(AS1_NC[2])
 


# plt.hist(AS1_NC[0], edgecolor='black', density=False, weights=np.ones(len(AS1_NC[0])) / len(AS1_NC[0]), bins=bins_f1)
# plt.hist(AS1_NC[1], edgecolor='black',density=False, weights=np.ones(len(AS1_NC[1])) / len(AS1_NC[1]), bins=bins_f2)
# plt.hist(AS1_NC[2], edgecolor='black',density=False, weights=np.ones(len(AS1_NC[2])) / len(AS1_NC[2]), bins=bins_f3)
# mn, mx = plt.xlim()
# plt.xlim(mn, mx)
# kde_xs = np.linspace(mn, mx, 300)
# kde_f1 = st.gaussian_kde(AS1_NC[0])
# kde_f2 = st.gaussian_kde(AS1_NC[1])
# kde_f3 = st.gaussian_kde(AS1_NC[2])
# plt.plot(kde_xs, kde_f1.pdf(kde_xs), lw=3, label="F1")
# plt.plot(kde_xs, kde_f2.pdf(kde_xs), lw=3, label="F2")
# plt.plot(kde_xs, kde_f3.pdf(kde_xs), lw=3, label="F3")

# plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# #plt.ylim(0, 0.02)
# plt.legend(loc="upper left")
# plt.ylabel('Probability')
# plt.xlabel('Frequency (Hz)')
# plt.title("AS1 No Condition")

# plt.show()
