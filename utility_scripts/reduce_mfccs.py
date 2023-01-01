# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 12:54:29 2022

@author: varya
"""

import pandas as pd
import os

def reduce_mfcs(Path):
    dfs = []
    means = []
    coef = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15']
    
    files = os.listdir(Path)
    #print(files)
    for i, f in enumerate(files):
        data = pd.read_csv(Path+'//'+f, sep='\t', names=coef)
        data['file'] = f'C1 {i}'
        dfs.append(data)
    for df in dfs:
        df[coef] = df[coef].astype(float)
        #print (df.dtypes)
        out = df.loc[:, coef].mean()
        out = out.to_frame().T
        means.append(out)
        
    frame = pd.concat(means, axis=0, ignore_index=True)
    #print(frame)
    return frame

#NC
C1_NC = reduce_mfcs('C:/Users/varya/master_thesis/C1_NC/')
C1_NC.insert(0, 'Talkers', 'C1')
C1_NC.insert(1, 'Category', 'C')
C1_NC.insert(2, 'Noise', 'NC')
#print(C1_NC)

C2_NC = reduce_mfcs('C:/Users/varya/master_thesis/C2_NC/')
C2_NC.insert(0, 'Talkers', 'C2')
C2_NC.insert(1, 'Category', 'C')
C2_NC.insert(2, 'Noise', 'NC')

C3_NC = reduce_mfcs('C:/Users/varya/master_thesis/C3_NC/')
C3_NC.insert(0, 'Talkers', 'C3')
C3_NC.insert(1, 'Category', 'C')
C3_NC.insert(2, 'Noise', 'NC')

C4_NC = reduce_mfcs('C:/Users/varya/master_thesis/C4_NC/')
C4_NC.insert(0, 'Talkers', 'C4')
C4_NC.insert(1, 'Category', 'C')
C4_NC.insert(2, 'Noise', 'NC')

C5_NC = reduce_mfcs('C:/Users/varya/master_thesis/C5_NC/')
C5_NC.insert(0, 'Talkers', 'C5')
C5_NC.insert(1, 'Category', 'C')
C5_NC.insert(2, 'Noise', 'NC')

C6_NC = reduce_mfcs('C:/Users/varya/master_thesis/C6_NC/')
C6_NC.insert(0, 'Talkers', 'C6')
C6_NC.insert(1, 'Category', 'C')
C6_NC.insert(2, 'Noise', 'NC')

#SSN
#C1_SSN = reduce_mfcs('C:/Users/varya/master_thesis/C1_SSN/')
file = 'C:/Users/varya/master_thesis/C1_SSN_means.txt'
coef = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15']
C1_SSN = pd.read_csv(file, sep='\t', names=coef)
#print(C1_SSN)
C1_SSN = C1_SSN[coef].astype(float)
C1_SSN.insert(0, 'Talkers', 'C1')
C1_SSN.insert(1, 'Category', 'C')
C1_SSN.insert(2, 'Noise', 'SSN')

C2_SSN = reduce_mfcs('C:/Users/varya/master_thesis/C2_SSN/')
C2_SSN.insert(0, 'Talkers', 'C2')
C2_SSN.insert(1, 'Category', 'C')
C2_SSN.insert(2, 'Noise', 'SSN')

C3_SSN = reduce_mfcs('C:/Users/varya/master_thesis/C3_SSN/')
C3_SSN.insert(0, 'Talkers', 'C3')
C3_SSN.insert(1, 'Category', 'C')
C3_SSN.insert(2, 'Noise', 'SSN')

C4_SSN = reduce_mfcs('C:/Users/varya/master_thesis/C4_SSN/')
C4_SSN.insert(0, 'Talkers', 'C4')
C4_SSN.insert(1, 'Category', 'C')
C4_SSN.insert(2, 'Noise', 'SSN')

C5_SSN = reduce_mfcs('C:/Users/varya/master_thesis/C5_SSN/')
C5_SSN.insert(0, 'Talkers', 'C5')
C5_SSN.insert(1, 'Category', 'C')
C5_SSN.insert(2, 'Noise', 'SSN')

C6_SSN = reduce_mfcs('C:/Users/varya/master_thesis/C6_SSN/')
C6_SSN.insert(0, 'Talkers', 'C6')
C6_SSN.insert(1, 'Category', 'C')
C6_SSN.insert(2, 'Noise', 'SSN')

#babble
C1_babble = reduce_mfcs('C:/Users/varya/master_thesis/C1_babble/')
C1_babble.insert(0, 'Talkers', 'C1')
C1_babble.insert(1, 'Category', 'C')
C1_babble.insert(2, 'Noise', 'babble')

C2_babble = reduce_mfcs('C:/Users/varya/master_thesis/C2_babble/')
C2_babble.insert(0, 'Talkers', 'C2')
C2_babble.insert(1, 'Category', 'C')
C2_babble.insert(2, 'Noise', 'babble')

C3_babble = reduce_mfcs('C:/Users/varya/master_thesis/C3_babble/')
C3_babble.insert(0, 'Talkers', 'C3')
C3_babble.insert(1, 'Category', 'C')
C3_babble.insert(2, 'Noise', 'babble')

C4_babble = reduce_mfcs('C:/Users/varya/master_thesis/C4_babble/')
C4_babble.insert(0, 'Talkers', 'C4')
C4_babble.insert(1, 'Category', 'C')
C4_babble.insert(2, 'Noise', 'babble')

C5_babble = reduce_mfcs('C:/Users/varya/master_thesis/C5_babble/')
C5_babble.insert(0, 'Talkers', 'C5')
C5_babble.insert(1, 'Category', 'C')
C5_babble.insert(2, 'Noise', 'babble')

C6_babble = reduce_mfcs('C:/Users/varya/master_thesis/C6_babble/')
C6_babble.insert(0, 'Talkers', 'C6')
C6_babble.insert(1, 'Category', 'C')
C6_babble.insert(2, 'Noise', 'babble')

df_Cs = pd.concat([C1_NC, C2_NC, C3_NC, C4_NC, C5_NC, C6_NC, C1_SSN, C2_SSN, C3_SSN, C4_SSN, C5_SSN, C6_SSN, C1_babble, C2_babble, C3_babble, C4_babble, C5_babble, C6_babble
], ignore_index=True, sort=False)
df_Cs.to_csv('C_mfcc_all.csv', encoding='utf-8', index=False)