#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 09:59:08 2024

@author: ciqus
"""

import pandas as pd
from UP import CyclicPeptide
CP = CyclicPeptide()
import numpy as np
import json
import sys

data = pd.read_csv("CycPeptMPDB_Monomer_All.csv")
data = pd.read_csv("CycPeptMPDB_AllPeptide_Properties.csv")
# print(data.columns.to_list()[:30])
sequences = data['Sequence'].to_numpy()

l2_3 = {'2' : [], '3' : []}
for seq in sequences:
    _, lst_aas = CP.encode(sequences = [seq], length = 15)
    if len(lst_aas) <= 3:
        for aa in lst_aas:
            if 'Mono' in aa:
                l2_3[f'{len(lst_aas)}'] += [aa]

l2_3['2'] = list(set(l2_3['2']))
l2_3['3'] = list(set(l2_3['3']))
l2_3['3'].append('H2NEt_Phe')
print(l2_3)
with open('length_restriction.json', 'w') as out:
    json.dump(l2_3, out)

# # data['Natural_Analog'] = np.where(data['Natural_Analog'] == 'dW', 'W')
# data['Natural_Analog'].iloc[-1] = 'E'
# data['Natural_Analog'].iloc[-2] = 'D'
# data['Natural_Analog'].iloc[-3] = 'W'


# groups = data['Natural_Analog'].to_numpy()
# aas = data['Symbol'].to_numpy()
# terminal = data['Monomer_Type'].to_numpy()
# head_tail = []
# d = {}

# for g in groups:
#     d[g] = []
    
# for idx,aa in enumerate(aas):
#     g = groups[idx]
#     if terminal[idx] == 'Terminal':
#         head_tail.append(aa)
#     else:
#         d[g].append(aa)
    

# d['W'] = [d['W'][0]] + [d['W'][-1]] + d['W'][1:-1]
# d['E'] = [d['E'][0]] + [d['E'][-1]] + d['E'][1:-1]
# d['D'] = [d['D'][0]] + [d['D'][-1]] + d['D'][1:-1]
# k = list(d.keys())
# # print(k)
# k.sort()
# k = k[:-2] + [k[-1]] + [k[-2]]

# k_long = ['Alanine', 'Cysteine', 'Aspartic Acid', 'Glutamic Acid', 'Phenylalanine',
#      'Glycine', 'Histidine', 'Isoleucine', 'Lysine', 'Leucine', 'Methionine',
#      'Asparagine', 'Proline', 'Glutamine', 'Arginine', 'Serine', 'Threonine',
#      'Valine', 'Tryptophan', 'Tyrosine', 'NonNatural']
# d_long = {k_long[idx] : d[name] for idx, name in enumerate(k)}


# k_ord = ['Glycine', 'Alanine', 'Leucine', 'Isoleucine', 'Proline',
#      'Valine', 'Phenylalanine', 'Tryptophan', 'Tyrosine', 'Aspartic Acid', 'Glutamic Acid',
#      'Asparagine', 'Glutamine', 'Arginine', 'Lysine', 'Histidine', 'Serine',
#      'Threonine', 'Cysteine', 'Methionine', 'NonNatural']

# d_ord = {name : d_long[name] for idx, name in enumerate(k_ord)}
# d_ord['Terminal'] = head_tail

# with open('aa_dict.json', 'w') as out:
#     json.dump(d_ord, out)