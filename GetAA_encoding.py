#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:33:56 2024

@author: ciqus
"""

import pandas as pd
import numpy as np
import sys
from smi_encoder import Smi_encoder
smi_encoder = Smi_encoder()

data = pd.read_csv("CycPeptMPDB_Monomer_All.csv")

aas = data['Symbol']
smi_aas = data['replaced_SMILES'].to_numpy()

aa_encode = {}

for idx, aa in enumerate(aas):
    smi = smi_aas[idx]
    encoded_smi = smi_encoder.encode_smi(smi, length=45)
    aa_encode[aa] = encoded_smi
    
print(aa_encode)