#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 12:05:33 2024

@author: alfonsocabezonvizoso
"""

'''This script has been prepared to be incorporated to the web server for Cyclic
Peptide permeability prediction.'''
#Time management
import time
start = time.time()
# Sequence property calculator
from smi_gen import smi_gen
smi_gen = smi_gen()
#Unique Permutation
from UP import CyclicPeptide
CP = CyclicPeptide()
# Import ML related libraries
import tensorflow as tf
# Import rdkit libraries
from rdkit import Chem
from rdkit.Chem import Draw
# Import generic number and data processing libraries
import pandas as pd
import numpy as np
# All purpose libraries
import argparse
import os
from pickle import load
import sys

'''Define user inputs'''
###############################################################################
parser = argparse.ArgumentParser(description='''ML code''')
parser.add_argument('-s', '--SEQ', help='List of sequences to predict', nargs='+')
parser.add_argument('-m', '--MODELS', help='List of Trained models. One for classification\
                    and the other for regression', nargs = '+', type=list,
                    default = ['OptimizedClassification.h5' , 'OptimizedRegression.h5'])
parser.add_argument('-t', '--TYPE', help='Cyclization Head-Tail or Head-Side', nargs = '+')
parser.add_argument('-br', '--IDX', help='Index of the aa that bridges branching to CP only for head-tail', 
                    nargs = '+', type = int, default = [0])
parser.add_argument('-f', '--File', help='File containing different sequence for prediction', 
                    action = 'store', type = str, default = None)
args = parser.parse_args()
###############################################################################
if args.File:
    df = pd.read_csv(args.File, names = ['Sequence', 'Cyclization', 'idx_br'])
    df = df.fillna('None')
    or_seqs = df['Sequence'].to_numpy()
    print(or_seqs)
    cyclization = df['Cyclization'].to_numpy()
    cyclization = [t.lower() for t in cyclization]
    idx_br = df['idx_br'].to_numpy()
    idx_br = [ None if idx == 'None' else int(idx) for idx in idx_br]
else:
    or_seqs = args.SEQ
    cyclization = args.TYPE
    cyclization = [t.lower() for t in args.TYPE]
    idx_br = args.IDX
    idx_br = [ None if idx == 0 else idx for idx in idx_br]


scaler_reg = load(open('scaler_reg.pkl', 'rb')) # Load scaler
scaler_prop = load(open('scaler_seqprop.pkl', 'rb')) # Load scaler


sequences = [] # Initialize container for sequences
for idx, seq in enumerate(or_seqs):
    if cyclization[idx] == 'head-side':
        sequences.append(seq)
    elif cyclization[idx] == 'head-tail':
        up = CP.generate_permutations_and_metrics(seq, 15)
        sequences.append(up)
    else:
        raise Exception(f"Cyclization type {cyclization[idx]} was not recognized for sequence {seq}")

seqs = np.array(sequences) # Convert to np.array

TEST_seq_prop = [] # Container for whole sequence properties
seq_check = []
for idx, seq in enumerate(seqs):
    st_check = CP.check_cyclization(seq, idx_br = idx_br[idx])
    seq_check.append(st_check)
    smi = smi_gen.Seq2Smile(seq, idx_br = idx_br[idx], cyclization = cyclization[idx]) # Create SMI of seq
    Draw.MolToFile(Chem.MolFromSmiles(smi), f'CP{idx+1}.png', size = (500,500), fit_Image = True) # Draw molecule to file
    prop = smi_gen.Seq2Prop(seq, idx_br = idx_br[idx], cyclization = cyclization[idx]) # Get properties
    TEST_seq_prop.append(prop) # Append properties to list of properties

TEST_seq_prop = np.array(TEST_seq_prop) # Convert to np.array
TEST_seq_prop = scaler_prop.transform(TEST_seq_prop) # Scale properties
TEST, _ = CP.encode(sequences = seqs, length = 15, stop_signal = False) # Encode sequence 
classification = tf.keras.models.load_model(args.MODELS[0]) # Load classification model
regression = tf.keras.models.load_model(args.MODELS[1]) # Load regression model


class_pred = classification.predict([TEST, TEST_seq_prop]) # Predict class
new_y = []
prob_y = []
#Round values to be 0 or 1
for i in class_pred:
    for j in i:
        new_y.append(round(j))
        prob_y.append(j)
for idx, pred in enumerate(new_y):
    if pred == 1:
        new_y[idx] = 'High Permeability'
    else:
        new_y[idx] = 'Low Permeability'
        prob_y[idx] = 1 - prob_y[idx]

class_pred = new_y

reg_pred = regression.predict(TEST) # Predict Regression
reg_pred = scaler_reg.inverse_transform(reg_pred.reshape(-1,1))
reg_pred = reg_pred.reshape((-1,))
plus_minus = "\u00B1"
result = []
class_str = []
for idx, clas in enumerate(new_y):
    st_class = f'The sequence {or_seqs[idx]} presents {clas} with a probability of {prob_y[idx]:.0%}. '
    st_reg = f'The predicted Permeability for sequence {or_seqs[idx]} is: {reg_pred[idx]:.3f} {plus_minus} 0.477.'
    result_str = st_class + st_reg + seq_check[idx]
    result.append(result_str)

# reg_str = []

# for idx, reg in enumerate(reg_pred):
#     st = f'The predicted LogP for sequence {or_seqs[idx]} is: {reg:.3f} {plus_minus} 0.407'
#     reg_str.append(st)
# print(class_str, reg_str)
for res in result:
    print(res)



