#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 11:18:27 2024

@author: alfonsocabezon.vizos@usc.es
"""
import pandas as pd
import numpy as np
import pickle
import re
import hashlib
from smi_gen import smi_gen



class Smi_encoder:
    smi_gen = smi_gen()
    # Load characters of smiles
    with open('tokens_embedding.pkl', 'rb') as f:
        tokens = pickle.load(f)
    
    def __init__(self):
        self.tokens_dict = self.tokens
        data = pd.read_csv('CycPeptMPDB_SeqsandSMILES.csv', header = 0) # Load data
        self.smiles = data['SMILES'].to_numpy() # Get array of SMILES in database
        self.sequences = data['Sequence'].to_numpy() # Get array of seqs in database
        self.tokens = np.array(list(self.tokens.keys())) # Convert dict to array
        self.token_size = 1
        self.n_tokens = len(self.tokens)
        # Sort tokens by their length in descending order
        self.tokens = sorted(self.tokens, key = lambda x: len(x), reverse = True)
        smi_lengths = [len(self.smi_parser(smi)) for smi in self.smiles]
        self.maxlength = np.amax(smi_lengths) # Get biggest SMILE
        
    def biggest_smi(self):
        return self.maxlength
    
    def number_tokens(self):
        return self.n_tokens
    
    def smi_tokenizer(self, smi):
        '''
        Tokenize a SMILE string

        Parameters
        ----------
        smi : String
            SMILE string of a molecule.

        Returns
        -------
        tokens : List
            All the tokens that form the SMILE.

        '''
        pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smi)]
        assert smi == ''.join(tokens)
        return tokens
    
    def smi_parser(self, smi):
        '''
        This function reads the SMILE and converts it into a list of the tokens
        that constitute it. The code will read the sequence from left to write
        until it consumes it. Then it will build the list backwards.

        Parameters
        ----------
        smi : str
            String of the SMILE.

        Returns
        -------
        list
            List of the individual tokens that form the SMILE.

        '''
        for token_idx, token in enumerate(self.tokens):
            if smi.startswith(token):
                lst_smi = [token] # Put the token found in a list
                smi = smi[len(token):]
                rest = self.smi_parser(smi) # Second call. It will run until rest
                # is an empty list. Then will go "backwards" appending the tokens
                # encountered in previous calls
                if rest is None:
                    continue
                else:
                    return lst_smi + rest
        if smi == "":    
            return []
        else:
            raise Exception(f'SMILE cannot be parsed, token not found in library {self.tokens}')
    
    def encode_smi(self, sequences : list, pos : bool = True):
        '''
        Encodes smile into a vector

        Parameters
        ----------
        sequences : list
            List of sequences we want to get the encoded smile
        pos : bool, optional
            Wheter we want to codify tokens and its position or only tokens. The default is True.

        Returns
        -------
        encoded_smiles : TYPE
            DESCRIPTION.

        '''
        encoded_smiles = []
        for seq_idx, seq in enumerate(sequences):
            seq_code = np.zeros((self.maxlength)) # store encoding of sequence
            mask = np.copy(seq_code)
            mask[0::2] = 1.
            mask[1::2] = -1.
            if seq in self.sequences:
                sequence_smi = self.smiles[np.where(self.sequences == seq)][0]
            else:
                sequence_smi  = self.smi_gen.Seq2Smile(seq)
            parsed_smi = self.smi_parser(sequence_smi)
            # print(sequence_smi)
            # seq_encoded = [[self.token_values[token] * np.exp(i + 1)] for i, token in enumerate(parsed_smi)]
            # seq_encoded = [self.token_values[token] * (i+1/self.maxlength) for i, token in enumerate(parsed_smi)]
            if pos:
                seq_encoded = [int(hashlib.sha256(f'{i}{token}'.encode()).hexdigest(), 16) % 0.1 for i, token in enumerate(parsed_smi)]
            else:
                seq_encoded = [self.tokens_dict[token] for token in parsed_smi] # Get codes of tokens
            # print(seq_encoded)
            # sys.exit()
            # seq_encoded = [[self.token_values[token] * np.exp((i + 1)/self.maxlength)] for i, token in enumerate(parsed_smi)]
            for idx in range(len(seq_encoded)):
                seq_code[idx] = seq_encoded[idx]
            if pos:
                seq_code = seq_code * mask
            encoded_smiles.append(seq_code)
        encoded_smiles = np.array(encoded_smiles)
        return encoded_smiles
