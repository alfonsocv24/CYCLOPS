#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:59:31 2024

@author: alfonsocabezon.vizoso@usc.es
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json

class CyclicPeptide:
    all_features = [
        "MaxEStateIndex", "MinEStateIndex", "MaxAbsEStateIndex", "MinAbsEStateIndex", 
        "qed", "MolWt", "HeavyAtomMolWt", "ExactMolWt", "NumValenceElectrons", "NumRadicalElectrons", 
        "MaxPartialCharge", "MinPartialCharge", "MaxAbsPartialCharge", "MinAbsPartialCharge", 
        "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3", "BCUT2D_MWHI", "BCUT2D_MWLOW", 
        "BCUT2D_CHGHI", "BCUT2D_CHGLO", "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW", "BCUT2D_MRHI", 
        "BCUT2D_MRLOW", "BalabanJ", "BertzCT", "Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n", 
        "Chi1v", "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v", "HallKierAlpha", "Ipc", 
        "Kappa1", "Kappa2", "Kappa3", "LabuteASA", "PEOE_VSA1", "PEOE_VSA10", "PEOE_VSA11", 
        "PEOE_VSA12", "PEOE_VSA13", "PEOE_VSA14", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5", 
        "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9", "SMR_VSA1", "SMR_VSA10", "SMR_VSA2", 
        "SMR_VSA3", "SMR_VSA4", "SMR_VSA5", "SMR_VSA6", "SMR_VSA7", "SMR_VSA8", "SMR_VSA9", 
        "SlogP_VSA1", "SlogP_VSA10", "SlogP_VSA11", "SlogP_VSA12", "SlogP_VSA2", "SlogP_VSA3", 
        "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6", "SlogP_VSA7", "SlogP_VSA8", "SlogP_VSA9", 
        "TPSA", "EState_VSA1", "EState_VSA10", "EState_VSA11", "EState_VSA2", "EState_VSA3", 
        "EState_VSA4", "EState_VSA5", "EState_VSA6", "EState_VSA7", "EState_VSA8", "EState_VSA9", 
        "VSA_EState1", "VSA_EState10", "VSA_EState2", "VSA_EState3", "VSA_EState4", "VSA_EState5", 
        "VSA_EState6", "VSA_EState7", "VSA_EState8", "VSA_EState9", "FractionCSP3", "HeavyAtomCount", 
        "NHOHCount", "NOCount", "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticRings",
        "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAromaticRings", "NumHAcceptors", 
        "NumHDonors", "NumHeteroatoms", "NumRotatableBonds", "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", 
        "NumSaturatedRings", "RingCount", "MolLogP", "MolMR", "fr_Al_COO", "fr_Al_OH", "fr_Al_OH_noTert", 
        "fr_ArN", "fr_Ar_COO", "fr_Ar_N", "fr_Ar_NH", "fr_Ar_OH", "fr_COO", "fr_COO2", "fr_C_O", "fr_C_O_noCOO", 
        "fr_C_S", "fr_HOCCN", "fr_Imine", "fr_NH0", "fr_NH1", "fr_NH2", "fr_N_O", "fr_Ndealkylation1", 
        "fr_Ndealkylation2", "fr_Nhpyrrole", "fr_SH", "fr_aldehyde", "fr_alkyl_carbamate", "fr_alkyl_halide", 
        "fr_allylic_oxid", "fr_amide", "fr_amidine", "fr_aniline", "fr_aryl_methyl", "fr_azide", "fr_azo", "fr_barbitur", 
        "fr_benzene", "fr_benzodiazepine", "fr_bicyclic", "fr_diazo", "fr_dihydropyridine", "fr_epoxide", "fr_ester", 
        "fr_ether", "fr_furan", "fr_guanido", "fr_halogen", "fr_hdrzine", "fr_hdrzone", "fr_imidazole", "fr_imide", 
        "fr_isocyan", "fr_isothiocyan", "fr_ketone", "fr_ketone_Topliss", "fr_lactam", "fr_lactone", "fr_methoxy", 
        "fr_morpholine", "fr_nitrile", "fr_nitro", "fr_nitro_arom", "fr_nitro_arom_nonortho", "fr_nitroso", 
        "fr_oxazole", "fr_oxime", "fr_para_hydroxylation", "fr_phenol", "fr_phenol_noOrthoHbond", "fr_phos_acid", 
        "fr_phos_ester", "fr_piperdine", "fr_piperzine", "fr_priamide", "fr_prisulfonamd", "fr_pyridine", 
        "fr_quatN", "fr_sulfide", "fr_sulfonamd", "fr_sulfone", "fr_term_acetylene", "fr_tetrazole", 
        "fr_thiazole", "fr_thiocyan", "fr_thiophene", "fr_unbrch_alkane", "fr_urea"]
    
    monomers = pd.read_csv("CycPeptMPDB_Monomer_All.csv")
    amino_acids = monomers['Symbol'].values
    amino_acid_values = {aa: i+1 for i, aa in enumerate(amino_acids)}
    bridge_term_N = ['T', 'Orn', 'dK', 'K', 'Me_dK', 'meK', 'dLeu(3R-OH)', 'Me_Bmt(E)']
    bridge_term_C = ['D', 'meD', 'E', 'meE','Mono9', 'Mono10', 'Mono11', 'Mono12']
    bridge_aas = bridge_term_N + bridge_term_C
    terminal_aa_dict = {'ac-' : ['T', 'Orn', 'dK', 'K', 'Me_dK', 'meK', 'dLeu(3R-OH)', 'Me_Bmt(E)'],
                        'deca-' : ['T', 'Orn', 'dK', 'K', 'Me_dK', 'meK', 'dLeu(3R-OH)', 'Me_Bmt(E)'],
                        'medl-' : ['T', 'Orn', 'dK', 'K', 'Me_dK', 'meK', 'dLeu(3R-OH)', 'Me_Bmt(E)'],
                        'glyco-' : ['T', 'Orn', 'dK', 'K', 'Me_dK', 'meK', 'dLeu(3R-OH)', 'Me_Bmt(E)'],
                        'Mono21-' : ['T', 'Orn', 'dK', 'K', 'Me_dK', 'meK', 'dLeu(3R-OH)', 'Me_Bmt(E)'],
                        'Mono22-' : ['T', 'Orn', 'dK', 'K', 'Me_dK', 'meK', 'dLeu(3R-OH)', 'Me_Bmt(E)'],
                        '-pip' : ['D', 'meD', 'E', 'meE','Mono9', 'Mono10', 'Mono11', 'Mono12']}
    # with open('terminal_aa_dict.json', 'w') as out:
    #     json.dump(terminal_aa_dict, out)
        
    def __init__(self):
        # data = pd.read_csv("CycPeptMPDB_AAs_Monomers.csv")
        data = pd.read_csv("CycPeptMPDB_Monomer_All_FixSMI.csv")
        self.seq_data = pd.read_csv("CycPeptMPDB_Peptide_Properties.csv")
        seq_prop_col = self.seq_data.columns.to_list()
        seq_prop_col = seq_prop_col[2:]
        self.all_symbols = data["Symbol"].values
        self.term_aas = np.array(list(self.terminal_aa_dict.keys()))
        self.features = data[self.all_features].to_numpy()
        # Scale features
        self.scaled_features = StandardScaler().fit_transform(self.features[:-3]) # Scale without NULL aa
        # Add all features to the sself.features array the last one will be all 0s corresponding to the NULL aa
        for idx in range(len(self.scaled_features)):
            self.features[idx] = self.scaled_features[idx] # Change features by scaled ones
        
        # Sort amino acids and properties by the length of the amino acid name (descending)
        self.all_symbols, final_feat = zip(*sorted(zip(self.all_symbols, self.features), key=lambda x: len(x[0]), reverse=True))
        # Convert features to numpy array. It will be easier later on.
        self.final_feat = np.array(final_feat)
        
    # Parses a sequence and returns an array with integers representing amino acids.
    def parse_sequence(self, subsequence):
        if len(subsequence) == 0:
            return []

        for symbol_idx, symbol in enumerate(self.all_symbols):
            if subsequence.startswith(symbol):
                ret = self.parse_sequence(subsequence[len(symbol):])
                if ret is None:
                    continue
                else:
                    return [symbol_idx] + ret
        return None

    def encode(self, sequences, length, stop_signal=True, sequence_properties = False):
        if stop_signal:
            ret = np.zeros((len(sequences), length, self.final_feat.shape[1] + 1))
        else:
            ret = np.zeros((len(sequences), length, self.final_feat.shape[1]))
        lst_seq_prop = []
        for seq_idx, seq in enumerate(sequences):
            sequence = []
            symbols_indices = self.parse_sequence(seq)
            if symbols_indices is None:
                raise Exception(f"Could not parse sequence {seq}")
            else:
                for pos, symbol_idx in enumerate(symbols_indices):
                    ret[seq_idx, pos, :self.final_feat.shape[1]] = self.final_feat[symbol_idx]
                    #print(self.all_symbols[symbol_idx])
                    sequence.append(self.all_symbols[symbol_idx])
            
            if stop_signal:
                ret[seq_idx, pos - 1, -1] = 1
            if sequence_properties:
                idx_feat_seq = self.seq_data[self.seq_data['Sequence'] == seq].index.values
                seq_feat = self.seq_properties[idx_feat_seq]
                # print(seq_feat[0])
                lst_seq_prop.append(seq_feat[0])
        lst_seq_prop = np.array(lst_seq_prop).astype(np.float32)
        if sequence_properties:
            return ret, sequence, lst_seq_prop
        else:
            return ret, sequence
    
    def fragment_peptide(self, seq, idx_br : int = None, length : int = 15):
        '''
        Fragment sequence into amino acids. If branching is detected, the aas
        forming the branching are put together.
    
        Parameters
        ----------
        seq : str
            Peptide sequence we want to fragment.
        length : int. Optional.
            Length of the biggest sequence. Default is 15.
        idx_br : int, optional
            Position of the amino acid that bounds branching to CP. The default is None.

        Returns
        -------
        new_seq : list
            List of the fragments that form the peptide

        '''
        _, lst_seq = self.encode([seq], length = length, stop_signal = False)
        check_term = any(aa in self.term_aas for aa in lst_seq)
        if not check_term:
            return lst_seq
        else:
            for term_aa in self.term_aas:
                if term_aa not in lst_seq:
                    continue
                if term_aa != '-pip':
                    new_seqs = []
                    branch_block = [''.join(lst_seq[:idx_br+1])] # Make whole the branched block
                    new_seq = branch_block + lst_seq[idx_br+1:] # Create new sequence with branched block
                    return new_seq
                else:
                    new_seqs = []
                    branch_block = [''.join(lst_seq[idx_br:])] # Make whole the branched block
                    new_seq = lst_seq[:idx_br] + branch_block # Create new sequence with branched block
                    new_seqs.append(new_seq)
# =============================================================================
#                             There aren't any other aas that can act as bridges in the sequence so we can do it this way
# =============================================================================
                    return new_seq
        # return permutations
        
    def check_cyclization(self, seq, idx_br : int = None, length : int = 15):
        '''
        Fragment sequence into amino acids. If branching is detected, the aas
        forming the branching are put together.
    
        Parameters
        ----------
        seq : str
            Peptide sequence we want to fragment.
        length : int. Optional.
            Length of the biggest sequence. Default is 15.
        idx_br : int, optional
            Position of the amino acid that bounds branching to CP. The default is None.

        Returns
        -------
        new_seq : list
            List of the fragments that form the peptide

        '''
        _, lst_seq = self.encode([seq], length = length, stop_signal = False)
        check_term = any(aa in self.term_aas for aa in lst_seq)
        if not check_term:
            return ''
        else:
            # Case where branching exists
            idx_br = idx_br - 1 # Equals to python numbering
            if lst_seq[idx_br] not in self.bridge_aas:
                # Branching exist and the index is incorrect
                fail_str1 = ' Caution the output is not a CP because'
                fail_str2 = f' the amino acid {lst_seq[idx_br]} at position {idx_br+1} of sequence {seq}'
                fail_str3 = ' cannot act as branching point. Revise the index of'
                fail_str4 = ' your of branching position'
                return fail_str1 + fail_str2 + fail_str3 + fail_str4
            else:
                return ''
        # return permutations
    
    def cyclic_permutations(self, seq, idx_br : int = None, length : int = 15):
        '''
        Generate all cyclic permutations of a given sequence.
    
        Parameters
        ----------
        seq : str
            Peptide sequence we want to permutate.
        idx_br : int, optional
            Position of the amino acid that bounds branching to CP. The default is None.
        length : int. Optional.
            Length of the biggest sequence. Default is 15.
    
        Returns
        -------
        permutations : list
            List of the possible permutations of the sequence
    
        '''
        permutations = [] # Initialize container for permutartions
        _, lst_seq = self.encode([seq], length = length, stop_signal = False) # Divide seq into aas
        check_term = any(aa in self.term_aas for aa in lst_seq) # Check if we have any terminal aa that leads to branching
        if not check_term:
            # If we dont have branching
            for i in range(len(lst_seq)):
                permutations.append(''.join(lst_seq[i:] + lst_seq[:i])) # Creates permutations
        else:
            # If wwe have branching
            for term_aa in self.term_aas:
                if term_aa not in lst_seq:
                    continue
                if term_aa != '-pip':
                    new_seqs = []
                    branch_block = [''.join(lst_seq[:idx_br+1])] # Make whole the branched block
                    new_seq = branch_block + lst_seq[idx_br+1:] # Create new sequence with branched block
                    permutations.append(''.join(new_seq))
                else:
                    new_seqs = []
                    branch_block = [''.join(lst_seq[idx_br:])] # Make whole the branched block
                    new_seq = lst_seq[:idx_br] + branch_block # Create new sequence with branched block
                    new_seqs.append(new_seq)
# =============================================================================
#                   There aren't any other aas that can act as bridges in the sequence so we can do it this way
# =============================================================================
                    permutations.append(''.join(new_seq))
        return permutations
    
    def calculate_metric(self, permutation, amino_acid_values):
        '''
        Calculate a simple metric for a given permutation based on amino acid values.
    
        Parameters
        ----------
        permutation : list
            Output of cyclic_permutation function
        amino_acid_values : dict
            Dictionary that contains each amino acid ad its corresponding value
    
        Returns
        -------
        float
            Computed metric of the sequence based on the collocation of the amino acids
    
        '''
        _, permutation = self.encode(sequences = [permutation], length = 15)
        return sum(amino_acid_values[aa] * np.exp(i + 1) for i, aa in enumerate(permutation))
    
    def generate_permutations_and_metrics(self, sequence, idx_br : int = None, length : int = 15):
        '''
        Generate all cyclic permutations of a sequence and their respective metrics.
        This function uses the previous two to generate all the permutations and 
        calculate their metrics.
    
        Parameters
        ----------
        sequence : list
            list containing the amino acids that form a peptide sequence
            
        amino_acid_values : dict
            Dictionary with the amino acids and their corresponding value for the metric
            
        length : int. Optional.
            Length of the biggest sequence. Default is 15.
    
        Returns
        -------
        list
            Returns a list with the permutation that yields the lower metric
    
        '''
        amino_acid_values = self.amino_acid_values
        permutations = self.cyclic_permutations(sequence, idx_br)
        metrics = [self.calculate_metric(perm, amino_acid_values) for perm in permutations]
        # return the permutation with lowest metric
        return   permutations[metrics.index(min(metrics))]
    
    def arr_add0(self, arrs : np.array, shape : tuple = (15,15,208)):
        '''
        Adds 0s to array until complete desired shape

        Parameters
        ----------
        arrs : np.array
            DESCRIPTION.
        shape : tuple, optional
            DESCRIPTION. The default is (15,15,208).

        Returns
        -------
        arrs : TYPE
            DESCRIPTION.

        '''
        arr_left = shape[0] - len(arrs) # Get number of rows to add to existing array
        arr_left = np.zeros((arr_left, shape[1], shape[2])) # Create array of 0s to complete desired shape
        arrs = np.concatenate((arrs, arr_left), axis = 0) # Concatenate array of 0s to original array
        return arrs
    
    def encode_permutations(self, sequences : np.array, idx_br : int = None, shape : tuple = (15,15,208)):
        '''
        Function that encodes all possible permutations of a secuence. Each sequence
        will be encoded into an array of shape = shape where the first dimension corresponds
        to each subsequence, the second to the biggest length of a sequence within the DB
        and the third to the number of features. The array returned contains the
        encoding of all the permutations of the sequences provided so it will have shape
        (len(sequences) + shape)

        Parameters
        ----------
        sequences : np.array
            Array of strings representing cyclic peptides.
        shape : tuple, optional
            Shape for the array containing permutations. The default is (15,15,208).

        Returns
        -------
        final_array : np.array
            Array that contains the encoding of all the permutations, each entry of
            the first dimension corresponds to a sequence and contains its permutations.

        '''
        final_array = np.zeros((len(sequences), shape[0], shape[1], shape[2])) # Creates final container for encoded permutations
        for idx, sequence in enumerate(sequences):
            permutations = self.cyclic_permutations(sequence, idx_br) # Get permutations of sequence
            encoded_perms, _ = self.encode(permutations, length=shape[1], stop_signal=False) # Get encoded permutations
            encoded_perms = self.arr_add0(encoded_perms, shape) # Add 0s to complete empty permutations
            final_array[idx] = encoded_perms
        return final_array
        

