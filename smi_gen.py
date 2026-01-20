#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 09:11:24 2024

@author: alfonsocabezon.vizoso@usc.es

Combine_fragments and make_peptide functions were adapted from: 
https://gist.github.com/iwatobipen/1ffa9b2a13b70cf3f424e7dd9af68a90#file-make_peptide-ipynb

Comments will be added to make it more clear.
The code includes functions that take a sequence of amino acids and generate their 
physicochemical properties and other to generate the SMILE. Graph generation will be added 
to generate graphs from the molecules
"""

from rdkit import Chem
from rdkit.Chem import molzip
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
import pandas as pd
import numpy as np
import copy
from UP import CyclicPeptide

class smi_gen(CyclicPeptide):
    def __init__(self):
        super().__init__() # inherits all methods from parent class
        # self.data_aas = pd.read_csv("CycPeptMPDB_Monomer_All_FixSMI.csv", index_col = [0]) # Load monomers of database
        self.data_aas = pd.read_csv("CycPeptMPDB_Monomer_All.csv", index_col = [0]) # Load monomers of database
        self.prop_names = list(self.data_aas.columns)[19:]
        self.symbols = self.data_aas['Symbol'].to_numpy()
        aa_CXSMILE = np.array(self.data_aas['CXSMILES'].to_list()) # Get list of CXSMILE
        aa_R3s = self.data_aas['R3'].to_numpy()
        self.amino_acids = [] # Store Mol objects of amino acids
        for idx, aa in enumerate(aa_CXSMILE):
            R3 = aa_R3s[idx]
            mol = Chem.MolFromSmiles(aa) # Create mol from smile
            
            for atom in mol.GetAtoms():
                if atom.HasProp('atomLabel') and atom.GetProp('atomLabel') == '_R3' and R3 == 'OH':
                    atom.SetProp('atomLabel', '_R4')
                elif atom.HasProp('atomLabel') and atom.GetProp('atomLabel') == '_R3' and R3 == 'H':
                    atom.SetProp('atomLabel', '_R5')
            mol.SetProp('Type', 'Backbone') # Set type of mol as backbone
            self.data_aas.loc[self.data_aas['Symbol'] == self.symbols[idx], 'CXSMILES'] = Chem.MolToCXSmiles(mol)
        
    def combine_fragments(self, m1, m2):
        '''
        This function combines fragments into one single Mol object
    
        Parameters
        ----------
        m1 : rdkit.Chem.rdchem.Mol object
            Mol object of desired aa
        m2 : rdkit.Chem.rdchem.Mol object
            Second amino acid of the sequence
    
        Returns
        -------
        rdkit.Chem.rdchem.Mol object
            Object of both amino acids together
    
        '''
        m1 = Chem.Mol(m1) #Get molecule from fragment
        m2 = Chem.Mol(m2) # Get molecule of second fragment
        '''We need to eliminate the R1 and R2 labels of the molecule. R2 goes to 
        carboxyl end so we will eliminate it from amino acid 1. R1 goes to amino end
        so we will eliminate it from the second amino acid. We will use molzip to
        wrap both molecules together. To do so we have to label the terminal ends 
        with the same number and molzip will combine the fragments connecting them
        on that label'''
        for atm in m1.GetAtoms():
            if atm.HasProp('atomLabel') and atm.GetProp('atomLabel') == '_R2':
                atm.SetAtomMapNum(1) # Where _R2 is found, a 1 is set
        for atm in m2.GetAtoms():
            if atm.HasProp('atomLabel') and atm.GetProp('atomLabel') == '_R1':
                atm.SetAtomMapNum(1) # Where R1 is found a 1 is set
        # Combine and return
        return molzip(m1, m2)
    
    def make_peptide(self, monomerlist : list):
        '''
        This function will take a list of monomers and put them together in a single
        object.
    
        Parameters
        ----------
        monomerlist : list of rdkit.Chem.rdchem.Mol object
            List containing the monomers for the peptide.
        idx_term : int
            Index of the amino acid that closes the branching.

        Returns
        -------
        res : rdkit.Chem.rdchem.Mol
            Object containing the peptide.

        '''
        monomerlist = copy.deepcopy(monomerlist)
        for idx, monomer in enumerate(monomerlist):
            if idx == 0:
                for atm in monomer.GetAtoms():
                    if atm.HasProp('atomLabel') and atm.GetProp('atomLabel') == '_R1':
                        atm.SetAtomMapNum(3) # Where _R1 is found, a 3 is set
                res = monomer
            else:
                if idx == len(monomerlist)-1:
                    for atm in monomer.GetAtoms():
                        if atm.HasProp('atomLabel') and atm.GetProp('atomLabel') == '_R2':
                            atm.SetAtomMapNum(3) # Where _R2 is found, a 3 is set¡
                res = self.combine_fragments(res, monomer) # Combines fragment with next monomer
        return res
    
    def make_subseq(self, monomerlist : list, idx_br : int = None):
        '''
        This function will take a list of monomers and put them together in a single
        object.
    
        Parameters
        ----------
        monomerlist : list of rdkit.Chem.rdchem.Mol object
            List containing the monomers for the peptide.
        idx_term : int
            Index of the amino acid that closes the branching.

        Returns
        -------
        res : rdkit.Chem.rdchem.Mol
            Object containing the peptide.

        '''
        monomerlist = copy.deepcopy(monomerlist)
        for idx, monomer in enumerate(monomerlist):
            if idx == 0 and idx != idx_br:
                for atm in monomer.GetAtoms():
                    if atm.HasProp('atomLabel') and atm.GetProp('atomLabel') == '_R1':
                        atm.SetAtomMapNum(2) # Where _R1 is found (amino end), a 3 is set
                res = monomer
            elif idx == 0 and idx == idx_br:
                for atm in monomer.GetAtoms():
                    if atm.HasProp('atomLabel') and atm.GetProp('atomLabel') == '_R1':
                        atm.SetAtomMapNum(3) # Where _R1 is found (amino end), a 3 is set
                    if atm.HasProp('atomLabel') and (atm.GetProp('atomLabel') == '_R4' or atm.GetProp('atomLabel') == '_R5'):
                        atm.SetAtomMapNum(2) # If _R4 or _R5 (side chain)a 3 is set. This will close head to side chain
                res = monomer
            else:
                if idx == len(monomerlist)-1 and idx != idx_br:
                    for atm in monomer.GetAtoms():
                        if atm.HasProp('atomLabel') and atm.GetProp('atomLabel') == '_R2':
                            atm.SetAtomMapNum(3) # Where _R2 is found (Carbo end), a 3 is set¡
                elif idx == len(monomerlist)-1 and idx == idx_br:
                    for atm in monomer.GetAtoms():
                        if atm.HasProp('atomLabel') and (atm.GetProp('atomLabel') == '_R4' or atm.GetProp('atomLabel') == '_R5'):
                            atm.SetAtomMapNum(3) # If _R4 or _R5 (side chain)a 3 is set. This will close head to side chain
                        if atm.HasProp('atomLabel') and atm.GetProp('atomLabel') == '_R2':
                            atm.SetAtomMapNum(2) # Where _R2 is found (Carbo end), a 3 is set¡
                res = self.combine_fragments(res, monomer) # Combines fragment with next monomer
        return res
    
    def cap_smi(self, smi : str):
        '''
        This function replace R groups in smiles by corresponding functional group

        Parameters
        ----------
        smi : str
            CXSMILE string of molecule.

        Returns
        -------
        str
            Capped SMILES of molecule.

        '''
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.RWMol(mol) # Allow modifications on mol object
        for atm in mol.GetAtoms():
            if atm.HasProp('atomLabel') and (atm.GetProp('atomLabel') == '_R2' or atm.GetProp('atomLabel') == '_R4'):
                mol.ReplaceAtom(atm.GetIdx(), Chem.Atom(8), updateLabel = True, preserveProps = False)
            elif atm.HasProp('atomLabel') and (atm.GetProp('atomLabel') == '_R1' or atm.GetProp('atomLabel') == '_R5'):
                # rem_idx.append(atm.GetIdx())
                mol.ReplaceAtom(atm.GetIdx(), Chem.Atom(1), updateLabel = True, preserveProps = False)
        # for idx in rem_idx:
        #     mol.RemoveAtom(idx)
        return Chem.MolToSmiles(mol)
    
    
    def Seq2Smile(self, sequence : str, idx_br : int = None, cyclization : str = 'head-tail'):
        '''
        This function takes a peptide sequence and returns its corresponding SMILE string

        Parameters
        ----------
        sequence : str
            String with the amino acids that form the CP.
        idx_br : int, optional
            Index within sequence of the amino acid that binds branching to the CP. The default is None.
        cyclization : str, optional
            How CP is closed. OPTIONS: 'head-tail' or ''head-side. The default is 'head-tail'.

        Returns
        -------
        CP : str
            SMILE of the Cyclic Peptide.

        '''
        if cyclization == 'head-tail':
            lst_seq = self.fragment_peptide(sequence, idx_br)
            # If there is no branching in the molecule
            mol_seq = []
            for aa in lst_seq:
                smi = self.data_aas.loc[self.data_aas['Symbol'] == aa, 'CXSMILES'].iloc[0] # Find aa in df and get CXSMILE that corresponds
                mol = Chem.MolFromSmiles(smi)
                mol_seq.append(mol)
            CP = self.make_peptide(mol_seq) # Make whole peptide
        else:
            # Case with branching
            idx_br = idx_br - 1 # Reduce index to 1 to match python numbering
            lst_seq = self.fragment_peptide(sequence, idx_br) # Get fragmented sequence
            submol_bb = [] # initialize list to store not branchend subsequence
            for aa in lst_seq:
                if aa in self.symbols:
                    smi = self.data_aas.loc[self.data_aas['Symbol'] == aa, 'CXSMILES'].iloc[0] # Find aa in df and get CXSMILE that corresponds
                    mol = Chem.MolFromSmiles(smi) # Convert SMILE to mol object
                    submol_bb.append(mol) # Append mol to subsequence list
                else:
                    # If there is a branching, we will create its mol object separately
                    _, lst_subseq = self.encode([aa], length = 15) # Get aa that form subsequence
                    submol_br = [] # Initialize list to store mol objects
                    for aa in lst_subseq:
                        smi = self.data_aas.loc[self.data_aas['Symbol'] == aa, 'CXSMILES'].iloc[0] # Find aa in df and get CXSMILE that corresponds
                        # print(aa, smi)
                        mol = Chem.MolFromSmiles(smi) # Get mol from smile
                        submol_br.append(mol) # Append mol object of aa to subsequence
                    idx_br = idx_br - len(submol_bb) # When we couple the branching fragment, we lose the indexing
                    # when the sequence is branched in the terminal end. By doing this we reset idx_br to 0 in those
                    # cases allowing the code to detect the branching. idx_br will be unchanged otherwise since len(seq) = 0
                    submol_br = self.make_subseq(submol_br, idx_br) # Creates mol object of subsequence
            submol_bb = self.make_subseq(submol_bb) # Unifies aas of backbone subsequence into one mol object
            CP = molzip(submol_bb, submol_br) # Reaction between subsequences to create final CP
        CP = self.cap_smi(Chem.MolToCXSmiles(CP)) # Convert mol object of whole CP to smile,
        # cap it substituting R groups by corresponding atom 
        return CP
    
    def Seq2Prop(self, sequence : str, prop_names = 'default', 
                 idx_br : int = None, cyclization : str = 'head-tail'):
        '''
        This function takes a peptide sequence and returns the properties computed
        with rdkit

        Parameters
        ----------
        sequence : str
            String with the amino acids that for the CP.
        prop_names : TYPE, optional
            List with property names or use pre-defined. Options: default,
            stored or give a list. The default is 'default'.
        idx_br : int, optional
            Index within sequence of the amino acid that binds branching to the CP.. The default is None.
        cyclization : str, optional
            How CP is closed. OPTIONS: 'head-tail' or 'head-side'. The default is 'head-tail'.

        Returns
        -------
        prop : np.array.
            Array of descriptors computed by rdkit.

        '''
        if prop_names == 'default':
            prop_names = self.prop_names
        elif prop_names == 'stored':
            import pickle
            with open('prop_names.pkl', 'rb') as f:
                prop_names = pickle.load(f)
        else:
            pass
        CP = self.Seq2Smile(sequence, idx_br, cyclization) # Get SMILE of CP
        CP = Chem.MolFromSmiles(CP) # Convert SMILE to mol object
        Chem.SanitizeMol(CP) # Needed to be able to generate some properties
        prop = Descriptors.CalcMolDescriptors(CP, silent = True) # Get descriptors
        final_feat = [] # Initialize container
        for name in prop_names:
            final_feat.append(prop[name]) # append feature to final list
        prop = np.array(final_feat)
        return prop

