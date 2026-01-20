# CYCLOPS

[CYCLOPS](https://cyclopep.com/builder) is a web based tool for the prediction of Cyclic Peptide membrane permeability. It has been trained on version 1.1 of [CycPeptMPDB](http://cycpeptmpdb.com). In this repo, you will find the code embedded in the platform.
For more detail on this work, we encourage you to read our [paper](https://pubs.rsc.org/en/content/articlelanding/2025/dd/d4dd00375f).

---

## Getting Started

To use the code provided in this repository, you need to create a Python environment and install the required libraries. A `.yml` file is provided for this:

```bash
python3 -m venv cyclops
source cyclops/bin/activate
pip install -r environment.txt
```

---

## Run Predictions

This repository contains code for predicting MD results of **Cyclic Peptides**. The main script, **`ML_web.py`**, takes as inputs a sequence of amino acids, the cyclization type (either head-tail of head-side), and the branching residue (either 0 if head-tail cyclization or the 1-based index of the amino acid where the branching begins).

```bash
python ML_web.py -s LmeALdPmeFLmeATdPmeF -t 'head-tail'
The sequence LmeALdPmeFLmeATdPmeF presents Low Permeability with a probability of 77% The predicted Permeability for sequence LmeALdPmeFLmeATdPmeF is: -4.898 ± 0.407

```
As alternative the user can provide a file like Seqs.txt through the `-f` option:

```bash
python ML_web.py -f Seqs.txt

The sequence PmeAMe_dLRdPmeLF presents High Permeability with a probability of 97% The predicted Permeability for sequence PmeAMe_dLRdPmeLF is: -6.346 ± 0.407
The sequence PALLFMe_dLF presents Low Permeability with a probability of 92% The predicted Permeability for sequence PALLFMe_dLF is: -5.587 ± 0.407
The sequence AALmeVLWWPITGD-pip presents High Permeability with a probability of 63% The predicted Permeability for sequence AALmeVLWWPITGD-pip is: -7.825 ± 0.407
```

---
