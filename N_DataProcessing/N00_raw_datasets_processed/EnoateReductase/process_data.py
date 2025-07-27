#!/usr/bin/env python
# coding: utf-8
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Chem.Fingerprints import FingerprintMols
#!/usr/bin/env python
# coding: utf-8
#====================================================================================================#
# The following code ensures the code work properly in 
# MS VS, MS VS CODE and jupyter notebook on both Linux and Windows.
import os 
import sys
from os import path
from sys import platform
from pathlib import Path

if __name__ == "__main__":
    print("\n\n")
    print("="*80)
    if os.name == 'nt' or platform == 'win32':
        print("Running on Windows")
        if 'ptvsd' in sys.modules:
            print("Running in Visual Studio")

    if os.name != 'nt' and platform != 'win32':
        print("Not Running on Windows")

    if "__file__" in globals().keys():
        print('CurrentDir: ', os.getcwd())
        try:
            os.chdir(os.path.dirname(__file__))
        except:
            print("Problems with navigating to the file dir.")
        print('CurrentDir: ', os.getcwd())
    else:
        print("Running in python jupyter notebook.")
        try:
            if not 'workbookDir' in globals():
                workbookDir = os.getcwd()
                print('workbookDir: ' + workbookDir)
                os.chdir(workbookDir)
        except:
            print("Problems with navigating to the workbook dir.")
#====================================================================================================#
# Imports
import pandas as pd

# Load the wildtype sequence
with open('wildtype.txt', 'r') as file:
    wildtype_seq = file.read().strip()

# Load the CSV file containing the variants and yields
df = pd.read_csv('productivity_ER_BC_variants.csv')

# Function to apply mutations to the wildtype sequence
def apply_mutations(wildtype_seq, mutations):
    # Skip entries labeled as "WT" or "Blank"
    if mutations in ["WT", "Blank"]:
        return wildtype_seq if mutations == "WT" else None
    
    seq_list = list(wildtype_seq)
    for mutation in mutations.split('-'):
        if len(mutation) < 3:  # Ensure the mutation string is valid
            raise ValueError(f"Invalid mutation format: {mutation}")
        
        try:
            original_aa = mutation[0]
            position = int(mutation[1:-1]) - 1  # Convert to zero-based index
            new_aa = mutation[-1]
        except ValueError as e:
            raise ValueError(f"Error parsing mutation '{mutation}': {e}")
        
        # Ensure the position is within the sequence length
        if position < 0 or position >= len(seq_list):
            raise ValueError(f"Position {position + 1} out of range for sequence length {len(seq_list)}")
        
        # Apply the mutation
        seq_list[position] = new_aa
    
    return ''.join(seq_list)

# Apply the mutations to the wildtype sequence
df['SEQ'] = df['Variants'].apply(lambda x: apply_mutations(wildtype_seq, x))

# Filter out any rows where 'SEQ' is None (from "Blank" or invalid entries)
df = df[df['SEQ'].notnull()]

# Rename the yield column
df = df.rename(columns={'Adipic acid (ug/mL)': 'yield'})

# Add an index column starting from 1
df.reset_index(drop=True, inplace=True)
df.index += 1
df.index.name = 'Index'

# Select the required columns and save to a new CSV file with index
df[['SEQ', 'yield']].to_csv('ERBC.csv', index=True)
