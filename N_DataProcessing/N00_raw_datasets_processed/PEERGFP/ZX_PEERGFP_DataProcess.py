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
import os
import re
import pandas as pd

import pandas as pd
import os


# List of CSV filenames
filenames = ["fluorescence_train.csv", "fluorescence_test.csv", "fluorescence_valid.csv"]

# Read all CSV files and concatenate them into a single DataFrame
df_list = [pd.read_csv(fname, index_col=0) for fname in filenames]
df_all = pd.concat(df_list, ignore_index=True)

# Split the DataFrame based on num_mutations
df_lt7 = df_all[df_all["num_mutations"] < 7]
df_ge7 = df_all[df_all["num_mutations"] >= 7]

# Save to CSV files
df_lt7.to_csv("GFP_train.csv", index=False)
df_ge7.to_csv("GFP_test.csv", index=False)




































































