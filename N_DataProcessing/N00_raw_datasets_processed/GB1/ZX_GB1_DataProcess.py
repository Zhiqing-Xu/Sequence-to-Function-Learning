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
import glob
import pandas as pd

"""
Process classification/regression CSVs into train/test splits for downstream use.

Input CSVs (expected in current working directory):
    - low_vs_high.csv
    - one_vs_rest.csv
    - two_vs_rest.csv
    - three_vs_rest.csv

Each input CSV must contain the following columns:
    sequence,target,set,validation
        sequence : protein sequence string
        target   : numeric value
        set      : "train" or "test"
        validation : ignored

Outputs (written alongside inputs):
    <basename>_train.csv
    <basename>_test.csv

Each output CSV has columns:
    "", SEQ, quantitative_function
The first column is a fresh 0-based index (not from the source file).
"""


# --------------------------------------------------------------------------- #
# Explicitly enumerate expected input files (safer than globbing everything).
EXPECTED_FILES = [
    "low_vs_high.csv",
    "one_vs_rest.csv",
    "two_vs_rest.csv",
    "three_vs_rest.csv",
]


# --------------------------------------------------------------------------- #
def process_file(path: str) -> None:
    """Read a source CSV, split by 'set', and write *_train.csv and *_test.csv."""
    if not os.path.isfile(path):
        print(f"[WARN] Skipping missing file: {path}")
        return

    # Read
    df = pd.read_csv(path)

    # Basic column sanity check (lightweight; asserts required columns exist)
    required = {"sequence", "target", "set", "validation"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    # (Optional) coerce target numeric; keeps NaNs if present
    df["target"] = pd.to_numeric(df["target"], errors="coerce")

    # Standardize set labels just in case (lowercase strip)
    df["set"] = df["set"].astype(str).str.strip().str.lower()

    # Split
    df_train = df[df["set"] == "train"].copy()
    df_test = df[df["set"] == "test"].copy()

    # Reformat -> output schema
    def reformat(sub_df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame({
            "SEQ": sub_df["sequence"].astype(str).values,
            "quantitative_function": sub_df["target"].values,
        })
        # Ensure new fresh index (0..n-1)
        out.reset_index(drop=True, inplace=True)
        return out

    out_train = reformat(df_train)
    out_test = reformat(df_test)

    # Derive output paths
    stem, ext = os.path.splitext(path)
    out_train_path = f"{stem}_train{ext}"
    out_test_path = f"{stem}_test{ext}"

    # Write with blank index label for the first column header
    out_train.to_csv(out_train_path, index=True, index_label="")
    out_test.to_csv(out_test_path, index=True, index_label="")

    print(f"[OK] Wrote {out_train_path} ({len(out_train)} rows) "
          f"and {out_test_path} ({len(out_test)} rows).")



# --------------------------------------------------------------------------- #
def main():
    for fname in EXPECTED_FILES:
        process_file(fname)


if __name__ == "__main__":
    main()


























































































