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

"""
Parse a FASTA file whose headers embed target values and a train/test split,
then emit two CSVs suitable for downstream analyses.

Input FASTA (expected in current working directory):
    betalactamase.fasta
Header format (one example):
    >Sequence0 TARGET=0.9426838159561157 SET=train VALIDATION=False
                 ^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^
                   quantitative target      split

Outputs (written alongside the FASTA):
    betalactamase_train.csv
    betalactamase_test.csv

Each output CSV has columns:
    "", SEQ, quantitative_function
The first column is a fresh 0‑based index (not derived from FASTA order).
"""

FASTA_FILE = "betalactamase.fasta"

# Regex pre‑compiled for performance and clarity
HDR_RE = re.compile(
    r"""
    ^>.*?                               # leading '>' plus any chars up to first space (SequenceID)
    \s+TARGET=(?P<target>[0-9.+eE-]+)   # numeric TARGET
    \s+SET=(?P<set>train|test)          # train / test label
    """,
    re.VERBOSE | re.IGNORECASE,
)


def parse_fasta(path: str):
    """Yield (sequence, target_value, set_label) tuples from a custom FASTA."""
    with open(path, "r", encoding="utf-8") as fh:
        seq_lines = []
        meta = None  # (target, set)
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                # Flush previous record if any
                if meta is not None:
                    yield "".join(seq_lines), meta[0], meta[1]
                    seq_lines.clear()

                # Extract TARGET and SET from header
                m = HDR_RE.search(line)
                if not m:
                    raise ValueError(f"Header malformed or missing fields:\n{line}")
                target_val = float(m.group("target"))
                set_label = m.group("set").lower()

                meta = (target_val, set_label)
            else:
                seq_lines.append(line.strip())

        # Emit final record
        if meta is not None:
            yield "".join(seq_lines), meta[0], meta[1]


def main():
    if not os.path.isfile(FASTA_FILE):
        raise FileNotFoundError(f"Expected FASTA file not found: {FASTA_FILE}")

    # Collect all entries into lists
    seqs, targets, sets = [], [], []
    for seq, tgt, st in parse_fasta(FASTA_FILE):
        seqs.append(seq)
        targets.append(tgt)
        sets.append(st)

    # Build master DataFrame then split
    df = pd.DataFrame(
        {
            "SEQ": seqs,
            "quantitative_function": targets,
            "set": sets,
        }
    )

    df_train = df[df["set"] == "train"].drop(columns="set").reset_index(drop=True)
    df_test = df[df["set"] == "test"].drop(columns="set").reset_index(drop=True)

    # Derive output paths
    stem, ext = os.path.splitext(FASTA_FILE)
    out_train = f"{stem}_train.csv"
    out_test  = f"{stem}_test.csv"

    # Write CSVs with blank index header
    df_train.to_csv(out_train, index=True, index_label="")
    df_test.to_csv(out_test, index=True, index_label="")

    print(
        f"[OK] Wrote {out_train} ({len(df_train)} rows) and "
        f"{out_test} ({len(df_test)} rows)."
    )


if __name__ == "__main__":
    main()


















































































