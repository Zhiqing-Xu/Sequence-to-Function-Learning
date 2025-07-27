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
import re
import sys
import time
import copy
import math
import scipy
import torch
import pickle
import random
import argparse
import subprocess
import numpy as np
import pandas as pd
#--------------------------------------------------#
from torch import nn
from torch.utils import data
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F
#from torchvision import models
#from torchsummary import summary
#--------------------------------------------------#
from tape import datasets
from tape import TAPETokenizer
#--------------------------------------------------#
from sklearn import datasets
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
#--------------------------------------------------#
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import auc
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
#--------------------------------------------------#
#from sklearn import svm
#from sklearn.model_selection import GridSearchCV
#from sklearn.tree import DecisionTreeRegressor
#--------------------------------------------------#
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
#--------------------------------------------------#
from pathlib import Path
from copy import deepcopy
from tpot import TPOTRegressor
from ipywidgets import IntProgress
#--------------------------------------------------#
from datetime import datetime
#--------------------------------------------------#
from ZX01_PLOT import *
from ZX02_nn_utils import StandardScaler, normalize_targets


seed=0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#                    `7MMF'`7MN.   `7MF'`7MM"""Mq. `7MMF'   `7MF'MMP""MM""YMM  .M"""bgd                                                                #
#                      MM    MMN.    M    MM   `MM.  MM       M  P'   MM   `7 ,MI    "Y                                                                #
#   ,pP""Yq.           MM    M YMb   M    MM   ,M9   MM       M       MM      `MMb.                                                                    #
#  6W'    `Wb          MM    M  `MN. M    MMmmdM9    MM       M       MM        `YMMNq.                                                                #
#  8M      M8          MM    M   `MM.M    MM         MM       M       MM      .     `MM                                                                #
#  YA.    ,A9 ,,       MM    M     YMM    MM         YM.     ,M       MM      Mb     dM                                                                #
#   `Ybmmd9'  db     .JMML..JML.    YM  .JMML.        `bmmmmd"'     .JMML.    P"Ybmmd"                                                                 #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$# 

## Args
Step_code    = "N05X_"

Model_code   = ["N05A_", "N05B_", "N05C_", "N04B_", ][0] # N05A_: Conv1D | N05B_: self-Attention | N04B_: Avg |

# Train Options: 
# TVT: Split a validation set and a test set from the training set, then train and validate and test.
# TT : Train and Test on different pre-split datasets.
# TR : Train Only
# CV : Cross Validation
train_option = ["TVT", "TT", "TR", "CV"][1]

#--------------------------------------------------#
dataset_nme_list     = ["NovoEnzyme"        ,         # 0
                        "PafAVariants"      ,         # 1
                        "GFP"               ,         # 2
                        "Rubisco"           ,         # 3
                        "ERBC"              ,         # 4

                        "GB1_LoVSHi_train"  ,         # 5
                        "GB1_LoVSHi_test"   ,         # 6
                        "GB1_1VSrest_train" ,         # 7
                        "GB1_1VSrest_test"  ,         # 8
                        "GB1_2VSrest_train" ,         # 9
                        "GB1_2VSrest_test"  ,         # 10
                        "GB1_3VSrest_train" ,         # 11
                        "GB1_3VSrest_test"  ,         # 12

                        "beta_lact_train"   ,         # 13
                        "beta_lact_test"    ,         # 14

                        "PEERGFP_train"     ,         # 15
                        "PEERGFP_test"      ,         # 16
                        "PEERGFP_trainonly" ,         # 17

                        ]


dataset_nme           = "dataset_nme"
data_train_nme        = dataset_nme_list[17]
data_valid_nme        = None
data_test_nme         = dataset_nme_list[16]

data_folder = Path("N_DataProcessing/")

embedding_file_list = [ "N03_" + dataset_nme + "_embedding_ESM_1B.p"      ,      # 0
                        "N03_" + dataset_nme + "_embedding_ESM_1V.p"      ,      # 1
                        "N03_" + dataset_nme + "_embedding_ESM_2_650.p"   ,      # 2
                        "N03_" + dataset_nme + "_embedding_ESM_2_3B.p"    ,      # 3

                        "N03_" + dataset_nme + "_embedding_BERT.p"        ,      # 4
                        "N03_" + dataset_nme + "_embedding_TAPE.p"        ,      # 5
                        "N03_" + dataset_nme + "_embedding_ALBERT.p"      ,      # 6
                        "N03_" + dataset_nme + "_embedding_T5.p"          ,      # 7
                        "N03_" + dataset_nme + "_embedding_T5XL.p"        ,      # 8
                        "N03_" + dataset_nme + "_embedding_Xlnet.p"       ,      # 9

                        "N03_" + dataset_nme + "_embedding_Ankh_Large.p"  ,      # 10
                        "N03_" + dataset_nme + "_embedding_Ankh_Base.p"   ,      # 11
			            "N03_" + dataset_nme + "_embedding_CARP_640M.p"   ,      # 12
                        "N03_" + dataset_nme + "_embedding_Unirep.p"      ,      # 13
                        ]

embedding_file        = embedding_file_list[0]                                                             # <<< Select pLM here.
embedding_file_train  = embedding_file.replace(dataset_nme, data_train_nme) if data_train_nme is not None else None
embedding_file_valid  = embedding_file.replace(dataset_nme, data_valid_nme) if data_valid_nme is not None else None
embedding_file_test   = embedding_file.replace(dataset_nme, data_test_nme ) if data_test_nme  is not None else None
embedding_file        = embedding_file_train

properties_file        = "N00_" + dataset_nme + "_seqs_prpty_list.p"
properties_file_train  = properties_file.replace(dataset_nme, data_train_nme) if data_train_nme is not None else None
properties_file_valid  = properties_file.replace(dataset_nme, data_valid_nme) if data_valid_nme is not None else None
properties_file_test   = properties_file.replace(dataset_nme, data_test_nme ) if data_test_nme  is not None else None
properties_file        = properties_file_train

seqs_fasta_file        = "N00_" + dataset_nme + ".fasta"
seqs_fasta_file_train  = seqs_fasta_file.replace(dataset_nme, data_train_nme) if data_train_nme is not None else None
seqs_fasta_file_valid  = seqs_fasta_file.replace(dataset_nme, data_valid_nme) if data_valid_nme is not None else None
seqs_fasta_file_test   = seqs_fasta_file.replace(dataset_nme, data_test_nme ) if data_test_nme  is not None else None
seqs_fasta_file        = seqs_fasta_file_train

dataset_nme            = data_train_nme

# The embedding_file is a dict , {"seq_embeddings"  : seq_embeddings  , 
#                                 "seq_ids"         : seq_ids         , 
#                                 "seq_all_hiddens" : seq_all_hiddens , 
#                                 "sequences"       : sequences       , }

# The properties_file is a dict, {"" : , "" : , }

#====================================================================================================#
# Select properties (Y) of the model 
prpty_list = [
              ["tm"],                      # NovoEnzyme

              ["kcat_cMUP"               , # 0
               "KM_cMUP"                 , # 1
               "kcatOverKM_cMUP"         , # 2
               "kcatOverKM_MeP"          , # 3
               "kcatOverKM_MecMUP"       , # 4
               "Ki_Pi"                   , # 5
               "fa"                      , # 6
               "kcatOverKM_MePkchem"     , # 7
               "FC1"                     , # 8
               "FC2_3"                   , # 9
               "FC4"                     , # 10
              ],

              ["quantitative_function"]  , # GFP
              ["Kcatmean"]               , # Rubisco
              ["yield"]                  , # ERBC

              ["quantitative_function"]  , # 5
              ["quantitative_function"]  , # 6
              ["quantitative_function"]  , # 7
              ["quantitative_function"]  , # 8
              ["quantitative_function"]  , # 9
              ["quantitative_function"]  , # 10
              ["quantitative_function"]  , # 11
              ["quantitative_function"]  , # 12

              ["quantitative_function"]  , # 13
              ["quantitative_function"]  , # 14

              ["quantitative_function"]  , # 15
              ["quantitative_function"]  , # 16
              ["quantitative_function"]  , # 17

             ][dataset_nme_list.index(dataset_nme)]


prpty_select = prpty_list[0]

#====================================================================================================#
# Prediction NN settings
NN_type        = ["Reg", "Clf"][0]
epoch_num      = 50
batch_size     = 256
learning_rate  = 5e-5

#====================================================================================================#
# Hyperparameters.
if Model_code == "N05A_":
    '''
    model = SQembCNN_Model(in_dim   = NN_input_dim ,
                           hid_dim  = 1024         ,
                           kernal_1 = 5            ,
                           out_dim  = 2            , 
                           kernal_2 = 3            ,
                           max_len  = seqs_max_len ,
                           last_hid = 2048         , 
                           dropout  = 0.0
                           )
                           '''
    hid_dim    =  320     # 256
    kernal_1   =  3       # 5
    out_dim    =  1       # 2
    kernal_2   =  3       # 3
    last_hid   =  1280     # 1440
    dropout    =  0.0     # 0

elif Model_code == "N05B_":
    '''
    model = SQembSAtt_Model(d_model    = NN_input_dim,
                            d_k        = d_k,
                            n_heads    = n_heads,
                            d_v        = d_v,
                            out_dim    = 1,
                            d_ff       = last_hid
                            )
                            '''
    d_k        =  256     # 256
    n_heads    =  2       # 1  
    out_dim    =  1       # Useless
    d_v        =  2       # 1
    last_hid   =  512     # d_ff
    dropout    =  0.0     # 0.1, 0.6

elif Model_code == "N04B_":
    '''
    model = SQembAvgD_Model( in_dim  = seqs_max_len , 
                             hid_1   = last_hid     , 
                             hid_2   = last_hid     ,
                             dropout = dropout      ,
                             )
                             '''
    hid_1      =  512
    hid_2      =  512
    dropout    =  0.0     # 0.1, 0.6


#====================================================================================================#
# Prepare print outputs.
# hyperparameters_dict = dict([])
# for one_hyperpara in ["hid_dim", "kernal_1", "out_dim", "kernal_2", "last_hid", "dropout"]:
#     hyperparameters_dict[one_hyperpara] = locals()[one_hyperpara]

#====================================================================================================#
# If log_value is True, screen_bool will be changed.
screen_bool = bool(0) # Currently screening y values is NOT supported.
log_value   = bool(0) ##### !!!!! If value is True, screen_bool will be changed
if log_value == True:
    screen_bool = True

#====================================================================================================#
# Results Output
results_folder     = Path("N_DataProcessing/" + Step_code +"intermediate_results/")
output_file_3      = Step_code + Model_code + "_all_X_y.p"
output_file_header = Step_code + Model_code + "_result_"


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#               `7MM"""Mq. `7MM"""Mq.  `7MM"""YMM  `7MM"""Mq.       db      `7MM"""Mq.        db      MMP""MM""YMM `7MMF'  .g8""8q.   `7MN.   `7MF'    #
#   __,           MM   `MM.  MM   `MM.   MM    `7    MM   `MM.     ;MM:       MM   `MM.      ;MM:     P'   MM   `7   MM  .dP'    `YM.   MMN.    M      #
#  `7MM           MM   ,M9   MM   ,M9    MM   d      MM   ,M9     ,V^MM.      MM   ,M9      ,V^MM.         MM        MM  dM'      `MM   M YMb   M      #
#    MM           MMmmdM9    MMmmdM9     MMmmMM      MMmmdM9     ,M  `MM      MMmmdM9      ,M  `MM         MM        MM  MM        MM   M  `MN. M      #
#    MM           MM         MM  YM.     MM   Y  ,   MM          AbmmmqMA     MM  YM.      AbmmmqMA        MM        MM  MM.      ,MP   M   `MM.M      #
#    MM  ,,       MM         MM   `Mb.   MM     ,M   MM         A'     VML    MM   `Mb.   A'     VML       MM        MM  `Mb.    ,dP'   M     YMM      #
#  .JMML.db     .JMML.     .JMML. .JMM..JMMmmmmMMM .JMML.     .AMA.   .AMMA..JMML. .JMM..AMA.   .AMMA.   .JMML.    .JMML.  `"bmmd"'   .JML.    YM      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$# 
# Create Temp Folder for Saving Results
print("="*80)
print("\n\n\n>>> Creating temporary subfolder and clear past empty folders... ")
print("="*80)
now = datetime.now()
#d_t_string = now.strftime("%Y%m%d_%H%M%S")
d_t_string = now.strftime("%m%d-%H%M%S")
#====================================================================================================#
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
results_folder_contents = os.listdir(results_folder)
count_non_empty_folder  = 0
for item in results_folder_contents:
    if os.path.isdir(results_folder / item):
        num_files = len(os.listdir(results_folder/item))
        if num_files in [1,2]:
            try:
                for idx in range(num_files):
                    os.remove(results_folder / item / os.listdir(results_folder/item)[0])
                os.rmdir(results_folder / item)
                print("Remove empty folder " + item + "!")
            except:
                print("Cannot remove empty folder " + item + "!")
        elif num_files == 0:
            try:
                os.rmdir(results_folder / item)
                print("Remove empty folder " + item + "!")
            except:
                print("Cannot remove empty folder " + item + "!")
        else:
            count_non_empty_folder += 1
print("Found " + str(count_non_empty_folder) + " non-empty folders: " + "!")
print("="*80)
#====================================================================================================#
# Get a name for the output. (Need to include all details of the model in the output name.)
embedding_code = embedding_file.replace("N03_" + dataset_nme + "_embedding_", "")
embedding_code = embedding_code.replace(".p", "")
#====================================================================================================#
temp_folder_name = Step_code 
temp_folder_name += Model_code 
temp_folder_name += d_t_string + "_"
temp_folder_name += dataset_nme + "_"
temp_folder_name += prpty_select + "_"
temp_folder_name += embedding_code.replace("_","") + "_"
temp_folder_name += NN_type.upper() + "_"
temp_folder_name += "scrn" + str(screen_bool)[0] + "_"
temp_folder_name += "lg" + str(log_value)[0]
#====================================================================================================#
results_sub_folder=Path("N_DataProcessing/" + Step_code + "intermediate_results/" + temp_folder_name +"/")
if not os.path.exists(results_sub_folder):
    os.makedirs(results_sub_folder)


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#                .g8""8q.   `7MMF'   `7MF'MMP""MM""YMM `7MM"""Mq. `7MMF'   `7MF'MMP""MM""YMM    `7MM"""Mq. `7MM"""Mq.  `7MMF'`7MN.   `7MF'MMP""MM""YMM #
#              .dP'    `YM.   MM       M  P'   MM   `7   MM   `MM.  MM       M  P'   MM   `7      MM   `MM.  MM   `MM.   MM    MMN.    M  P'   MM   `7 #
#  pd*"*b.     dM'      `MM   MM       M       MM        MM   ,M9   MM       M       MM           MM   ,M9   MM   ,M9    MM    M YMb   M       MM      #
# (O)   j8     MM        MM   MM       M       MM        MMmmdM9    MM       M       MM           MMmmdM9    MMmmdM9     MM    M  `MN. M       MM      #
#     ,;j9     MM.      ,MP   MM       M       MM        MM         MM       M       MM           MM         MM  YM.     MM    M   `MM.M       MM      #
#  ,-='    ,,  `Mb.    ,dP'   YM.     ,M       MM        MM         YM.     ,M       MM           MM         MM   `Mb.   MM    M     YMM       MM      #
# Ammmmmmm db    `"bmmd"'      `bmmmmd"'     .JMML.    .JMML.        `bmmmmd"'     .JMML.       .JMML.     .JMML. .JMM..JMML..JML.    YM     .JMML.    #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Save print to a file.
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()
#--------------------------------------------------#
orig_stdout = sys.stdout
f = open(results_sub_folder / 'print_out.txt', 'w')
sys.stdout = Tee(sys.stdout, f)
#--------------------------------------------------#
print("\n\n\n>>> Initializing hyperparameters and settings... ")
print("="*80)
#--------------------------------------------------#
print("Model_code            : ", Model_code     )
print("dataset_nme           : ", dataset_nme    )
print("embedding_file        : ", embedding_file )
#--------------------------------------------------#
print("log_value             : ", log_value,   " (Whether to use log values of Y.)")
print("screen_bool           : ", screen_bool, " (Whether to remove zeroes.)")
#--------------------------------------------------#
print("NN_type               : ", NN_type       )
print("Random Seed           : ", seed          )
print("epoch_num             : ", epoch_num     )
print("batch_size            : ", batch_size    )
print("learning_rate         : ", learning_rate )
#--------------------------------------------------#
# print("-"*80)
# for one_hyperpara in hyperparameters_dict:
#     print(one_hyperpara, " "*(21-len(one_hyperpara)), ": ", hyperparameters_dict[one_hyperpara])
# print("="*80)



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#   pd""b.          `7MM"""Yb.      db  MMP""MM""YMM  db          `7MM"""Mq. `7MM"""Mq.  `7MM"""YMM  `7MM"""Mq.                                        #
#  (O)  `8b           MM    `Yb.   ;MM: P'   MM   `7 ;MM:           MM   `MM.  MM   `MM.   MM    `7    MM   `MM.                                       #
#       ,89           MM     `Mb  ,V^MM.     MM     ,V^MM.          MM   ,M9   MM   ,M9    MM   d      MM   ,M9                                        #
#     ""Yb.           MM      MM ,M  `MM     MM    ,M  `MM          MMmmdM9    MMmmdM9     MMmmMM      MMmmdM9                                         #
#        88           MM     ,MP AbmmmqMA    MM    AbmmmqMA         MM         MM  YM.     MM   Y  ,   MM                                              #
#  (O)  .M'   ,,      MM    ,dP'A'     VML   MM   A'     VML        MM         MM   `Mb.   MM     ,M   MM                                              #
#   bmmmd'    db    .JMMmmmdP'.AMA.   .AMMA.JMML.AMA.   .AMMA.    .JMML.     .JMML. .JMM..JMMmmmmMMM .JMML.                                            #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

###################################################################################################################
#                               `7MM"""YMM `7MMF'`7MMF'      `7MM"""YMM   .M"""bgd                                #
#                                 MM    `7   MM    MM          MM    `7  ,MI    "Y                                #
#                                 MM   d     MM    MM          MM   d    `MMb.                                    #
#                                 MM""MM     MM    MM          MMmmMM      `YMMNq.                                #
#                                 MM   Y     MM    MM      ,   MM   Y  , .     `MM                                #
#                                 MM         MM    MM     ,M   MM     ,M Mb     dM                                #
#                               .JMML.     .JMML..JMMmmmmMMM .JMMmmmmMMM P"Ybmmd"                                 #
###################################################################################################################
# Get Input files
print("\n\n\n>>> Getting all input files and splitting the data... ")
print("="*80)
#====================================================================================================#
# Get Sequence Embeddings from N03 pickles.
print("embedding_file: ", embedding_file)

if os.path.exists(data_folder / embedding_file):
    print("Sequence embeddings found in one file.")
    with open( data_folder / embedding_file, 'rb') as seqs_embeddings:
        seqs_embeddings_pkl = pickle.load(seqs_embeddings)
    try: 
        X_seqs_all_hiddens_list = seqs_embeddings_pkl['seq_all_hiddens']
    except:
        X_seqs_all_hiddens_list = seqs_embeddings_pkl['seqs_all_hiddens']
    del(seqs_embeddings_pkl)
else:
    print("Sequence embeddings found in mulitple files.")
    embedding_file_name = embedding_file.replace(".p", "")
    embedding_file = embedding_file.replace(".p", "_0.p")
    X_seqs_all_hiddens_list_full = []
    while os.path.exists(data_folder / embedding_file):
        with open( data_folder / embedding_file, 'rb') as seqs_embeddings:
            seqs_embeddings_pkl = pickle.load(seqs_embeddings)
        try: 
            X_seqs_all_hiddens_list = seqs_embeddings_pkl['seq_all_hiddens']
        except:
            X_seqs_all_hiddens_list = seqs_embeddings_pkl['seqs_all_hiddens']
        del(seqs_embeddings_pkl)
        X_seqs_all_hiddens_list_full = X_seqs_all_hiddens_list_full + X_seqs_all_hiddens_list

        next_index = str(int(embedding_file.replace(embedding_file_name + "_", "").replace(".p", "")) + 1)
        embedding_file = embedding_file_name + "_" + next_index + ".p"

    X_seqs_all_hiddens_list = copy.deepcopy(X_seqs_all_hiddens_list_full)
    del X_seqs_all_hiddens_list_full
#====================================================================================================#
# Get properties_list.
with open( data_folder / properties_file, 'rb') as seqs_properties:
    properties_dict = pickle.load(seqs_properties)


###################################################################################################################
#                     `7MMF'`7MN.   `7MF'`7MM"""Mq. `7MMF'   `7MF'MMP""MM""YMM  .M"""bgd                          #
#                       MM    MMN.    M    MM   `MM.  MM       M  P'   MM   `7 ,MI    "Y                          #
#                       MM    M YMb   M    MM   ,M9   MM       M       MM      `MMb.                              #
#                       MM    M  `MN. M    MMmmdM9    MM       M       MM        `YMMNq.                          #
#                       MM    M   `MM.M    MM         MM       M       MM      .     `MM                          #
#                       MM    M     YMM    MM         YM.     ,M       MM      Mb     dM                          #
#                     .JMML..JML.    YM  .JMML.        `bmmmmd"'     .JMML.    P"Ybmmd"                           #
###################################################################################################################
# Inputs
def Get_X_y_data(X_seqs_all_hiddens_list, properties_dict, prpty_select, log_value):
    # new: X_seqs_all_hiddens_list
    y_data = properties_dict[prpty_select]
    X_seqs_all_hiddens = []
    y_seqs_prpty = []

    print("len(X_seqs_all_hiddens_list): ", len(X_seqs_all_hiddens_list))
    print("len(y_data): ", len(y_data))


    for j in range(len(X_seqs_all_hiddens_list)):
        if not (np.isnan(y_data[j])):
            X_seqs_all_hiddens.append(X_seqs_all_hiddens_list[j])
            y_seqs_prpty.append(y_data[j])

    y_seqs_prpty = np.array(y_seqs_prpty)
    if log_value == True:
        y_seqs_prpty = np.log10(y_seqs_prpty)

    return X_seqs_all_hiddens, y_seqs_prpty


# Currently this model doesnt support classfication.
if NN_type=="Reg":
    X_seqs_all_hiddens, y_seqs_prpty = Get_X_y_data(X_seqs_all_hiddens_list, properties_dict, prpty_select, log_value)
# if NN_type=="Clf":
#     X_seqs_all_hiddens, y_seqs_prpty = Get_X_y_data_clf(X_seqs_all_hiddens_list, properties_dict, prpty_select)


###################################################################################################################
#                      `7MM"""Mq. `7MM"""Mq.  `7MMF'`7MN.   `7MF'MMP""MM""YMM  .M"""bgd                           #
#                        MM   `MM.  MM   `MM.   MM    MMN.    M  P'   MM   `7 ,MI    "Y                           #
#                        MM   ,M9   MM   ,M9    MM    M YMb   M       MM      `MMb.                               #
#                        MMmmdM9    MMmmdM9     MM    M  `MN. M       MM        `YMMNq.                           #
#                        MM         MM  YM.     MM    M   `MM.M       MM      .     `MM                           #
#                        MM         MM   `Mb.   MM    M     YMM       MM      Mb     dM                           #
#                      .JMML.     .JMML. .JMM..JMML..JML.    YM     .JMML.    P"Ybmmd"                            #
###################################################################################################################
# Prints
print("len(X_seqs_all_hiddens): ", len(X_seqs_all_hiddens), ", len(y_seqs_prpty): ", len(y_seqs_prpty) )

#====================================================================================================#
# Get size of some interested parameters.
X_seqs_all_hiddens_dim = [ max([ X_seqs_all_hiddens_list[i].shape[0] for i in range(len(X_seqs_all_hiddens_list)) ]), X_seqs_all_hiddens_list[0].shape[1], ]
X_seqs_num             = len(X_seqs_all_hiddens_list)
print("seqs dimensions: ", X_seqs_all_hiddens_dim)
print("seqs counts: ", X_seqs_num)

seqs_max_len = max([  X_seqs_all_hiddens_list[i].shape[0] for i in range(len(X_seqs_all_hiddens_list))  ])
print("seqs_max_len: ", seqs_max_len)

NN_input_dim=X_seqs_all_hiddens_dim[1]
print("NN_input_dim: ", NN_input_dim)

# Print the total number of data points.
count_y = len(y_seqs_prpty)
print("Number of Data Points (#y-values): ", count_y)


###################################################################################################################
#        `7MMF'        .g8""8q.      db     `7MM"""Yb.      MMP""MM""YMM `7MM"""YMM   .M"""bgd MMP""MM""YMM       #
#          MM        .dP'    `YM.   ;MM:      MM    `Yb.    P'   MM   `7   MM    `7  ,MI    "Y P'   MM   `7       #
#          MM        dM'      `MM  ,V^MM.     MM     `Mb         MM        MM   d    `MMb.          MM            #
#          MM        MM        MM ,M  `MM     MM      MM         MM        MMmmMM      `YMMNq.      MM            #
#          MM      , MM.      ,MP AbmmmqMA    MM     ,MP         MM        MM   Y  , .     `MM      MM            #
#          MM     ,M `Mb.    ,dP'A'     VML   MM    ,dP'         MM        MM     ,M Mb     dM      MM            #
#        .JMMmmmmMMM   `"bmmd"'.AMA.   .AMMA.JMMmmmdP'         .JMML.    .JMMmmmmMMM P"Ybmmd"     .JMML.          #
###################################################################################################################
# Checklist: embedding_file, properties_file, seqs_fasta_file
if train_option == "TT":

    # Get Input files
    print("\n\n\n>>> Getting all input files and splitting the data... ")
    print("="*80)

    # Get Sequence Embeddings from N03 pickles.
    print("embedding_file_test: ", embedding_file_test)

    if os.path.exists(data_folder / embedding_file_test):
        print("Sequence embeddings found in one file.")
        with open( data_folder / embedding_file_test, 'rb') as seqs_embeddings:
            seqs_embeddings_pkl = pickle.load(seqs_embeddings)
        try: 
            X_seqs_all_hiddens_list = seqs_embeddings_pkl['seq_all_hiddens']
        except:
            X_seqs_all_hiddens_list = seqs_embeddings_pkl['seqs_all_hiddens']
        del(seqs_embeddings_pkl)
    else:
        print("Sequence embeddings found in mulitple files.")
        embedding_file_test_name = embedding_file_test.replace(".p", "")
        embedding_file_test = embedding_file_test.replace(".p", "_0.p")
        X_seqs_all_hiddens_list_full = []
        while os.path.exists(data_folder / embedding_file_test):
            with open( data_folder / embedding_file_test, 'rb') as seqs_embeddings:
                seqs_embeddings_pkl = pickle.load(seqs_embeddings)
            try: 
                X_seqs_all_hiddens_list = seqs_embeddings_pkl['seq_all_hiddens']
            except:
                X_seqs_all_hiddens_list = seqs_embeddings_pkl['seqs_all_hiddens']
            del(seqs_embeddings_pkl)
            X_seqs_all_hiddens_list_full = X_seqs_all_hiddens_list_full + X_seqs_all_hiddens_list

            next_index = str(int(embedding_file_test.replace(embedding_file_test_name + "_", "").replace(".p", "")) + 1)
            embedding_file_test = embedding_file_test_name + "_" + next_index + ".p"

        X_seqs_all_hiddens_list = copy.deepcopy(X_seqs_all_hiddens_list_full)
        del X_seqs_all_hiddens_list_full

    # Get properties_list.
    with open( data_folder / properties_file_test, 'rb') as seqs_properties:
        properties_dict = pickle.load(seqs_properties)


    #====================================================================================================#
    # Inputs
    # Currently this model doesnt support classfication.
    if NN_type=="Reg":
        X_seqs_all_hiddens_test, y_seqs_prpty_test = Get_X_y_data(X_seqs_all_hiddens_list, properties_dict, prpty_select, log_value)


    #====================================================================================================#
    # Prints
    print("len(X_seqs_all_hiddens): ", len(X_seqs_all_hiddens_test), ", len(y_seqs_prpty): ", len(y_seqs_prpty_test) )

    # Get size of some interested parameters.
    X_seqs_all_hiddens_dim_test = [ max([ X_seqs_all_hiddens_list[i].shape[0] for i in range(len(X_seqs_all_hiddens_list)) ]), X_seqs_all_hiddens_list[0].shape[1], ]
    X_seqs_num_test             = len(X_seqs_all_hiddens_list)
    print("seqs dimensions test : ", X_seqs_all_hiddens_dim_test)
    print("seqs counts test     : ", X_seqs_num_test            )

    seqs_max_len_test = max([  X_seqs_all_hiddens_list[i].shape[0] for i in range(len(X_seqs_all_hiddens_list))  ])
    print("seqs_max_len_test: ", seqs_max_len_test); seqs_max_len = seqs_max_len_test if seqs_max_len_test > seqs_max_len else seqs_max_len

    NN_input_dim_test = X_seqs_all_hiddens_dim_test[1]
    print("NN_input_dim_test: ", NN_input_dim_test)

    # Print the total number of data points.
    count_y_test = len(y_seqs_prpty_test)
    print("Number of Data Points (#y-values) (test): ", count_y_test)

###################################################################################################################
#                                  .M"""bgd `7MM"""Mq. `7MMF'      `7MMF'MMP""MM""YMM                             #
#                                 ,MI    "Y   MM   `MM.  MM          MM  P'   MM   `7                             #
#                                 `MMb.       MM   ,M9   MM          MM       MM                                  #
#                                   `YMMNq.   MMmmdM9    MM          MM       MM                                  #
#                                 .     `MM   MM         MM      ,   MM       MM                                  #
#                                 Mb     dM   MM         MM     ,M   MM       MM                                  #
#                                 P"Ybmmd"  .JMML.     .JMMmmmmMMM .JMML.   .JMML.                                #
###################################################################################################################
def split_data(X_seqs_all_hiddens, y_seqs_prpty, train_split, test_split, random_state = seed):
    #y_seqs_prpty=np.log10(y_seqs_prpty)
    X_tr, X_ts, y_tr, y_ts = train_test_split(X_seqs_all_hiddens, y_seqs_prpty, test_size=(1-train_split), random_state = random_state)
    X_va, X_ts, y_va, y_ts = train_test_split(X_ts, y_ts, test_size = (test_split/(1.0-train_split)) , random_state = random_state)
    return X_tr, y_tr, X_ts, y_ts, X_va, y_va


if train_option == "TT": 
    X_tr = X_seqs_all_hiddens
    y_tr = y_seqs_prpty
    X_ts = X_seqs_all_hiddens_test
    y_ts = y_seqs_prpty_test
    X_va = X_seqs_all_hiddens_test
    y_va = y_seqs_prpty_test

else:
    X_tr, y_tr, X_ts, y_ts, X_va, y_va = split_data(X_seqs_all_hiddens, y_seqs_prpty, 0.8, 0.1, random_state = seed)
    print("len(X_tr): ", len(X_tr))
    print("len(X_ts): ", len(X_ts))
    print("len(X_va): ", len(X_va))

if log_value == False:
    y_tr, y_scalar = normalize_targets(y_tr)
    y_ts = y_scalar.transform(y_ts)
    y_va = y_scalar.transform(y_va)

    y_tr = np.array(y_tr, dtype = np.float32)
    y_ts = np.array(y_ts, dtype = np.float32)
    y_va = np.array(y_va, dtype = np.float32)


###################################################################################################################
#                   `7MMF'                              `7MM                            db                        #
#                     MM                                  MM                           ;MM:                       #
#                     MM         ,pW"Wq.   ,6"Yb.    ,M""bMM   .gP"Ya  `7Mb,od8       ,V^MM.                      #
#                     MM        6W'   `Wb 8)   MM  ,AP    MM  ,M'   Yb   MM' "'      ,M  `MM                      #
#                     MM      , 8M     M8  ,pm9MM  8MI    MM  8M""""""   MM          AbmmmqMA                     #
#                     MM     ,M YA.   ,A9 8M   MM  `Mb    MM  YM.    ,   MM         A'     VML                    #
#                   .JMMmmmmMMM  `Ybmd9'  `Moo9^Yo. `Wbmd"MML. `Mbmmd' .JMML.     .AMA.   .AMMA.                  #
###################################################################################################################
class CNN_dataset(data.Dataset):
    def __init__(self, embedding, target, max_len, X_scaler = None):
        super().__init__()
        self.embedding = embedding
        self.target    = target
        self.max_len   = max_len
        self.X_scaler  = X_scaler
    def __len__(self):
        return len(self.embedding)
    def __getitem__(self, idx):
        return self.embedding[idx], self.target[idx]
    def collate_fn(self, batch:List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        embedding, target = zip(*batch)
        batch_size = len(embedding)
        emb_dim = embedding[0].shape[1]
        arra = np.full([batch_size, self.max_len, emb_dim], 0.0)
        for arr, seq in zip(arra, embedding):
            arrslice = tuple(slice(dim) for dim in seq.shape)
            arr[arrslice] = seq
        arra = np.reshape(arra, (batch_size, self.max_len * emb_dim))
        self.X_scaler = sklearn.preprocessing.StandardScaler() if self.X_scaler == None else self.X_scaler
        arra = self.X_scaler.transform(arra) if hasattr(self.X_scaler, "n_features_in_") else self.X_scaler.fit_transform(arra)
        arra = np.reshape(arra, (batch_size, self.max_len, emb_dim))
        return {'seqs_embeddings': torch.from_numpy(arra),  'y_property': torch.tensor(list(target))}

def generate_CNN_loader(X_tr_seqs, y_tr,
                        X_va_seqs, y_va,
                        X_ts_seqs, y_ts,
                        seqs_max_len, batch_size):
    X_y_tr = CNN_dataset(list(X_tr_seqs), y_tr, seqs_max_len)
    X_y_va = CNN_dataset(list(X_va_seqs), y_va, seqs_max_len, X_y_tr.X_scaler)
    X_y_ts = CNN_dataset(list(X_ts_seqs), y_ts, seqs_max_len, X_y_tr.X_scaler)
    train_loader = data.DataLoader(X_y_tr, batch_size, True,  collate_fn = X_y_tr.collate_fn)
    valid_loader = data.DataLoader(X_y_va, batch_size, False, collate_fn = X_y_va.collate_fn)
    test_loader  = data.DataLoader(X_y_ts, batch_size, False, collate_fn = X_y_ts.collate_fn)
    return train_loader, valid_loader, test_loader

if Model_code == "N05A_":
    train_loader, valid_loader, test_loader = generate_CNN_loader(X_tr, y_tr, X_va, y_va, X_ts, y_ts, seqs_max_len, batch_size)

###################################################################################################################
#                   `7MMF'                              `7MM                        `7MM"""Yp,                    #
#                     MM                                  MM                          MM    Yb                    #
#                     MM         ,pW"Wq.   ,6"Yb.    ,M""bMM   .gP"Ya  `7Mb,od8       MM    dP                    #
#                     MM        6W'   `Wb 8)   MM  ,AP    MM  ,M'   Yb   MM' "'       MM"""bg.                    #
#                     MM      , 8M     M8  ,pm9MM  8MI    MM  8M""""""   MM           MM    `Y                    #
#                     MM     ,M YA.   ,A9 8M   MM  `Mb    MM  YM.    ,   MM           MM    ,9                    #
#                   .JMMmmmmMMM  `Ybmd9'  `Moo9^Yo. `Wbmd"MML. `Mbmmd' .JMML.       .JMMmmmd9                     #
###################################################################################################################
class ATT_dataset(data.Dataset):
    def __init__(self, embedding, label, X_scaler = None):
        super().__init__()
        self.embedding = embedding
        self.label     = label
        self.X_scaler  = X_scaler
    def __len__(self):
        return len(self.embedding)
    def __getitem__(self, idx):
        return self.embedding[idx], self.label[idx]
    def collate_fn(self, batch:List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        embedding, target = zip(*batch)
        batch_size = len(embedding)
        emb_dim = embedding[0].shape[1]
        max_len = np.max([s.shape[0] for s in embedding],0)
        arra = np.full([batch_size, max_len, emb_dim], 0.0)
        seq_mask = []
        for arr, seq in zip(arra, embedding):
            padding_len = max_len - len(seq)
            seq_mask.append(np.concatenate((np.ones(len(seq)), np.zeros(padding_len))).reshape(-1, max_len))
            arrslice = tuple(slice(dim) for dim in seq.shape)
            arr[arrslice] = seq
        seq_mask = np.concatenate(seq_mask, axis=0)
        arra = np.reshape(arra, (batch_size, max_len * emb_dim))
        self.X_scaler = sklearn.preprocessing.StandardScaler() if self.X_scaler == None else self.X_scaler
        arra = self.X_scaler.transform(arra) if hasattr(self.X_scaler, "n_features_in_") else self.X_scaler.fit_transform(arra)
        arra = np.reshape(arra, (batch_size, max_len, emb_dim))
        return {'embedding': torch.from_numpy(arra), 'mask': torch.from_numpy(seq_mask), 'y_property': torch.tensor(list(target))}

#====================================================================================================#
def ATT_loader(dataset_class, 
               X_tr_seqs, y_tr,
               X_va_seqs, y_va,
               X_ts_seqs, y_ts,
               batch_size,):
    X_y_tr = dataset_class(list(X_tr_seqs), y_tr, X_scaler = None)
    X_y_va = dataset_class(list(X_va_seqs), y_va, X_y_tr.X_scaler)
    X_y_ts = dataset_class(list(X_ts_seqs), y_ts, X_y_tr.X_scaler)
    train_loader = data.DataLoader(X_y_tr, batch_size, True , collate_fn = X_y_tr.collate_fn)
    valid_loader = data.DataLoader(X_y_va, batch_size, False, collate_fn = X_y_va.collate_fn)
    test_loader  = data.DataLoader(X_y_ts, batch_size, False, collate_fn = X_y_ts.collate_fn)
    return train_loader, valid_loader, test_loader

if Model_code == "N05B_" or Model_code == "N04B_" or Model_code == "N05C_":
    train_loader, valid_loader, test_loader = ATT_loader(ATT_dataset, X_tr, y_tr, X_va, y_va, X_ts, y_ts, batch_size)

###################################################################################################################
#               `7MMM.     ,MMF'  .g8""8q.   `7MM"""Yb.   `7MM"""YMM  `7MMF'                db                    #
#                 MMMb    dPMM  .dP'    `YM.   MM    `Yb.   MM    `7    MM                 ;MM:                   #
#                 M YM   ,M MM  dM'      `MM   MM     `Mb   MM   d      MM                ,V^MM.                  #
#                 M  Mb  M' MM  MM        MM   MM      MM   MMmmMM      MM               ,M  `MM                  #
#                 M  YM.P'  MM  MM.      ,MP   MM     ,MP   MM   Y  ,   MM      ,        AbmmmqMA                 #
#                 M  `YM'   MM  `Mb.    ,dP'   MM    ,dP'   MM     ,M   MM     ,M       A'     VML                #
#               .JML. `'  .JMML.  `"bmmd"'   .JMMmmmdP'   .JMMmmmmMMM .JMMmmmmMMM     .AMA.   .AMMA.              #
###################################################################################################################
class SQembCNN_Model(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 kernal_1: int,
                 out_dim: int,
                 kernal_2: int,
                 max_len: int,
                 last_hid: int,
                 dropout: float = 0.
                 ):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim)
        self.conv1 = nn.Conv1d(in_dim, hid_dim, kernal_1, padding=int((kernal_1-1)/2))
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        
        self.conv2_1 = nn.Conv1d(hid_dim, out_dim, kernal_2, padding=int((kernal_2-1)/2))
        self.dropout2_1 = nn.Dropout(dropout, inplace=True)
        
        self.conv2_2 = nn.Conv1d(hid_dim, hid_dim, kernal_2, padding=int((kernal_2-1)/2))
        self.dropout2_2 = nn.Dropout(dropout, inplace=True)
        
        self.conv3 = nn.Conv1d(hid_dim, out_dim, kernal_2, padding=int((kernal_2-1)/2))
        self.dropout3 = nn.Dropout(dropout, inplace=True)
        #self.pooling = nn.MaxPool1d(3, stride=3,padding=1)
        
        self.fc_1 = nn.Linear(int(2*max_len*out_dim),last_hid)
        self.fc_2 = nn.Linear(last_hid,last_hid)
        self.fc_3 = nn.Linear(last_hid,1)
        self.cls = nn.Sigmoid()

    def forward(self, enc_inputs):
        
        output = enc_inputs.transpose(1, 2)
        output = self.norm(output)
        output = nn.functional.leaky_relu(self.conv1(output))
        output = self.dropout1(output)
        
        output_1 = nn.functional.leaky_relu(self.conv2_1(output))
        output_1 = self.dropout2_1(output_1)
        
        output_2 = nn.functional.leaky_relu(self.conv2_2(output)) + output
        output_2 = self.dropout2_2(output_2)
        
        output_2 = nn.functional.leaky_relu(self.conv3(output_2))
        output_2 = self.dropout3(output_2)
        
        output = torch.cat((output_1,output_2),1)
        #print(output.size())
        
        #output = self.pooling(output)
        
        output = torch.flatten(output,1)
        #print(output.size())
        
        output = self.fc_1(output)
        output = nn.functional.leaky_relu(output)
        output = self.fc_2(output)
        output = nn.functional.leaky_relu(output)
        output = self.fc_3(output)
        return output, output_1




###################################################################################################################
#               `7MMM.     ,MMF'  .g8""8q.   `7MM"""Yb.   `7MM"""YMM  `7MMF'             `7MM"""Yp,               #
#                 MMMb    dPMM  .dP'    `YM.   MM    `Yb.   MM    `7    MM                 MM    Yb               #
#                 M YM   ,M MM  dM'      `MM   MM     `Mb   MM   d      MM                 MM    dP               #
#                 M  Mb  M' MM  MM        MM   MM      MM   MMmmMM      MM                 MM"""bg.               #
#                 M  YM.P'  MM  MM.      ,MP   MM     ,MP   MM   Y  ,   MM      ,          MM    `Y               #
#                 M  `YM'   MM  `Mb.    ,dP'   MM    ,dP'   MM     ,M   MM     ,M          MM    ,9               #
#               .JML. `'  .JMML.  `"bmmd"'   .JMMmmmdP'   .JMMmmmmMMM .JMMmmmmMMM        .JMMmmmd9                #
###################################################################################################################


class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q,K.transpose(-1, -2)) / math.sqrt(self.d_k)
        scores.masked_fill_(attn_mask,-1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttentionwithonekey(nn.Module):
    def __init__(self,d_model,d_k,n_heads,d_v,out_dim):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.out_dim = out_dim
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc_att = nn.Linear(n_heads * d_v, out_dim, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        Q = self.W_Q(input_Q).view(input_Q.size(0),-1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(input_Q.size(0),-1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(input_V.size(0),-1, self.n_heads, self.d_v).transpose(1, 2)

        attn_mask     = attn_mask.unsqueeze(1).repeat(1,self.n_heads,1,1)
        context, attn = ScaledDotProductAttention(self.d_k)(Q, K, V, attn_mask)
        context       = context.transpose(1, 2).reshape(input_Q.size(0), -1, self.n_heads * self.d_v)
        output        = context # [batch_size, len_q, n_heads * d_v]
        return context, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, 1)
            )
        #--------------------------------------------------#
        self.weights = nn.Parameter(torch.from_numpy(np.array([0.0,])), requires_grad = True)
        self.fc_1    = nn.Linear(d_model, d_ff)
        self.fc_2    = nn.Linear(d_ff, d_ff)
        self.fc_3    = nn.Linear(d_ff, 1)

    def forward(self, inputs, input_emb):
        '''
        inputs: [batch_size, src_len, out_dim]
        '''
        output = torch.flatten(inputs, start_dim = 1)
        #output += input_emb.mean(dim = 1) # Avg Pooling
        #output = output*self.weights + input_emb.mean(dim = 1) * (1-self.weights)
        #print(self.weights)

        output = output
        output = self.fc_1(output)
        output = nn.functional.relu(output)
        last_layer = self.fc_2(output)
        output = nn.functional.relu(last_layer)
        output = self.fc_3(output)

        return output, last_layer


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, n_heads, d_v, out_dim, d_ff): #out_dim = 1, n_head = 4, d_k = 256
        super(EncoderLayer, self).__init__()
        self.emb_self_attn = MultiHeadAttentionwithonekey(d_model, d_k, n_heads, d_v, out_dim)
        self.pos_ffn       = PoswiseFeedForwardNet(d_model * n_heads * d_v + d_model, d_ff)

    def forward(self, input_emb, emb_self_attn_mask, input_mask):
        '''
        input_emb: [batch_size, src_len, d_model]
        emb_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # output_emb: [batch_size, src_len, 1], attn: [batch_size, n_heads, src_len, src_len]
        output_emb, attn = self.emb_self_attn(input_emb, input_emb, input_emb, emb_self_attn_mask) # [batch_size, len_q, n_heads * d_v]

        
        mask_expanded      = input_mask.unsqueeze(-1)                         # [B, L, 1]
        output_emb         = output_emb.masked_fill(mask_expanded == 0, -1e9) # output_emb: [batch_size, len_q, n_heads * d_v]
        weights            = F.softmax(output_emb, dim=1)                     # [B, L, n_heads*d_v]
        pooled             = torch.bmm(input_emb.transpose(1, 2), weights)    # [B, d_model, H] (H = n_heads*d_v)
        pooled_flat        = pooled.reshape(pooled.size(0), -1)               # [B, d_model*H]

        lengths            = input_mask.sum(dim=1, keepdim=True).clamp(min=1) # [B, 1]
        avg_emb            = (input_emb * mask_expanded).sum(dim=1) / lengths # [B, d_model]
        pooled_concat      = torch.cat([pooled_flat, avg_emb], dim=1)         # [B, d_model*H + d_model]

        output, last       = self.pos_ffn(pooled_concat, input_emb)           # [B, d_ff]
        
        """
        context       = output_emb
        mask_expanded = input_mask.unsqueeze(-1)                        # [B, L, 1]
        scores        = context.mean(-1, keepdim = True)                # [B, L, 1]
        scores        = scores.masked_fill(mask_expanded == 0, -1e9)
        weights       = F.softmax(scores, dim=1)                        # [B, L, 1]

        pooled        = torch.bmm(weights.transpose(1,2), input_emb)    # [B, 1, d_model]
        pooled        = pooled.squeeze(1)                               # [B, d_model]
        output, last  = self.pos_ffn(pooled, input_emb)                 # as before
        """

        return output, attn


# N05B: SQembSAtt_Model
class SQembSAtt_Model(nn.Module):
    def __init__(self, d_model, d_k, n_heads, d_v, out_dim, d_ff):
        super(SQembSAtt_Model, self).__init__()
        self.layers = EncoderLayer(d_model, d_k, n_heads, d_v, out_dim, d_ff)

    def get_attn_pad_mask(self, seq_mask):
        batch_size, len_q = seq_mask.size()
        _, len_k          = seq_mask.size()
        # eq(zero) is PAD token
        pad_attn_mask = seq_mask.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
        return pad_attn_mask.expand(batch_size, len_q, len_k)        

    def forward(self, input_emb, input_mask):
        '''
        input_emb  : [batch_size, src_len, embedding_dim]
        input_mask : [batch_size, src_len]
        '''
        emb_self_attn_mask = self.get_attn_pad_mask(input_mask) # [batch_size, src_len, src_len]
        # output_emb: [batch_size, src_len, out_dim], emb_self_attn: [batch_size, n_heads, src_len, src_len]
        output_emb, last_layer = self.layers(input_emb, emb_self_attn_mask, input_mask)
        return output_emb, last_layer

###################################################################################################################
#                         `7MMM.     ,MMF'  .g8""8q.   `7MM"""Yb.   `7MM"""YMM  `7MMF'                            #
#                           MMMb    dPMM  .dP'    `YM.   MM    `Yb.   MM    `7    MM                              #
#                           M YM   ,M MM  dM'      `MM   MM     `Mb   MM   d      MM                              #
#                           M  Mb  M' MM  MM        MM   MM      MM   MMmmMM      MM                              #
#                           M  YM.P'  MM  MM.      ,MP   MM     ,MP   MM   Y  ,   MM      ,                       #
#                           M  `YM'   MM  `Mb.    ,dP'   MM    ,dP'   MM     ,M   MM     ,M                       #
#                         .JML. `'  .JMML.  `"bmmd"'   .JMMmmmdP'   .JMMmmmmMMM .JMMmmmmMMM                       #
###################################################################################################################
# N04B_
class SQembAvgP_Model(nn.Module):
    def __init__(self,
                 in_dim  : int   ,
                 hid_1   : int   ,
                 hid_2   : int   ,
                 dropout : float , 
                 ):
        super(SQembAvgP_Model, self).__init__()
        self.fc1 = weight_norm(nn.Linear(in_dim, hid_1),dim=None) 
        self.dropout1 = nn.Dropout(p = dropout) 
        self.fc2 = weight_norm(nn.Linear(hid_1, hid_2),dim=None)
        self.fc3 = weight_norm(nn.Linear(hid_2, 1),dim=None)

    def forward(self, input_emb, input_mask):

        input = input_emb.mean(dim = 1)
        #print(input_emb.size())
        #print(input.size())

        output = nn.functional.leaky_relu(self.fc1(input))
        output = self.dropout1(output)

        last_layer = self.fc2(output)

        output = nn.functional.leaky_relu(last_layer)
        output = self.fc3(output)

        return output, last_layer


class SelfAttentiveEmb(nn.Module):                # N05C_
    """
    Self-attention pooling (Lin et al., ICLR-2017) with a GELU scorer
    + a two-layer regression head that maps the pooled representation
    to a single scalar target.

    Parameters
    ----------
    d_model      : token-embedding size (1280 for ESM-1b).
    d_attn       : hidden size inside the scorer MLP.
    r            : number of attention rows / heads.
    hidden_pred  : hidden size of the 2-layer prediction head.
    """
    def __init__(
        self,
        d_model: int = 1280,
        d_attn:  int = 128,
        r:       int = 1,
        hidden_pred: int = 512       # <-- requested default
    ):
        super().__init__()
        # scorer
        self.W1 = nn.Linear(d_model, d_attn, bias=False)
        self.W2 = nn.Linear(d_attn,  r,      bias=False)
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)

        # optional diversity loss coefficient
        self.penalty_lambda = 1e-3

        # 2-layer prediction head:  (r * d_model) → hidden_pred → 1
        self.head = nn.Sequential(
            nn.LayerNorm(r * d_model),
            nn.Linear(r * d_model, hidden_pred),
            nn.GELU(),
            nn.Linear(hidden_pred, 1)
        )

    def forward(
        self,
        H: torch.Tensor,                       # [B, L, d_model]
        mask: Optional[torch.Tensor] = None    # [B, L]  (1 = keep, 0 = pad)
    ):
        # ----- self-attention pooling -----
        scores = self.W2(F.gelu(self.W1(H)))          # [B, L, r]
        scores = scores.transpose(1, 2)               # [B, r, L]
        if mask is not None:
            scores = scores.masked_fill(mask[:, None, :] == 0, -1e9)
        A = F.softmax(scores, dim=-1)                 # [B, r, L]
        M = torch.bmm(A, H)                           # [B, r, d_model]
        pooled = M.reshape(M.size(0), -1)             # [B, r*d_model]

        # ----- optional diversity penalty -----
        if self.penalty_lambda > 0 and M.size(1) > 1:
            I   = torch.eye(A.size(1), device=A.device)
            ATA = torch.bmm(A, A.transpose(1, 2))     # [B, r, r]
            penalty = self.penalty_lambda * ((ATA - I) ** 2).sum(dim=(1, 2)).mean()
        else:
            penalty = 0.0

        # ----- scalar prediction -----
        target = self.head(pooled)                    # [B, 1]

        # return:  scalar prediction, pooled representation
        # (access `self.last_penalty` if you need the diversity loss)
        self.last_penalty = penalty
        return target, pooled






###################################################################################################################
#                                `7MMF'        .g8""8q.         db      `7MM"""Yb.                                #
#                                  MM        .dP'    `YM.      ;MM:       MM    `Yb.                              #
#                                  MM        dM'      `MM     ,V^MM.      MM     `Mb                              #
#                                  MM        MM        MM    ,M  `MM      MM      MM                              #
#                                  MM      , MM.      ,MP    AbmmmqMA     MM     ,MP                              #
#                                  MM     ,M `Mb.    ,dP'   A'     VML    MM    ,dP'                              #
#                                .JMMmmmmMMM   `"bmmd"'   .AMA.   .AMMA..JMMmmmdP'                                #
###################################################################################################################

if Model_code == "N05A_":
    model = SQembCNN_Model(in_dim       =  NN_input_dim ,
                           hid_dim      =  hid_dim      ,
                           kernal_1     =  kernal_1     ,
                           out_dim      =  out_dim      ,
                           kernal_2     =  kernal_2     ,
                           max_len      =  seqs_max_len ,
                           last_hid     =  last_hid     ,
                           dropout      =  dropout      ,
                           )
    input_var_names_list = ["seqs_embeddings", ]

if Model_code == "N05B_":
    model = SQembSAtt_Model(d_model    = NN_input_dim  , 
                            d_k        = d_k           , 
                            n_heads    = n_heads       , 
                            d_v        = d_v           , 
                            out_dim    = 1             , 
                            d_ff       = last_hid      , 
                            )
    input_var_names_list =  ["embedding" , "mask", ]


if Model_code == "N04B_":
    model = SQembAvgP_Model( in_dim  = NN_input_dim , 
                             hid_1   = hid_1        , 
                             hid_2   = hid_2        ,
                             dropout = dropout      ,
                             )
    input_var_names_list =  ["embedding" , "mask", ]


if Model_code == "N05C_":
    model = SelfAttentiveEmb(d_model    = NN_input_dim  , 
                             )
    input_var_names_list =  ["embedding" , "mask", ]



num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{num_params:,} trainable parameters.")

model.double()
model.cuda()
#--------------------------------------------------#
print("#"*50)
print(model)
#model.float()
#print( summary( model,[(seqs_max_len, NN_input_dim),] )  )
#model.double()
print("#"*50)
#--------------------------------------------------#
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
criterion = nn.MSELoss()


###################################################################################################################
#                        MMP""MM""YMM `7MM"""Mq.        db      `7MMF'`7MN.   `7MF'                               #
#                        P'   MM   `7   MM   `MM.      ;MM:       MM    MMN.    M                                 #
#                             MM        MM   ,M9      ,V^MM.      MM    M YMb   M                                 #
#                             MM        MMmmdM9      ,M  `MM      MM    M  `MN. M                                 #
#                             MM        MM  YM.      AbmmmqMA     MM    M   `MM.M                                 #
#                             MM        MM   `Mb.   A'     VML    MM    M     YMM                                 #
#                           .JMML.    .JMML. .JMM..AMA.   .AMMA..JMML..JML.    YM                                 #
###################################################################################################################



###################################################################################################################
###################################################################################################################
# Step 6. Now, train the model
print("\n\n\n>>>  Training... ")
print("="*80)

for epoch in range(epoch_num): 
    begin_time = time.time()
    #====================================================================================================#
    # Train
    model.train()
    count_x=0
    for one_seq_ppt_group in train_loader:
        len_train_loader=len(train_loader)
        count_x+=1

        if count_x == 20 :
            print(" " * 12, end = " ") 
        if ((count_x) % 160) == 0:
            print( str(count_x) + "/" + str(len_train_loader) + "->" + "\n" + " " * 12, end=" ")
        elif ((count_x) % 20) == 0:
            print( str(count_x) + "/" + str(len_train_loader) + "->", end=" ")
        #--------------------------------------------------#
        input_vars = [one_seq_ppt_group[one_var].double().cuda() for one_var in input_var_names_list]
        output, _  = model(*input_vars)
        target     = one_seq_ppt_group["y_property"].double().cuda()
        loss = criterion(output,target.view(-1,1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #====================================================================================================#
    model.eval()
    y_pred_valid = []
    y_real_valid = []
    #--------------------------------------------------#
    # Validation.
    for one_seq_ppt_group in valid_loader:
        input_vars = [one_seq_ppt_group[one_var].double().cuda() for one_var in input_var_names_list]
        output, _  = model(*input_vars)
        output = output.cpu().detach().numpy().reshape(-1)
        target = one_seq_ppt_group["y_property"].numpy()
        y_pred_valid.append(output)
        y_real_valid.append(target)
    y_pred_valid = np.concatenate(y_pred_valid)
    y_real_valid = np.concatenate(y_real_valid)
    slope, intercept, r_value_va, p_value, std_err = scipy.stats.linregress(y_pred_valid, y_real_valid)

    #====================================================================================================#
    y_pred = []
    y_real = []
    #--------------------------------------------------#

    for one_seq_ppt_group in test_loader:
        input_vars = [one_seq_ppt_group[one_var].double().cuda() for one_var in input_var_names_list]
        output, _  = model(*input_vars)
        output = output.cpu().detach().numpy().reshape(-1)
        target = one_seq_ppt_group["y_property"].numpy()
        y_pred.append(output)
        y_real.append(target)
    y_pred = np.concatenate(y_pred)
    y_real = np.concatenate(y_real)
    #--------------------------------------------------#
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_pred, y_real)

    #====================================================================================================#
    # Report.
    loss_copy = copy.copy(loss)
    print("\n" + "_" * 101, end = " ")
    print("\nepoch: {} | time_elapsed: {:5.4f} | train_loss: {:5.4f} | vali_R_VALUE: {:5.4f} | test_R_VALUE: {:5.4f} ".format( 
            str((epoch+1)+1000).replace("1","",1), 

            np.round((time.time()-begin_time), 5),
            np.round(loss_copy.cpu().detach().numpy(), 5), 
            np.round(r_value_va, 5), 
            np.round(r_value, 5),
            )
            )

    r_value, r_value_va = r_value, r_value_va 

    va_MAE  = np.round(mean_absolute_error(y_pred_valid, y_real_valid), 4)
    va_MSE  = np.round(mean_squared_error (y_pred_valid, y_real_valid), 4)
    va_RMSE = np.round(math.sqrt(va_MSE), 4)
    va_R2   = np.round(r2_score(y_real_valid, y_pred_valid), 4)
    va_rho  = np.round(scipy.stats.spearmanr(y_pred_valid, y_real_valid)[0], 4)
    
    
    ts_MAE  = np.round(mean_absolute_error(y_pred, y_real), 4)
    ts_MSE  = np.round(mean_squared_error (y_pred, y_real), 4)
    ts_RMSE = np.round(math.sqrt(ts_MSE), 4) 
    ts_R2   = np.round(r2_score(y_real, y_pred), 4)
    ts_rho  = np.round(scipy.stats.spearmanr(y_pred, y_real)[0], 4)

    print("           | va_MAE: {:4.3f} | va_MSE: {:4.3f} | va_RMSE: {:4.3f} | va_R2: {:4.3f} | va_rho: {:4.3f} ".format( 
            va_MAE, 
            va_MSE,
            va_RMSE, 
            va_R2, 
            va_rho,
            )
            )

    print("           | ts_MAE: {:4.3f} | ts_MSE: {:4.3f} | ts_RMSE: {:4.3f} | ts_R2: {:4.3f} | ts_rho: {:4.3f} ".format( 
            ts_MAE, 
            ts_MSE,
            ts_RMSE, 
            ts_R2, 
            ts_rho,
            )
            )

    y_pred_all = np.concatenate([y_pred, y_pred_valid], axis = None)
    y_real_all = np.concatenate([y_real, y_real_valid], axis = None)

    all_rval = np.round(scipy.stats.pearsonr(y_pred_all, y_real_all), 5)
    all_MAE  = np.round(mean_absolute_error(y_pred_all, y_real_all), 4)
    all_MSE  = np.round(mean_squared_error (y_pred_all, y_real_all), 4)
    all_RMSE = np.round(math.sqrt(ts_MSE), 4) 
    all_R2   = np.round(r2_score(y_real_all, y_pred_all), 4)
    all_rho  = np.round(scipy.stats.spearmanr(y_pred_all, y_real_all)[0], 4)

    print("           | tv_MAE: {:4.3f} | tv_MSE: {:4.3f} | tv_RMSE: {:4.3f} | tv_R2: {:4.3f} | tv_rho: {:4.3f} ".format( 
            all_MAE ,
            all_MSE ,
            all_RMSE,
            all_R2  ,
            all_rho ,
            )
            )
    print("           | tv_R_VALUE:", all_rval)

    print("_" * 101)

    #====================================================================================================#
    # Plot.
    if ((epoch+1) % 1) == 0:
        if log_value == False:
            y_pred = y_scalar.inverse_transform(y_pred)
            y_real = y_scalar.inverse_transform(y_real)


        _, _, r_value, _ , _ = scipy.stats.linregress(y_pred, y_real)
        reg_scatter_distn_plot(y_pred,
                                y_real,
                                fig_size        =  (10,8),
                                marker_size     =  35,
                                fit_line_color  =  "brown",
                                distn_color_1   =  "gold",
                                distn_color_2   =  "lightpink",
                                # title         =  "Predictions vs. Actual Values\n R = " + \
                                #                         str(round(r_value,3)) + \
                                #                         ", Epoch: " + str(epoch+1) ,
                                title           =  "",
                                plot_title      =  "R = " + str(round(r_value,3)) + \
                                                        "\n" + r'$\rho$' + " = "  + \
                                                        str(round(ts_rho,3))      ,
                                x_label         =  "Actual Values",
                                y_label         =  "Predictions",
                                cmap            =  None,
                                cbaxes          =  (0.425, 0.055, 0.525, 0.015),
                                font_size       =  18,
                                result_folder   =  results_sub_folder,
                                file_name       =  output_file_header + "_TS_" + "epoch_" + str(epoch+1),
                                ) #For checking predictions fittings.




        _, _, r_value, _ , _ = scipy.stats.linregress(y_pred_valid, y_real_valid)                       
        reg_scatter_distn_plot(y_pred_valid,
                                y_real_valid,
                                fig_size        =  (10,8),
                                marker_size     =  35,
                                fit_line_color  =  "brown",
                                distn_color_1   =  "gold",
                                distn_color_2   =  "lightpink",
                                # title         =  "Predictions vs. Actual Values\n R = " + \
                                #                         str(round(r_value,3)) + \
                                #                         ", Epoch: " + str(epoch+1) ,
                                title           =  "",
                                plot_title      =  "R = " + str(round(r_value,3)) + \
                                                        "\n" + r'$\rho$' + " = "  + \
                                                        str(round(va_rho,3))      ,
                                x_label         =  "Actual Values",
                                y_label         =  "Predictions",
                                cmap            =  None,
                                cbaxes          =  (0.425, 0.055, 0.525, 0.015),
                                font_size       =  18,
                                result_folder   =  results_sub_folder,
                                file_name       =  output_file_header + "_VA_" + "epoch_" + str(epoch+1),
                                ) #For checking predictions fittings.




    #====================================================================================================#
        if log_value == False and screen_bool==True:
            y_real = np.delete(y_real, np.where(y_pred == 0.0))
            y_pred = np.delete(y_pred, np.where(y_pred == 0.0))
            y_real = np.log10(y_real)
            y_pred = np.log10(y_pred)

            reg_scatter_distn_plot(y_pred,
                                y_real,
                                fig_size       = (10,8),
                                marker_size    = 20,
                                fit_line_color = "brown",
                                distn_color_1  = "gold",
                                distn_color_2  = "lightpink",
                                # title         =  "Predictions vs. Actual Values\n R = " + \
                                #                         str(round(r_value,3)) + \
                                #                         ", Epoch: " + str(epoch+1) ,
                                title           =  "",
                                plot_title      =  "R = " + str(round(r_value,3)) + \
                                                          "\nEpoch: " + str(epoch+1) ,
                                x_label        = "Actual Values",
                                y_label        = "Predictions",
                                cmap           = None,
                                font_size      = 18,
                                result_folder  = results_sub_folder,
                                file_name      = output_file_header + "_logplot" + "epoch_" + str(epoch+1),
                                ) #For checking predictions fittings.

###################################################################################################################
###################################################################################################################
print(Step_code, "Done!")


#       M              M              M              M              M               M              M              M              M              M      #
#       M              M              M              M              M               M              M              M              M              M      #
#       M              M              M              M              M               M              M              M              M              M      #
#   `7M'M`MF'      `7M'M`MF'      `7M'M`MF'      `7M'M`MF'      `7M'M`MF'       `7M'M`MF'      `7M'M`MF'      `7M'M`MF'      `7M'M`MF'      `7M'M`MF'  #
#     VAMAV          VAMAV          VAMAV          VAMAV          VAMAV           VAMAV          VAMAV          VAMAV          VAMAV          VAMAV    #
#      VVV            VVV            VVV            VVV            VVV             VVV            VVV            VVV            VVV            VVV     #
#       V              V              V              V              V               V              V              V              V              V      #

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
###################################################################################################################
###################################################################################################################
#====================================================================================================#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#--------------------------------------------------#
#------------------------------

#                                                                                                                                                          
#      `MM.              `MM.             `MM.             `MM.             `MM.             `MM.             `MM.             `MM.             `MM.       
#        `Mb.              `Mb.             `Mb.             `Mb.             `Mb.             `Mb.             `Mb.             `Mb.             `Mb.     
# MMMMMMMMMMMMD     MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD   
#         ,M'               ,M'              ,M'              ,M'              ,M'              ,M'              ,M'              ,M'              ,M'     
#       .M'               .M'              .M'              .M'              .M'              .M'              .M'              .M'              .M'       
#                                                                                                                                                          

#------------------------------
#--------------------------------------------------#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#====================================================================================================#
###################################################################################################################
###################################################################################################################
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

#       A              A              A              A              A               A              A              A              A              A      #
#      MMM            MMM            MMM            MMM            MMM             MMM            MMM            MMM            MMM            MMM     #
#     MMMMM          MMMMM          MMMMM          MMMMM          MMMMM           MMMMM          MMMMM          MMMMM          MMMMM          MMMMM    #
#   ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.       ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.  #
#       M              M              M              M              M               M              M              M              M              M      #
#       M              M              M              M              M               M              M              M              M              M      #
#       M              M              M              M              M               M              M              M              M              M      #




