#!/usr/bin/env python
# coding: utf-8
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Microsoft VS header
#--------------------------------------------------#
import os 
import sys
import os.path
from sys import platform
from pathlib import Path
#--------------------------------------------------#
if os.name == 'nt' or platform == 'win32':
    print("Running on Windows")
    if 'ptvsd' in sys.modules:
        print("Running in Visual Studio")
        try:
            os.chdir(os.path.dirname(__file__))
            print('CurrentDir: ', os.getcwd())
        except:
            pass
#--------------------------------------------------#
    else:
        print("Running outside Visual Studio")
        try:
            if not 'workbookDir' in globals():
                workbookDir = os.getcwd()
                print('workbookDir: ' + workbookDir)
                os.chdir(workbookDir)
        except:
            pass
#--------------------------------------------------#
###################################################################################################################
###################################################################################################################
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

from Z05K_KNN import *

seed = 0
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
Step_code = "N05K_"
#--------------------------------------------------#
dataset_nme_list     = ["NovoEnzyme",            # 0
                        "PafAVariants",          # 1
                        ]
dataset_nme          = dataset_nme_list[1]

data_folder = Path("N_DataProcessing/")

embedding_file_list = [ "N03_" + dataset_nme + "_embedding_ESM_1B.p"    ,      # 0
                        "N03_" + dataset_nme + "_embedding_ESM_1V.p"    ,      # 1
                        "N03_" + dataset_nme + "_embedding_ESM_2_650.p" ,      # 2
                        "N03_" + dataset_nme + "_embedding_ESM_2_3B.p"  ,      # 3
                        "N03_" + dataset_nme + "_embedding_BERT.p"      ,      # 4

                        "N03_" + dataset_nme + "_embedding_TAPE.p"      ,      # 5
                        "N03_" + dataset_nme + "_embedding_ALBERT.p"    ,      # 6
                        "N03_" + dataset_nme + "_embedding_T5.p"        ,      # 7
                        "N03_" + dataset_nme + "_embedding_TAPE_FT.p"   ,      # 8
                        "N03_" + dataset_nme + "_embedding_Xlnet.p"     ,      # 9
                        ]

embedding_file      = embedding_file_list[0]

properties_file     = "N00_" + dataset_nme + "_seqs_prpty_list.p"
seqs_fasta_file     = "N00_" + dataset_nme + ".fasta"
# The embedding_file is a dict, {"seq_embeddings":seq_embeddings, "seq_ids":seq_ids, "seq_all_hiddens":seq_all_hiddens}
# The properties_file is a dict.

#====================================================================================================#
# Select properties (Y) of the model 
prpty_list = [
              [
               "tm"                    , # 0
              ],

              ["kcat_cMUP"             , # 0
               "KM_cMUP"               , # 1
               "kcatOverKM_cMUP"       , # 2
               "kcatOverKM_MeP"        , # 3
               "kcatOverKM_MecMUP"     , # 4
               "Ki_Pi"                 , # 5
               "fa"                    , # 6
               "kcatOverKM_MePkchem"   , # 7
               "FC1"                   , # 8
               "FC2_3"                 , # 9
               "FC4"                   , # 10
              ],
             ][dataset_nme_list.index(dataset_nme)]


prpty_select = prpty_list[5]

#====================================================================================================#
# Prediction NN settings
NN_type_list   = ["Reg", "Clf"]
NN_type        = NN_type_list[0]
epoch_num      = 100
batch_size     = 256 * 4
learning_rate  =  [0.01        , # 0
                   0.005       , # 1
                   0.002       , # 2
                   0.001       , # 3
                   0.0005      , # 4
                   0.0002      , # 5
                   0.0001      , # 6
                   0.00005     , # 7
                   0.00002     , # 8
                   0.00001     , # 9
                   0.000005    , # 10
                   0.000002    , # 11
                   0.000001    , # 12
                   ][5]          # 

#====================================================================================================#
# Hyperparameters.
d_k       =  256    # 256
n_heads   =  1      # 1  
out_dim   =  1     
d_v       =  256    # 
last_hid  =  1024   # d_ff
dropout   =  0.0    # 0.1, 0.6
sub_vab   =  50
#====================================================================================================#
# Prepare print outputs.
hyperparameters_dict = dict([])
for one_hyperpara in ["d_k", "n_heads", "out_dim", "d_v", "last_hid", "sub_vab", "dropout"]:
    hyperparameters_dict[one_hyperpara] = locals()[one_hyperpara]
#====================================================================================================#
# If log_value is True, screen_bool will be changed.
screen_bool = bool(0) # Currently screening y values is NOT supported.
log_value   = bool(0) ##### !!!!! If value is True, screen_bool will be changed
if log_value == True:
    screen_bool = True
#====================================================================================================#
# Results Output
results_folder = Path("N_DataProcessing/" + Step_code +"intermediate_results/")
output_file_3  = Step_code + "_all_X_y.p"
output_file_header = Step_code + "_result_"


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
count_non_empty_folder = 0
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
print("dataset_nme           : ", dataset_nme)
print("embedding_file        : ", embedding_file)
#--------------------------------------------------#
print("log_value             : ", log_value,   " (Whether to use log values of Y.)")
print("screen_bool           : ", screen_bool, " (Whether to remove zeroes.)")
#--------------------------------------------------#
print("NN_type               : ", NN_type)
print("Random Seed           : ", seed)
print("epoch_num             : ", epoch_num)
print("batch_size            : ", batch_size)
print("learning_rate         : ", learning_rate)
#--------------------------------------------------#
print("-"*80)
for one_hyperpara in hyperparameters_dict:
    print(one_hyperpara, " "*(21-len(one_hyperpara)), ": ", hyperparameters_dict[one_hyperpara])
print("="*80)



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

print(embedding_file)

if os.path.exists(data_folder / embedding_file):
    print("Sequence embeddings found in one file.")
    with open( data_folder / embedding_file, 'rb') as seqs_embeddings:
        seqs_embeddings_pkl = pickle.load(seqs_embeddings)

    try: 
        X_seqs_all_hiddens_list = seqs_embeddings_pkl['seq_all_hiddens']
    except:
        X_seqs_all_hiddens_list = seqs_embeddings_pkl['seqs_all_hiddens']

    X_seqs_all_ids_list = seqs_embeddings_pkl['seq_ids']

    del(seqs_embeddings_pkl)
else:
    print("Sequence embeddings found in mulitple files.")
    embedding_file_name = embedding_file.replace(".p", "")
    embedding_file = embedding_file.replace(".p", "_0.p")
    X_seqs_all_hiddens_list_full = []
    X_seqs_all_ids_list_full = []
    while os.path.exists(data_folder / embedding_file):
        with open( data_folder / embedding_file, 'rb') as seqs_embeddings:
            seqs_embeddings_pkl = pickle.load(seqs_embeddings)
        try: 
            X_seqs_all_hiddens_list = seqs_embeddings_pkl['seq_all_hiddens']
        except:
            X_seqs_all_hiddens_list = seqs_embeddings_pkl['seqs_all_hiddens']
        X_seqs_all_ids_list = seqs_embeddings_pkl['seq_ids']
        del(seqs_embeddings_pkl)
        X_seqs_all_hiddens_list_full = X_seqs_all_hiddens_list_full + X_seqs_all_hiddens_list
        X_seqs_all_ids_list_full = X_seqs_all_ids_list_full + X_seqs_all_ids_list
        next_index = str(int(embedding_file.replace(embedding_file_name + "_", "").replace(".p", "")) + 1)
        embedding_file = embedding_file_name + "_" + next_index + ".p"

    X_seqs_all_ids_list = copy.deepcopy(X_seqs_all_ids_list_full)
    X_seqs_all_hiddens_list = copy.deepcopy(X_seqs_all_hiddens_list_full)
    del X_seqs_all_hiddens_list_full

#====================================================================================================#
# Get a list of all sequences.
with open(data_folder / seqs_fasta_file) as f:
    lines = f.readlines()
one_sequence = ""
X_all_seqs_list = []
seqs_nme = ""
for line_idx, one_line in enumerate(lines):
    if ">seq" in one_line:
        if one_sequence != "":
            X_all_seqs_list.append(one_sequence)
        # new sequence start from here
        one_sequence = ""
        seqs_nme = one_line.replace("\n", "")
    if ">seq" not in one_line:
        one_sequence = one_sequence + one_line.replace("\n", "")
    if line_idx == len(lines) - 1:
        X_all_seqs_list.append(one_sequence)


#====================================================================================================#
emb_to_seqs_dict = dict([])
for i, (embd, seqs) in enumerate(zip(X_seqs_all_hiddens_list , X_all_seqs_list)):
    emb_to_seqs_dict[str(embd)] = seqs




    
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

###################################################################################################################
###################################################################################################################
# Currently doesnt support classfication model.
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
#save_dict=dict([])
#save_dict["X_seqs_all_hiddens"] = X_seqs_all_hiddens
#save_dict["y_seqs_prpty"] = y_seqs_prpty
#pickle.dump( save_dict , open( results_folder / output_file_3, "wb" ) )
#print("Done getting X_seqs_all_hiddens and y_seqs_prpty!")
print("len(X_seqs_all_hiddens): ", len(X_seqs_all_hiddens), ", len(y_seqs_prpty): ", len(y_seqs_prpty) )

#====================================================================================================#
# Get size of some interested parameters.
X_seqs_all_hiddens_dim = [ max([ X_seqs_all_hiddens_list[i].shape[0] for i in range(len(X_seqs_all_hiddens_list)) ]), X_seqs_all_hiddens_list[0].shape[1], ]
X_seqs_num = len(X_seqs_all_hiddens_list)
print("seqs dimensions: ", X_seqs_all_hiddens_dim)
print("seqs counts: ", X_seqs_num)

seqs_max_len = max([  X_seqs_all_hiddens_list[i].shape[0] for i in range(len(X_seqs_all_hiddens_list))  ])
print("seqs_max_len: ", seqs_max_len)

NN_input_dim = X_seqs_all_hiddens_dim[1]
print("NN_input_dim: ", NN_input_dim)

# Print the total number of data points.
count_y = len(y_seqs_prpty)
print("Number of Data Points (#y-values): ", count_y)


###################################################################################################################
#                                  .M"""bgd `7MM"""Mq. `7MMF'      `7MMF'MMP""MM""YMM                             #
#                                 ,MI    "Y   MM   `MM.  MM          MM  P'   MM   `7                             #
#                                 `MMb.       MM   ,M9   MM          MM       MM                                  #
#                                   `YMMNq.   MMmmdM9    MM          MM       MM                                  #
#                                 .     `MM   MM         MM      ,   MM       MM                                  #
#                                 Mb     dM   MM         MM     ,M   MM       MM                                  #
#                                 P"Ybmmd"  .JMML.     .JMMmmmmMMM .JMML.   .JMML.                                #
###################################################################################################################
def split_data(X_seqs_all_hiddens, X_all_seqs_list, y_seqs_prpty, train_split, test_split, random_state = seed):
    #y_seqs_prpty=np.log10(y_seqs_prpty)
    X_tr, X_ts, y_tr, y_ts = train_test_split(X_seqs_all_hiddens, y_seqs_prpty, test_size=(1-train_split), random_state = random_state)
    X_va, X_ts, y_va, y_ts = train_test_split(X_ts, y_ts, test_size = (test_split/(1.0-train_split)) , random_state = random_state)

    X_tr_seqs = [emb_to_seqs_dict[str(one_emb)] for one_emb in X_tr]
    X_ts_seqs = [emb_to_seqs_dict[str(one_emb)] for one_emb in X_ts]
    X_va_seqs = [emb_to_seqs_dict[str(one_emb)] for one_emb in X_va]

    return X_tr, y_tr, X_ts, y_ts, X_va, y_va, X_tr_seqs, X_ts_seqs, X_va_seqs

X_tr, y_tr, X_ts, y_ts, X_va, y_va, X_tr_seqs, X_ts_seqs, X_va_seqs = \
    split_data(X_seqs_all_hiddens, X_all_seqs_list, y_seqs_prpty, 0.8, 0.1, random_state = seed)

print("len(X_tr): ", len(X_tr))
print("len(X_ts): ", len(X_ts))
print("len(X_va): ", len(X_va))

#====================================================================================================#
if log_value == False:
    y_tr, y_scalar = normalize_targets(y_tr)
    y_ts = y_scalar.transform(y_ts)
    y_va = y_scalar.transform(y_va)

    y_tr = np.array(y_tr, dtype = np.float32)
    y_ts = np.array(y_ts, dtype = np.float32)
    y_va = np.array(y_va, dtype = np.float32)


###################################################################################################################
#                         `7MMF'                              `7MM                                                #
#                           MM                                  MM                                                #
#                           MM         ,pW"Wq.   ,6"Yb.    ,M""bMM   .gP"Ya  `7Mb,od8                             #
#                           MM        6W'   `Wb 8)   MM  ,AP    MM  ,M'   Yb   MM' "'                             #
#                           MM      , 8M     M8  ,pm9MM  8MI    MM  8M""""""   MM                                 #
#                           MM     ,M YA.   ,A9 8M   MM  `Mb    MM  YM.    ,   MM                                 #
#                         .JMMmmmmMMM  `Ybmd9'  `Moo9^Yo. `Wbmd"MML. `Mbmmd' .JMML.                               #
###################################################################################################################

if log_value == False:
    y_tr = y_scalar.inverse_transform(y_tr)
    y_ts = y_scalar.inverse_transform(y_ts)
    y_va = y_scalar.inverse_transform(y_va)

if log_value == True:
    y_tr = np.power(10, y_tr)
    y_ts = np.power(10, y_ts)
    y_va = np.power(10, y_va)


all_data_split_dict = dict([])
all_data_split_dict_file_output = Step_code + dataset_nme  + "_data_splt.p"

all_data_split_dict["X_tr_seqs"] = X_tr_seqs           # sequences in training set.
all_data_split_dict["X_ts_seqs"] = X_ts_seqs           # sequences in test set.
all_data_split_dict["X_va_seqs"] = X_va_seqs           # sequences in validation set.

#all_data_split_dict["X_tr_seqs_emb"] = X_tr_seqs_emb   # seqs embeddings in training set.
#all_data_split_dict["X_ts_seqs_emb"] = X_ts_seqs_emb   # seqs embeddings in test set.
#all_data_split_dict["X_va_seqs_emb"] = X_va_seqs_emb   # seqs embeddings in validation set.


all_data_split_dict["y_tr"] = y_tr                     # target values in training set.
all_data_split_dict["y_ts"] = y_ts                     # target values in test set.
all_data_split_dict["y_va"] = y_va                     # target values in validation set.


pickle.dump(all_data_split_dict, open(data_folder / all_data_split_dict_file_output, "wb"))


###################################################################################################################
#                       `7MMM.     ,MMF'  .g8""8q.   `7MM"""Yb.   `7MM"""YMM  `7MMF'                              #
#                         MMMb    dPMM  .dP'    `YM.   MM    `Yb.   MM    `7    MM                                #
#                         M YM   ,M MM  dM'      `MM   MM     `Mb   MM   d      MM                                #
#                         M  Mb  M' MM  MM        MM   MM      MM   MMmmMM      MM                                #
#                         M  YM.P'  MM  MM.      ,MP   MM     ,MP   MM   Y  ,   MM      ,                         #
#                         M  `YM'   MM  `Mb.    ,dP'   MM    ,dP'   MM     ,M   MM     ,M                         #
#                       .JML. `'  .JMML.  `"bmmd"'   .JMMmmmdP'   .JMMmmmmMMM .JMMmmmmMMM                         #
###################################################################################################################


#====================================================================================================#
#   M******    `7MM     `7MMF'                              `7MM  
#  .M            MM       MM                                  MM  
#  |bMMAg.       MM       MM         ,pW"Wq.   ,6"Yb.    ,M""bMM  
#       `Mb ,,   MM       MM        6W'   `Wb 8)   MM  ,AP    MM  
#        jM db .JMML.     MM      , 8M     M8  ,pm9MM  8MI    MM  
#  (O)  ,M9               MM     ,M YA.   ,A9 8M   MM  `Mb    MM  
#   6mmm9               .JMMmmmmMMM  `Ybmd9'  `Moo9^Yo. `Wbmd"MML.
#====================================================================================================#\

all_data_split_dict_file_output = Step_code + dataset_nme  + "_data_splt.p"
all_data_split_dict = pickle.load(open(data_folder / all_data_split_dict_file_output, "rb"))

X_tr_seqs  =  all_data_split_dict["X_tr_seqs"] 
X_ts_seqs  =  all_data_split_dict["X_ts_seqs"] 
X_va_seqs  =  all_data_split_dict["X_va_seqs"] 

y_tr       = all_data_split_dict["y_tr"]       
y_ts       = all_data_split_dict["y_ts"]       
y_va       = all_data_split_dict["y_va"]    

all_seqs = X_tr_seqs + X_ts_seqs + X_va_seqs
all_y    = np.concatenate([y_tr, y_ts, y_va])


#====================================================================================================#
#                            .M"""bgd            mm      mm      db                                
#                           ,MI    "Y            MM      MM                                        
#   M******     pd*"*b.     `MMb.      .gP"Ya  mmMMmm  mmMMmm  `7MM  `7MMpMMMb.   .P"Ybmmm ,pP"Ybd 
#  .M          (O)   j8       `YMMNq. ,M'   Yb   MM      MM      MM    MM    MM  :MI  I8   8I   `" 
#  |bMMAg.         ,;j9     .     `MM 8M""""""   MM      MM      MM    MM    MM   WmmmP"   `YMMMa. 
#       `Mb ,,  ,-='        Mb     dM YM.    ,   MM      MM      MM    MM    MM  8M        L.   I8 
#        jM db Ammmmmmm     P"Ybmmd"   `Mbmmd'   `Mbmo   `Mbmo .JMML..JMML  JMML. YMMMMMb  M9mmmP' 
#  (O)  ,M9                                                                      6'     dP         
#   6mmm9                                                                        Ybmmmd'           
#====================================================================================================#

# Step 5. Create NN model and Initiate
print("\n\n\n>>> Initializing the model... ")
#====================================================================================================#
# Get the model (CNN)

if log_value == True:
    y_tr = np.log2(y_tr)
    y_ts = np.log2(y_ts)
    y_va = np.log2(y_va)

train_seqs, train_vals = (X_tr_seqs, y_tr, )
valid_seqs, valid_vals = (X_va_seqs, y_va, )
test_seqs , test_vals  = (X_ts_seqs, y_ts, )


# Featurizer sequences with kmers
print("Featurizing sequences")
seqs_featurizer  = KMERFeaturizer(ngram_min=2, ngram_max=3, unnormalized=True)
train_seqs_feats = np.vstack(seqs_featurizer.featurize(train_seqs))
valid_seqs_feats = np.vstack(seqs_featurizer.featurize(valid_seqs))
test_seqs_feats  = np.vstack(seqs_featurizer.featurize(test_seqs ))

knn_params = {"n": 3, "seqs_dist_weight": 5, "comp_dist_weight": 1}


#====================================================================================================#
#    M******     pd""b.        MMP""MM""YMM              db              db                          #
#   .M          (O)  `8b       P'   MM   `7                                                          #
#   |bMMAg.          ,89            MM `7Mb,od8 ,6"Yb. `7MM `7MMpMMMb. `7MM `7MMpMMMb.  .P"Ybmmm     #
#        `Mb ,,    ""Yb.            MM   MM' "'8)   MM   MM   MM    MM   MM   MM    MM :MI  I8       #
#         jM db       88            MM   MM     ,pm9MM   MM   MM    MM   MM   MM    MM  WmmmP"       #
#   (O)  ,M9    (O)  .M'            MM   MM    8M   MM   MM   MM    MM   MM   MM    MM 8M            #
#    6mmm9       bmmmd'           .JMML.JMML.  `Moo9^Yo.JMML.JMML  JMML.JMML.JMML  JMML.YMMMMMb      #
#                                                                                      6'     dP     #
#                                                                                      Ybmmmd'       #
#====================================================================================================#



knn_model = KNNModel(**knn_params)
knn_model.fit(
                train_seqs_feats ,
                train_vals       ,
                valid_seqs_feats ,
                valid_vals       ,
                )


# Get Test Results
inds = np.arange(len(test_seqs_feats))
num_splits = min(100, len(inds))
ars = np.array_split(inds, num_splits)
ar_vec = []
for ar in tqdm(ars):
    test_preds = knn_model.predict(test_seqs_feats[ar], )
    ar_vec.append(test_preds)
test_preds = np.concatenate(ar_vec)


print("Conducting evaluation")
y_real = test_vals
y_pred = test_preds
if log_value == True:
    y_real = np.log10(np.power(2, test_vals))
    y_pred = np.log10(np.power(2, test_preds))


# Get Validation Results
inds = np.arange(len(valid_seqs_feats))
num_splits = min(100, len(inds))
ars = np.array_split(inds, num_splits)
ar_vec = []
for ar in tqdm(ars):
    valid_preds = knn_model.predict(valid_seqs_feats[ar], )
    ar_vec.append(valid_preds)
valid_preds = np.concatenate(ar_vec)


print("Conducting evaluation")
y_real_va = valid_vals
y_pred_va = valid_preds

if log_value == True:
    y_real_va = np.log10(np.power(2, valid_vals))
    y_pred_va = np.log10(np.power(2, valid_preds))


print("len(y_pred): ", len(y_pred))
print("len(y_pred_va): ", len(y_pred_va))

#====================================================================================================#
#   M******         ,AM      `7MM"""YMM                 `7MM                    mm           
#  .M              AVMM        MM    `7                   MM                    MM           
#  |bMMAg.       ,W' MM        MM   d `7M'   `MF',6"Yb.   MM `7MM  `7MM  ,6"YbmmMMmm .gP"Ya  
#       `Mb ,, ,W'   MM        MMmmMM   VA   ,V 8)   MM   MM   MM    MM 8)   MM MM  ,M'   Yb 
#        jM db AmmmmmMMmm      MM   Y  , VA ,V   ,pm9MM   MM   MM    MM  ,pm9MM MM  8M"""""" 
#  (O)  ,M9          MM        MM     ,M  VVV   8M   MM   MM   MM    MM 8M   MM MM  YM.    , 
#   6mmm9            MM      .JMMmmmmMMM   W    `Moo9^Yo.JMML. `Mbod"YML`Moo9^Yo`Mbmo`Mbmmd' 
#====================================================================================================#

epoch  = 0
loss_train = 0

slope, intercept, r_value_va, p_value, std_err = scipy.stats.linregress(y_pred_va, y_real_va)

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_pred, y_real)



slope, intercept, r_value_va, p_value, std_err = scipy.stats.linregress(y_pred_va, y_real_va)

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_pred, y_real)

va_MAE  = np.round(mean_absolute_error(y_pred_va, y_real_va), 4)
va_MSE  = np.round(mean_squared_error (y_pred_va, y_real_va), 4)
va_RMSE = np.round(math.sqrt(va_MSE), 4)
va_R2   = np.round(r2_score(y_real_va, y_pred_va), 4)
va_rho  = np.round(scipy.stats.spearmanr(y_pred_va, y_real_va)[0], 4)


ts_MAE  = np.round(mean_absolute_error(y_pred, y_real), 4)
ts_MSE  = np.round(mean_squared_error (y_pred, y_real), 4)
ts_RMSE = np.round(math.sqrt(ts_MSE), 4) 
ts_R2   = np.round(r2_score(y_real, y_pred), 4)
ts_rho  = np.round(scipy.stats.spearmanr(y_pred, y_real)[0], 4)


print("_" * 101)
print("\nepoch: {} | time_elapsed: {:5.4f} | train_loss: {:5.4f} | vali_R_VALUE: {:5.4f} | test_R_VALUE: {:5.4f} ".format( 
        str((epoch+1)+1000).replace("1","",1), 
        np.round((time.time()-time.time()), 5),
        np.round(loss_train, 5), 
        np.round(r_value_va, 5), 
        np.round(r_value, 5),
        )
        )


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


#====================================================================================================#
#   M******     M******   `7MM"""Mq. `7MM              mm    
#  .M          .M           MM   `MM.  MM              MM    
#  |bMMAg.     |bMMAg.      MM   ,M9   MM   ,pW"Wq.  mmMMmm  
#       `Mb ,,      `Mb     MMmmdM9    MM  6W'   `Wb   MM    
#        jM db       jM     MM         MM  8M     M8   MM    
#  (O)  ,M9    (O)  ,M9     MM         MM  YA.   ,A9   MM    
#   6mmm9       6mmm9     .JMML.     .JMML. `Ybmd9'    `Mbmo 
#====================================================================================================#

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
                                                    "\nEpoch: " + "N/A",
                        x_label         =  "Actual Values",
                        y_label         =  "Predictions",
                        cmap            =  None,
                        font_size       =  18,
                        result_folder   =  results_sub_folder,
                        file_name       =  "TS" + output_file_header,
                        ) #For checking predictions fittings.




_, _, r_value, _ , _ = scipy.stats.linregress(y_pred_va, y_real_va)

reg_scatter_distn_plot(y_pred_va,
                        y_real_va,
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
                                                    "\nEpoch: " + "N/A",
                        x_label         =  "Actual Values",
                        y_label         =  "Predictions",
                        cmap            =  None,
                        font_size       =  18,
                        result_folder   =  results_sub_folder,
                        file_name       =  "VA" + output_file_header,
                        ) #For checking predictions fittings.                            

#====================================================================================================#
if log_value == False and screen_bool==True:
    y_real = np.delete(y_real, np.where(y_pred == 0.0))
    y_pred = np.delete(y_pred, np.where(y_pred == 0.0))
    y_real = np.log10(y_real)
    y_pred = np.log10(y_pred)
    
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
                                                    "\nEpoch: " + "N/A",
                            x_label         =  "Actual Values",
                            y_label         =  "Predictions",
                            cmap            =  None,
                            font_size       =  18,
                            result_folder   =  results_sub_folder,
                            file_name       =  output_file_header + "_logplot"
                            ) #For checking predictions fittings.
#########################################################################################################
#########################################################################################################



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




