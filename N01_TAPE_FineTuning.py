#!/usr/bin/env python
# coding: utf-8
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
###################################################################################################################
###################################################################################################################
import sys
import time
import torch
import numpy as ny
import pandas as pd
import argparse
#--------------------------------------------------#
from pathlib import Path
from torch import nn
from torch.utils import data as data
import torch.optim as optim
#--------------------------------------------------#
from tape import datasets
from tape.datasets import *
#--------------------------------------------------#
from Z01_ModifiedModels import *
#====================================================================================================#
from tape import ProteinBertModel, TAPETokenizer, ProteinBertForMaskedLM
model = ProteinBertForMaskedLM.from_pretrained('bert-base')
tokenizer = TAPETokenizer(vocab='iupac')  # iupac is the vocab for TAPE models, use unirep for the UniRep model
#from tape import UniRepModel, UniRepForLM
###################################################################################################################
###################################################################################################################
def X01_TAPE_Fine_Tuning(data_folder, 
                         datafile_name, 
                         batch_size, 
                         epoch_num, 
                         intermediate_savings, 
                         final_saving, 
                         datafile_FT, 
                         datafile_Va, 
                         seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #====================================================================================================#
    data_file = data_folder / datafile_name
    embed = MLMD(data_path = data_file,tokenizer="iupac")
    print("Dataset Size: ", embed.__len__(), "[seqs]")
    Loader = data.DataLoader(embed,batch_size,True,collate_fn = embed.collate_fn)
    #====================================================================================================#
    embed_FT=MLMD(data_path = data_folder / datafile_FT, tokenizer="iupac")
    embed_Va=MLMD(data_path = data_folder / datafile_Va, tokenizer="iupac")
    Fine_Tuning_Loader = data.DataLoader(embed_FT, batch_size, True, collate_fn = embed_FT.collate_fn)
    Validation_Loader = data.DataLoader(embed_Va, min(embed_Va.__len__(),512), True, collate_fn = embed_Va.collate_fn) #min(embed_Va.__len__(),512)
    print("Validation Loader Length: ", len(Validation_Loader))

    ###################################################################################################################
    ###################################################################################################################
    model.cuda()
    optimizer = optim.Adam(model.parameters(),lr=0.00001)
    model.train()
    scaler = torch.cuda.amp.GradScaler() # (Mixed Precision Training)
    #====================================================================================================#
    # TAPE fine-tuning

    count_x=0
    start_time = time.time()
    for epoch in range(epoch_num):
        count_x+=1
        print("="*30, " Epoch #", str(count_x+1000)[-2:], "="*30)
        for seq_batch in Fine_Tuning_Loader:
            input_ids, input_mask, targets = seq_batch['input_ids'], seq_batch['input_mask'], seq_batch['targets']
            input_ids, input_mask, targets = input_ids.cuda(), input_mask.cuda(), targets.cuda()
        
            optimizer.zero_grad()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            with torch.cuda.amp.autocast(): # (Mixed Precision Training)
                outputs = model(input_ids,input_mask)
                #print(outputs[0].size())
                #print(input_ids.size())
                lmloss = loss_fct(outputs[0].reshape(-1,30),targets.reshape(-1))
            scaler.scale(lmloss).backward() # (Mixed Precision Training)
            scaler.step(optimizer) # (Mixed Precision Training)
            scaler.update() # (Mixed Precision Training)
        if (epoch+1)%5 == 0:
            intermediate_saving_name = intermediate_savings + "_epoch" + str(epoch+1) + "_trial_training.pt"
            intermediate_saving_path = data_folder / intermediate_saving_name
            torch.save({'epoch': count_x, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': lmloss}, intermediate_saving_path)
            print("epoch: %s |Time Elapsed : %s seconds " % (count_x, time.time() - start_time, ))
            model.eval()
            validation_loss_set = []
            for seq_batch in Validation_Loader:
                input_ids, input_mask, targets = seq_batch['input_ids'], seq_batch['input_mask'], seq_batch['targets']
                input_ids, input_mask, targets = input_ids.cuda(), input_mask.cuda(), targets.cuda()
                with torch.no_grad():
                    outputs = model(input_ids)
                validation_loss = loss_fct(outputs[0].reshape(-1,30),input_ids.reshape(-1))
                validation_loss_set.append(validation_loss.view(-1,1))
            validation_loss = torch.mean(torch.cat(validation_loss_set))
            print("epoch: {} |Loss: {} | Validaiton Loss: {}".format(count_x,lmloss,validation_loss))
            model.train()
        #print(torch.cuda.memory_summary(device=None, abbreviated=False))
        if (epoch+1)%10 == 0:
            print("\n")

    final_saving_name = final_saving+"_Result.pt"
    final_saving_path = data_folder / final_saving_name
    torch.save({'epoch': epoch+1,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': lmloss}, final_saving_path)
    
    return

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

if __name__ == "__main__":
    ###################################################################################################################
    ###################################################################################################################
    # Args
    #--------------------------------------------------#
    # Inputs
    Step_code = "X01_"
    dataset_nme_list = ["phosphatase",        # 0
                        "kinase",             # 1
                        "esterase",           # 2
                        "halogenase",         # 3
                        "aminotransferase",   # 4

                         "kcat",               # 5
                         "kcat_mt",            # 6
                         "kcat_wt",            # 7

                         "KiCore",           # 8
                         "KiScrn",           # 9
                         "KiFine",           # 10
                        
                         "KmCore",           # 11
                         "KmScrn",           # 12
                         "KmFine",           # 13

                         "KcatCore",         # 14
                         "KcatScrn",         # 15
                         "KcatFine",         # 16

                         "KcatKmCore",       # 17
                         "KcatKmScrn",       # 18
                         "KcatKmFine",       # 19
                         ]

    dataset_nme      = dataset_nme_list[19]
    data_folder   = Path("X_DataProcessing/")
    datafile_name = "X00_" + dataset_nme + ".fasta"
    #--------------------------------------------------#
    # Inputs
    datafile_FT = "X00_" + dataset_nme + "_FT_FT.fasta" # Fine-Tuning
    datafile_Va = "X00_" + dataset_nme + "_FT_Va.fasta" # Validation
    #--------------------------------------------------#
    # Outputs
    inter_saving="X01_" + dataset_nme + "_FT_inter"
    final_saving="X01_" + dataset_nme + "_FT_final"
    #--------------------------------------------------#
    # Settings
    seed       = 42
    batch_size = 6
    epoch_num  = 100
    ###################################################################################################################
    ###################################################################################################################
    # ArgParser
    parser = argparse.ArgumentParser(
        description="Preprocesses the sequence datafile.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(      "--data_folder"          , type = Path , default = data_folder   , help = "Path to the directory containing your datasets." )
    parser.add_argument(      "--datafile_name"        , type = str  , default = datafile_name , help = "Name of your datasets."                          )
    parser.add_argument("-b", "--batch_size"           , type = int  , default = batch_size    , help = "Batch size."                                     )
    parser.add_argument("-e", "--epoch_num"            , type = int  , default = epoch_num     , help = "Epoch number."                                   )
    parser.add_argument(      "--intermediate_savings" , type = str  , default = inter_saving  , help = "Name of your datasets."                          )
    parser.add_argument(      "--final_saving"         , type = str  , default = final_saving  , help = "Name of your datasets."                          )
    parser.add_argument(      "--datafile_FT"          , type = str  , default = datafile_FT   , help = "Name of your datasets."                          )
    parser.add_argument(      "--datafile_Va"          , type = str  , default = datafile_Va   , help = "Name of your datasets."                          )
    parser.add_argument("-s", "--seed"                 , type = int  , default = 42            , help = "Random seed."                                    )

    args = parser.parse_args()
    #====================================================================================================#
    # Main
    X01_TAPE_Fine_Tuning(**vars(args))

    print("*" * 50)
    print(Step_code + " Done!")




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