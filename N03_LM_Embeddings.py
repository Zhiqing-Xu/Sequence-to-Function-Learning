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
import sys
import time
import torch
import numpy as np
import pandas as pd
import pickle
import argparse
import requests
import subprocess
#--------------------------------------------------#
from torch import nn
from torch.utils import data as data
#--------------------------------------------------#
from tape import datasets
from tape import TAPETokenizer
from tape import ProteinBertForMaskedLM
from tape import UniRepForLM
#--------------------------------------------------#
#from Z01_ModifiedModels import *
from pathlib import Path
#--------------------------------------------------#
from Bio import SeqIO
from tqdm.auto import tqdm
#====================================================================================================#
# Imports for LM
from transformers import BertModel, BertTokenizer
from transformers import AlbertModel, AlbertTokenizer
from transformers import ElectraTokenizer, ElectraForPreTraining, ElectraForMaskedLM, ElectraModel
from transformers import T5EncoderModel, T5Tokenizer
from transformers import XLNetModel, XLNetTokenizer
#--------------------------------------------------#
import esm
#--------------------------------------------------#
import ankh
#--------------------------------------------------#
from sequence_models.pretrained import load_model_and_alphabet
#====================================================================================================#
# Imports for MSA
from glob import glob
from Bio.Align.Applications import MafftCommandline

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
class LoaderClass(data.Dataset):
    def __init__(self, input_ids, attention_mask):
        super().__init__()
        self.input_ids = input_ids
        self.attention_mask = attention_mask
    def __len__(self):
        return self.input_ids.shape[0]
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx]
#====================================================================================================#
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x,target = None):
        return (x,)

#====================================================================================================#





#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#      `7MM"""YMM  `7MMM.     ,MMF'`7MM"""Yp,     `7MM"""YMM `7MMF'   `7MF'`7MN.   `7MF'  .g8"""bgd                                                    #
#        MM    `7    MMMb    dPMM    MM    Yb       MM    `7   MM       M    MMN.    M  .dP'     `M                                                    #
#        MM   d      M YM   ,M MM    MM    dP       MM   d     MM       M    M YMb   M  dM'       `                                                    #
#        MMmmMM      M  Mb  M' MM    MM"""bg.       MM""MM     MM       M    M  `MN. M  MM                                                             #
#        MM   Y  ,   M  YM.P'  MM    MM    `Y       MM   Y     MM       M    M   `MM.M  MM.                                                            #
#        MM     ,M   M  `YM'   MM    MM    ,9       MM         YM.     ,M    M     YMM  `Mb.     ,'                                                    #
#      .JMMmmmmMMM .JML. `'  .JMML..JMMmmmd9      .JMML.        `bmmmmd"'  .JML.    YM    `"bmmmd'                                                     #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
def N03_embedding_LM(dataset_nme,
                     model_select, 
                     data_folder ,
                     input_seqs_fasta_file, 
                     output_file_name_header, 
                     pretraining_name=None, 
                     batch_size=100, 
                     xlnet_mem_len=512):

    #====================================================================================================#
    assert model_select in models_list, "query model is not found, currently support ESM-1b, TAPE, BERT, AlBERT, Electra, T5 and Xlnet !!"
    #====================================================================================================#
    input_file = data_folder / input_seqs_fasta_file #data path (fasta)
    output_file = data_folder / (output_file_name_header + model_select + ".p") #output path (pickle)




    ###################################################################################################################
    #                       `7MMF'   `7MF'`7MN.   `7MF'`7MMF'`7MM"""Mq.  `7MM"""YMM  `7MM"""Mq.                       #
    #                         MM       M    MMN.    M    MM    MM   `MM.   MM    `7    MM   `MM.                      #
    #                         MM       M    M YMb   M    MM    MM   ,M9    MM   d      MM   ,M9                       #
    #                         MM       M    M  `MN. M    MM    MMmmdM9     MMmmMM      MMmmdM9                        #
    #                         MM       M    M   `MM.M    MM    MM  YM.     MM   Y  ,   MM                             #
    #                         YM.     ,M    M     YMM    MM    MM   `Mb.   MM     ,M   MM                             #
    #                          `bmmmmd"'  .JML.    YM  .JMML..JMML. .JMM..JMMmmmmMMM .JMML.                           #
    ###################################################################################################################
    if model_select == "Unirep" or model_select == "Unirep_FT":
        #--------------------------------------------------#
        # Load model.
        model = UniRepForLM.from_pretrained('babbler-1900')
        #--------------------------------------------------#
        if model_select == "Unirep":
            print("No pretraining model is used.")
            pretraining_name = None
        #--------------------------------------------------#
        if pretraining_name is not None:
            checkpoint = torch.load( data_folder / pretraining_name )
            model.load_state_dict(checkpoint['model_state_dict'])
        #--------------------------------------------------#
        model.mlm = Identity()
        model.eval()
        embed  = datasets.EmbedDataset(data_file = input_file, tokenizer = "unirep")
        loader = data.DataLoader(embed, batch_size, False, collate_fn = embed.collate_fn)
        #--------------------------------------------------#
        count_x = 0
        model.cuda()
        #--------------------------------------------------#
        seq_encodings = []
        seq_all_hiddens = []
        seq_ids = []
        for seq_batch in loader:
            count_x += 1
            ids, input_ids, input_mask = seq_batch["ids"],seq_batch["input_ids"],seq_batch["input_mask"]
            input_ids, input_mask = input_ids.cuda(), input_mask.cuda()
            with torch.no_grad():
                output = model(input_ids,input_mask)
            output = output[0].cpu().detach().numpy()
            features = [] 
            for seq_num in range(len(output)):
                seq_len = (input_mask[seq_num] == 1).sum()
                seq_emd = output[seq_num][1:seq_len-1]
                #print(seq_emd)
                seq_all_hiddens.append(seq_emd)
                features.append(np.mean(seq_emd,axis=0))
            features = np.stack(features)
            print("features.shape: ", features.shape)
            seq_encodings.append(features)
            seq_ids += ids
        seq_embeddings = np.concatenate(seq_encodings)
        print("seq_embeddings.shape: ", seq_embeddings.shape)
        seq_embedding_output = {"seq_embeddings" : seq_embeddings, "seq_ids" : seq_ids, "seq_all_hiddens" : seq_all_hiddens}
        pickle.dump( seq_embedding_output, open( output_file, "wb" ) )
        print("done")


    ###################################################################################################################
    #                               MMP""MM""YMM       db      `7MM"""Mq. `7MM"""YMM                                  #
    #                               P'   MM   `7      ;MM:       MM   `MM.  MM    `7                                  #
    #                                    MM          ,V^MM.      MM   ,M9   MM   d                                    #
    #                                    MM         ,M  `MM      MMmmdM9    MMmmMM                                    #
    #                                    MM         AbmmmqMA     MM         MM   Y  ,                                 #
    #                                    MM        A'     VML    MM         MM     ,M                                 #
    #                                  .JMML.    .AMA.   .AMMA..JMML.     .JMMmmmmMMM                                 #
    ###################################################################################################################
    if model_select == "TAPE_FT" or model_select == "TAPE":
        model = ProteinBertForMaskedLM.from_pretrained('bert-base')
        #--------------------------------------------------#
        if model_select == "TAPE":
            pretraining_name == None
        #--------------------------------------------------#
        if pretraining_name is not None:
            checkpoint = torch.load( data_folder / pretraining_name )
            model.load_state_dict(checkpoint['model_state_dict'])
        #--------------------------------------------------#
        model.mlm = Identity()
        model.eval()
        embed = datasets.EmbedDataset(data_file=input_file,tokenizer="iupac")
        loader = data.DataLoader(embed,batch_size,False,collate_fn = embed.collate_fn)
        #--------------------------------------------------#
        count_x = 0
        model.cuda()
        #--------------------------------------------------#
        seq_encodings = []
        seq_all_hiddens = []
        seq_ids = []
        for seq_batch in loader:
            count_x+=1
            ids, input_ids, input_mask = seq_batch["ids"],seq_batch["input_ids"],seq_batch["input_mask"]
            input_ids, input_mask = input_ids.cuda(), input_mask.cuda()
            with torch.no_grad():
                output = model(input_ids,input_mask)
            output = output[0].cpu().detach().numpy()
            features = [] 
            for seq_num in range(len(output)):
                seq_len = (input_mask[seq_num] == 1).sum()
                seq_emd = output[seq_num][1:seq_len-1]
                #print(seq_emd)
                seq_all_hiddens.append(seq_emd)
                features.append(np.mean(seq_emd,axis=0))
            features = np.stack(features)
            print("features.shape: ", features.shape)
            seq_encodings.append(features)
            seq_ids += ids
        seq_embeddings = np.concatenate(seq_encodings)
        print("seq_embeddings.shape: ", seq_embeddings.shape)
        seq_embedding_output = {"seq_embeddings":seq_embeddings, "seq_ids":seq_ids, "seq_all_hiddens":seq_all_hiddens}
        pickle.dump( seq_embedding_output, open( output_file, "wb" ) )
        print("done")




    ###################################################################################################################
    #                               `7MM"""Yp, `7MM"""YMM  `7MM"""Mq.  MMP""MM""YMM                                   #
    #                                 MM    Yb   MM    `7    MM   `MM. P'   MM   `7                                   #
    #                                 MM    dP   MM   d      MM   ,M9       MM                                        #
    #                                 MM"""bg.   MMmmMM      MMmmdM9        MM                                        #
    #                                 MM    `Y   MM   Y  ,   MM  YM.        MM                                        #
    #                                 MM    ,9   MM     ,M   MM   `Mb.      MM                                        #
    #                               .JMMmmmd9  .JMMmmmmMMM .JMML. .JMM.   .JMML.                                      #
    ###################################################################################################################
    if model_select == "BERT":
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        model = BertModel.from_pretrained("Rostlab/prot_bert")
        training_set = []
        seq_ids = []
        for seq_record in SeqIO.parse(input_file, "fasta"):
            seq = str(seq_record.seq)
            new_string = ""
            for i in range(len(seq)-1):
                new_string += seq[i]
                new_string += " "
            new_string += seq[-1]
            seq_ids.append(str(seq_record.id))
            training_set.append(new_string)
        ids = tokenizer.batch_encode_plus(training_set, add_special_tokens=True, padding=True)
        #--------------------------------------------------#
        loader = data.DataLoader(LoaderClass(np.array(ids["input_ids"]), np.array(ids["attention_mask"])), batch_size, False)
        model.cuda()
        model.eval()
        seq_encodings = []
        seq_all_hiddens = []
        count = 0
        #--------------------------------------------------#
        for seq_batch in loader:
            count+=1
            input_ids, input_mask = seq_batch[0],seq_batch[1]
            input_ids, input_mask = input_ids.cuda(), input_mask.cuda()
            with torch.no_grad():
                output = model(input_ids,input_mask)
            output = output[0].cpu().detach().numpy()
            features = []
            for seq_num in range(len(output)):
                seq_len = (input_mask[seq_num] == 1).sum()
                seq_emd = output[seq_num][1:seq_len-1]
                seq_all_hiddens.append(seq_emd)
                features.append(np.mean(seq_emd,axis=0))
            features = np.stack(features)
            print(features.shape)
            seq_encodings.append(features)
        seq_embeddings = np.concatenate(seq_encodings)
        print("seq_embeddings.shape: ", seq_embeddings.shape)
        seq_embedding_output = {"seq_embeddings":seq_embeddings, "seq_ids":seq_ids, "seq_all_hiddens":seq_all_hiddens}
        pickle.dump( seq_embedding_output, open( output_file, "wb" ) )
        print("done")







    ###################################################################################################################
    #                         db      `7MMF'      `7MM"""Yp, `7MM"""YMM  `7MM"""Mq.  MMP""MM""YMM                     #
    #                        ;MM:       MM          MM    Yb   MM    `7    MM   `MM. P'   MM   `7                     #
    #                       ,V^MM.      MM          MM    dP   MM   d      MM   ,M9       MM                          #
    #                      ,M  `MM      MM          MM"""bg.   MMmmMM      MMmmdM9        MM                          #
    #                      AbmmmqMA     MM      ,   MM    `Y   MM   Y  ,   MM  YM.        MM                          #
    #                     A'     VML    MM     ,M   MM    ,9   MM     ,M   MM   `Mb.      MM                          #
    #                   .AMA.   .AMMA..JMMmmmmMMM .JMMmmmd9  .JMMmmmmMMM .JMML. .JMM.   .JMML.                        #
    ###################################################################################################################
    if model_select == "ALBERT":
        tokenizer = AlbertTokenizer.from_pretrained("Rostlab/prot_albert", do_lower_case=False)
        model = AlbertModel.from_pretrained("Rostlab/prot_albert")
        training_set = []
        seq_ids = []
        for seq_record in SeqIO.parse(input_file, "fasta"):
            seq = str(seq_record.seq)
            new_string = ""
            for i in range(len(seq)-1):
                new_string += seq[i]
                new_string += " "
            new_string += seq[-1]
            seq_ids.append(str(seq_record.id))
            training_set.append(new_string)
        ids = tokenizer.batch_encode_plus(training_set, add_special_tokens=True, padding=True)
        #--------------------------------------------------#
        loader = data.DataLoader(LoaderClass(np.array(ids["input_ids"]), np.array(ids["attention_mask"])), batch_size, False)
        model.cuda()
        model.eval()
        seq_encodings = []
        seq_all_hiddens = []
        count = 0
        #--------------------------------------------------#
        for seq_batch in loader:
            count+=1
            input_ids, input_mask = seq_batch[0],seq_batch[1]
            input_ids, input_mask = input_ids.cuda(), input_mask.cuda()
            with torch.no_grad():
                output = model(input_ids,input_mask)
            output = output[0].cpu().detach().numpy()
            features = [] 
            for seq_num in range(len(output)):
                seq_len = (input_mask[seq_num] == 1).sum()
                seq_emd = output[seq_num][1:seq_len-1]
                seq_all_hiddens.append(seq_emd)
                features.append(np.mean(seq_emd,axis=0))
            features = np.stack(features)
            print(features.shape)
            seq_encodings.append(features)
        seq_embeddings = np.concatenate(seq_encodings)
        print("seq_embeddings.shape: ", seq_embeddings.shape)
        seq_embedding_output = {"seq_embeddings":seq_embeddings, "seq_ids":seq_ids, "seq_all_hiddens":seq_all_hiddens}
        pickle.dump( seq_embedding_output, open( output_file, "wb" ) )
        print("done")



    ###################################################################################################################
    #         `7MM"""YMM  `7MM  `7MM"""YMM    .g8"""bgd MMP""MM""YMM `7MM"""Mq.        db                .g8""8q.     #
    #           MM    `7    MM    MM    `7  .dP'     `M P'   MM   `7   MM   `MM.      ;MM:             .dP      YM.   #
    #           MM   d      MM    MM   d    dM'       `      MM        MM   ,M9      ,V^MM.            dM'    ,V`MM   #
    #           MMmmMM      MM    MMmmMM    MM               MM        MMmmdM9      ,M  `MM            MM   ,M'  MM   #
    #           MM   Y  ,   MM    MM   Y  , MM.              MM        MM  YM.      AbmmmqMA           MM.,V'   ,MP   #
    #           MM     ,M   MM    MM     ,M `Mb.     ,'      MM        MM   `Mb.   A'     VML          `Mb     ,dP'   #
    #         .JMMmmmmMMM .JMML..JMMmmmmMMM   `"bmmmd'     .JMML.    .JMML. .JMM..AMA.   .AMMA.          `"bmmd"'     #
    ###################################################################################################################
    # A now deprecated model.
    if model_select == "Electra": ##### !!!!! Deprecated!
        generatorModelUrl       =  'https://www.dropbox.com/s/5x5et5q84y3r01m/pytorch_model.bin?dl=1'
        discriminatorModelUrl   =  'https://www.dropbox.com/s/9ptrgtc8ranf0pa/pytorch_model.bin?dl=1'
        generatorConfigUrl      =  'https://www.dropbox.com/s/9059fvix18i6why/config.json?dl=1'
        discriminatorConfigUrl  =  'https://www.dropbox.com/s/jq568evzexyla0p/config.json?dl=1'
        vocabUrl                =  'https://www.dropbox.com/s/wck3w1q15bc53s0/vocab.txt?dl=1'
        downloadFolderPath      =  'models/electra/'

        discriminatorFolderPath     =  os.path.join(downloadFolderPath, 'discriminator')
        generatorFolderPath         =  os.path.join(downloadFolderPath, 'generator')
        discriminatorModelFilePath  =  os.path.join(discriminatorFolderPath, 'pytorch_model.bin')
        generatorModelFilePath      =  os.path.join(generatorFolderPath, 'pytorch_model.bin')
        discriminatorConfigFilePath =  os.path.join(discriminatorFolderPath, 'config.json')
        generatorConfigFilePath     =  os.path.join(generatorFolderPath, 'config.json')
        vocabFilePath               =  os.path.join(downloadFolderPath, 'vocab.txt')

        if not os.path.exists(discriminatorFolderPath):
            os.makedirs(discriminatorFolderPath)
        if not os.path.exists(generatorFolderPath):
            os.makedirs(generatorFolderPath)
        def download_file(url, filename):
            response = requests.get(url, stream=True)
            with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                                total=int(response.headers.get('content-length', 0)),
                                desc=filename) as fout:
                for chunk in response.iter_content(chunk_size=4096):
                    fout.write(chunk)
        if not os.path.exists(generatorModelFilePath):
            download_file(generatorModelUrl, generatorModelFilePath)
        if not os.path.exists(discriminatorModelFilePath):
            download_file(discriminatorModelUrl, discriminatorModelFilePath)
        if not os.path.exists(generatorConfigFilePath):
            download_file(generatorConfigUrl, generatorConfigFilePath)
        if not os.path.exists(discriminatorConfigFilePath):
            download_file(discriminatorConfigUrl, discriminatorConfigFilePath)
        if not os.path.exists(vocabFilePath):
            download_file(vocabUrl, vocabFilePath)

        tokenizer = ElectraTokenizer(vocabFilePath, do_lower_case=False )
        model = ElectraModel.from_pretrained(discriminatorFolderPath)
        model.cuda()
        model.eval()    
        training_set = []
        seq_ids = []
        for seq_record in SeqIO.parse(input_file, "fasta"):
            seq = str(seq_record.seq)
            new_string = ""
            for i in range(len(seq)-1):
                new_string += seq[i]
                new_string += " "
            new_string += seq[-1]
            seq_ids.append(str(seq_record.id))
            training_set.append(new_string)
        ids = tokenizer.batch_encode_plus(training_set, add_special_tokens=True, padding=True)
        loader = data.DataLoader(LoaderClass(np.array(ids["input_ids"]), np.array(ids["attention_mask"])), batch_size, False)
        seq_encodings = []
        seq_all_hiddens = []
        count = 0
        for seq_batch in loader:
            count+=1
            input_ids, input_mask = seq_batch[0],seq_batch[1]
            input_ids, input_mask = input_ids.cuda(), input_mask.cuda()
            with torch.no_grad():
                output = model(input_ids,input_mask)
            output = output[0].cpu().detach().numpy()
            features = [] 
            for seq_num in range(len(output)):
                seq_len = (input_mask[seq_num] == 1).sum()
                seq_emd = output[seq_num][1:seq_len-1]
                seq_all_hiddens.append(seq_emd)
                features.append(np.mean(seq_emd,axis=0))
            features = np.stack(features)
            print(features.shape)
            seq_encodings.append(features)
        seq_embeddings = np.concatenate(seq_encodings)
        print("seq_embeddings.shape: ", seq_embeddings.shape)
        seq_embedding_output = {"seq_embeddings":seq_embeddings, "seq_ids":seq_ids, "seq_all_hiddens":seq_all_hiddens}
        pickle.dump( seq_embedding_output, open( output_file, "wb" ) )
        print("done")



    ###################################################################################################################
    #                                             MMP""MM""YMM   M******                                              #
    #                                             P'   MM   `7  .M                                                    #
    #                                                  MM       |bMMAg.                                               #
    #                                                  MM            `Mb                                              #
    #                                                  MM             jM                                              #
    #                                                  MM       (O)  ,M9                                              #
    #                                                .JMML.      6mmm9                                                #
    ###################################################################################################################
    if model_select == "T5":
        
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case = False )
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

        print("123")

        model.cuda()
        model.eval()    
        training_set = []
        seq_ids = []
        for seq_record in SeqIO.parse(input_file, "fasta"):
            seq = str(seq_record.seq)
            new_string = ""
            for i in range(len(seq)-1):
                new_string += seq[i]
                new_string += " "
            new_string += seq[-1]
            seq_ids.append(str(seq_record.id))
            training_set.append(new_string)
        ids = tokenizer.batch_encode_plus(training_set, add_special_tokens = True, padding = True)
        loader = data.DataLoader(LoaderClass(np.array(ids["input_ids"]), np.array(ids["attention_mask"])), batch_size, False)
        seq_encodings = []
        seq_all_hiddens = []
        count = 0
        for seq_batch in loader:
            count += 1
            input_ids, input_mask = seq_batch[0], seq_batch[1]
            input_ids, input_mask = input_ids.cuda(), input_mask.cuda()
            with torch.no_grad():
                output = model(input_ids,input_mask)
            output = output[0].cpu().detach().numpy()
            features = [] 
            for seq_num in range(len(output)):
                seq_len = (input_mask[seq_num] == 1).sum()
                seq_emd = output[seq_num][:seq_len-1]
                seq_all_hiddens.append(seq_emd)
                features.append(np.mean(seq_emd, axis=0))
            features = np.stack(features)
            print(features.shape)
            seq_encodings.append(features)
        seq_embeddings = np.concatenate(seq_encodings)
        print("seq_embeddings.shape: ", seq_embeddings.shape)
        seq_embedding_output = {"seq_embeddings": seq_embeddings, "seq_ids": seq_ids, "seq_all_hiddens": seq_all_hiddens}
        pickle.dump( seq_embedding_output, open( output_file, "wb" ) )
        print("done")

    ###################################################################################################################
    #                                    MMP""MM""YMM   M****** `YMM'   `MP' `7MMF'                                   #
    #                                    P'   MM   `7  .M         VMb.  ,P     MM                                     #
    #                                         MM       |bMMAg.     `MM.M'      MM                                     #
    #                                         MM            `Mb      MMb       MM                                     #
    #                                         MM             jM    ,M'`Mb.     MM      ,                              #
    #                                         MM       (O)  ,M9   ,P   `MM.    MM     ,M                              #
    #                                       .JMML.      6mmm9   .MM:.  .:MMa..JMMmmmmMMM                              #
    ###################################################################################################################
    if model_select == "T5XL":
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case = False )
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

        data_set = []
        for seq_record in SeqIO.parse(input_file, "fasta"):
            data_set.append((str(seq_record.id), str(seq_record.seq)))
        chunk_size = 1000
        data_set_list = [data_set[i:i + chunk_size] for i in range(0, len(data_set), chunk_size)]

        for data_set_id, one_data_set in enumerate(data_set_list):
            model.eval()
            model.cuda()
            seq_encodings   = []
            seq_all_hiddens = []
            seq_ids         = []
            sequences       = []  

            for i in range(0, len(one_data_set), batch_size):

                print(i, "out of", len(one_data_set), "; ",data_set_id + 1, "out of", len(data_set_list))

                batch = one_data_set[i : i + batch_size]
                batch_ids  = [p[0] for p in batch]
                batch_seqs = [p[1] for p in batch]

                seq_ids   += batch_ids
                sequences += batch_seqs

                # ProtT5 expects space‑separated residues
                spaced_seqs = [" ".join(list(s)) for s in batch_seqs]

                enc = tokenizer.batch_encode_plus(
                    spaced_seqs,
                    add_special_tokens=True,
                    padding=True,
                    return_tensors="pt",
                )

                input_ids      = enc["input_ids"].cuda()
                attention_mask = enc["attention_mask"].cuda()

                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                hidden = outputs.last_hidden_state.cpu().detach()   # [B, L, D]

                sequence_representations = []
                for j, seq in enumerate(batch_seqs):
                    # first token is <cls>, so skip it
                    seq_hidden = hidden[j, 1 : len(seq) + 1]

                    # per‑residue representations
                    seq_all_hiddens.append(seq_hidden.numpy())

                    # mean‑pooled per‑sequence embedding
                    sequence_representations.append(seq_hidden.mean(0))

                seq_encodings.append(np.stack(sequence_representations))

            # Save embeddings for this chunk
            seq_embeddings = np.concatenate(seq_encodings)
            print("seq_embeddings.shape:", seq_embeddings.shape)

            seq_embedding_output = {
                "seq_embeddings":  seq_embeddings,
                "seq_ids":         seq_ids,
                "seq_all_hiddens": seq_all_hiddens,
                "sequences":       sequences,
            }

            out_path = Path(str(output_file).replace(".p", f"_{data_set_id}.p"))
            with open(out_path, "wb") as h:
                pickle.dump(seq_embedding_output, h)

        print("done")



    ###################################################################################################################
    #                       `YMM'   `MP' `7MMF'      `7MN.   `7MF'`7MM"""YMM  MMP""MM""YMM                            #
    #                         VMb.  ,P     MM          MMN.    M    MM    `7  P'   MM   `7                            #
    #                          `MM.M'      MM          M YMb   M    MM   d         MM                                 #
    #                            MMb       MM          M  `MN. M    MMmmMM         MM                                 #
    #                          ,M'`Mb.     MM      ,   M   `MM.M    MM   Y  ,      MM                                 #
    #                         ,P   `MM.    MM     ,M   M     YMM    MM     ,M      MM                                 #
    #                       .MM:.  .:MMa..JMMmmmmMMM .JML.    YM  .JMMmmmmMMM    .JMML.                               #
    ###################################################################################################################
    if model_select == "Xlnet":
        tokenizer = XLNetTokenizer.from_pretrained("Rostlab/prot_xlnet", do_lower_case = False)
        model = XLNetModel.from_pretrained("Rostlab/prot_xlnet", mem_len = xlnet_mem_len)
        model.cuda()
        model.eval()
        training_set = []
        seq_ids = []
        for seq_record in SeqIO.parse(input_file, "fasta"):
            seq = str(seq_record.seq)
            new_string = ""
            for i in range(len(seq)-1):
                new_string += seq[i]
                new_string += " "
            new_string += seq[-1]
            seq_ids.append(str(seq_record.id))
            training_set.append(new_string)
        ids = tokenizer.batch_encode_plus(training_set, add_special_tokens = True, padding = True)
        loader = data.DataLoader(LoaderClass(np.array(ids["input_ids"]), np.array(ids["attention_mask"])), batch_size, False)
        seq_encodings = []
        seq_all_hiddens = []
        count = 0
        for seq_batch in loader:
            count += 1
            input_ids, input_mask = seq_batch[0],seq_batch[1]
            input_ids, input_mask = input_ids.cuda(), input_mask.cuda()
            with torch.no_grad():
                output = model(input_ids,input_mask)
            output = output[0].cpu().detach().numpy()
            features = [] 
            for seq_num in range(len(output)):
                seq_len = (input_mask[seq_num] == 1).sum()
                padded_seq_len = len(input_mask[seq_num])
                seq_emd = output[seq_num][padded_seq_len-seq_len:padded_seq_len-2]
                seq_all_hiddens.append(seq_emd)
                features.append(np.mean(seq_emd, axis=0))
            features = np.stack(features)
            print(features.shape)
            seq_encodings.append(features)
        seq_embeddings = np.concatenate(seq_encodings)
        print("seq_embeddings.shape: ", seq_embeddings.shape)
        seq_embedding_output = {"seq_embeddings": seq_embeddings, "seq_ids": seq_ids, "seq_all_hiddens": seq_all_hiddens}
        pickle.dump( seq_embedding_output, open( output_file, "wb" ) )
        print("done")



    ###################################################################################################################
    #                   `7MM"""YMM   .M"""bgd `7MMM.     ,MMF'                      *MM                               #
    #                     MM    `7  ,MI    "Y   MMMb    dPMM                 __,     MM                               #
    #                     MM   d    `MMb.       M YM   ,M MM                `7MM     MM,dMMb.                         #
    #                     MMmmMM      `YMMNq.   M  Mb  M' MM                  MM     MM    `Mb                        #
    #                     MM   Y  , .     `MM   M  YM.P'  MM      mmmmm       MM     MM     M8                        #
    #                     MM     ,M Mb     dM   M  `YM'   MM                  MM     MM.   ,M9                        #
    #                   .JMMmmmmMMM P"Ybmmd"  .JML. `'  .JMML.              .JMML.   P^YbmdP'                         #
    ###################################################################################################################
    if model_select == "ESM_1B":
        # model, alphabet = torch.hub.load("facebookresearch/esm", "esm1b_t33_650M_UR50S")
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        batch_converter = alphabet.get_batch_converter()

        # Load FASTA
        data_set = [(str(rec.id), str(rec.seq)) for rec in SeqIO.parse(input_file, "fasta")]

        # Split into chunks
        chunk_size    = 2000                
        data_set_list = [data_set[i : i + chunk_size] for i in range(0, len(data_set), chunk_size)]

        model.eval()
        model.cuda()

        for data_set_id, one_data_set in enumerate(data_set_list):
            seq_encodings, seq_all_hiddens, seq_ids, sequences = [], [], [], []

            for i in range(0, len(one_data_set), batch_size):
                print(i, "out of", len(one_data_set), "; ",
                    data_set_id + 1, "out of", len(data_set_list))

                batch = one_data_set[i : i + batch_size]
                batch_labels, batch_strs, batch_tokens = batch_converter(batch)
                batch_tokens = batch_tokens.cuda()

                seq_ids   += batch_labels
                sequences += batch_strs   # keep raw sequences

                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33])
                results = results["representations"][33].cpu().detach()

                # pool per‑residue embeddings → per‑sequence
                sequence_representations = []
                for j, (_, seq) in enumerate(batch):
                    seq_all_hiddens.append(results[j, 1 : len(seq) + 1].numpy())
                    sequence_representations.append(results[j, 1 : len(seq) + 1].mean(0))
                seq_encodings.append(np.stack(sequence_representations))

            # ------- save chunk -------
            seq_embeddings = np.concatenate(seq_encodings)
            print("seq_embeddings.shape:", seq_embeddings.shape)

            seq_embedding_output = {
                "seq_embeddings"  :   seq_embeddings , 
                "seq_ids"         :   seq_ids        , 
                "sequences"       :   sequences      , 
                "seq_all_hiddens" :  seq_all_hiddens , 
            }

            out_path = Path(str(output_file).replace(".p", f"_{data_set_id}.p"))
            with open(out_path, "wb") as fh:
                pickle.dump(seq_embedding_output, fh)

        print("done")



    ###################################################################################################################
    #                   `7MM"""YMM   .M"""bgd `7MMM.     ,MMF'                                                        #
    #                     MM    `7  ,MI    "Y   MMMb    dPMM                  __,                                     #
    #                     MM   d    `MMb.       M YM   ,M MM                 `7MM  `7M'   `MF'                        #
    #                     MMmmMM      `YMMNq.   M  Mb  M' MM                   MM    VA   ,V                          #
    #                     MM   Y  , .     `MM   M  YM.P'  MM      mmmmm        MM     VA ,V                           #
    #                     MM     ,M Mb     dM   M  `YM'   MM                   MM      VVV                            #
    #                   .JMMmmmmMMM P"Ybmmd"  .JML. `'  .JMML.               .JMML.     W                             #
    ###################################################################################################################
    if model_select == "ESM_1V":
        # model, alphabet = torch.hub.load("facebookresearch/esm", "esm1v_t33_650M_UR90S_1")
        model, alphabet   = esm.pretrained.esm1v_t33_650M_UR90S_1()
        batch_converter   = alphabet.get_batch_converter()

        # build full dataset
        data_set = [(str(rec.id), str(rec.seq)) for rec in SeqIO.parse(input_file, "fasta")]

        chunk_size = 2000                           # split into manageable pieces
        data_set_list = [
            data_set[i : i + chunk_size]            # one sub‑FASTA per chunk
            for i in range(0, len(data_set), chunk_size)
        ]

        for data_set_id, one_data_set in enumerate(data_set_list):
            model.eval()
            model.cuda()

            seq_encodings   = []
            seq_all_hiddens = []
            seq_ids         = []
            sequences       = []

            for i in range(0, len(one_data_set), batch_size):
                print(i, "out of", len(one_data_set), "; ",
                    data_set_id + 1, "out of", len(data_set_list))

                batch = one_data_set[i : i + batch_size]
                batch_labels, batch_strs, batch_tokens = batch_converter(batch)
                batch_tokens = batch_tokens.cuda()

                seq_ids   += batch_labels
                sequences += batch_strs

                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33])
                results = results["representations"][33].cpu().detach()

                sequence_representations = []
                for j, (_, seq) in enumerate(batch):
                    # per‑residue representations
                    seq_all_hiddens.append(results[j, 1 : len(seq) + 1].numpy())
                    # mean‑pool to one vector / seq
                    sequence_representations.append(results[j, 1 : len(seq) + 1].mean(0))

                seq_encodings.append(np.stack(sequence_representations))

            seq_embeddings = np.concatenate(seq_encodings)
            print("seq_embeddings.shape:", seq_embeddings.shape)

            seq_embedding_output = {
                "seq_embeddings"  :   seq_embeddings  , 
                "seq_ids"         :   seq_ids         , 
                "sequences"       :   sequences       , 
                "seq_all_hiddens" :   seq_all_hiddens , 
            }

            out_path = Path(str(output_file).replace(".p", f"_{data_set_id}.p"))
            with open(out_path, "wb") as h:
                pickle.dump(seq_embedding_output, h)

        print("done")

    ###################################################################################################################
    #          `7MM"""YMM   .M"""bgd `7MMM.     ,MMF'                         .6*"    M******                         #
    #            MM    `7  ,MI    "Y   MMMb    dPMM                         ,M'      .M                               #
    #            MM   d    `MMb.       M YM   ,M MM         pd*"*b.        ,Mbmmm.   |bMMAg.   ,pP""Yq.               #
    #            MMmmMM      `YMMNq.   M  Mb  M' MM        (O)   j8        6M'  `Mb.      `Mb 6W'    `Wb              #
    #            MM   Y  , .     `MM   M  YM.P'  MM  mmmmm     ,;j9 mmmmm  MI     M8       jM 8M      M8              #
    #            MM     ,M Mb     dM   M  `YM'   MM         ,-='           WM.   ,M9 (O)  ,M9 YA.    ,A9              #
    #          .JMMmmmmMMM P"Ybmmd"  .JML. `'  .JMML.      Ammmmmmm         WMbmmd9   6mmm9    `Ybmmd9'               #
    ###################################################################################################################
    if model_select == "ESM_2_650":
        # model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        model, alphabet   = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter   = alphabet.get_batch_converter()


        data_set = []
        for seq_record in SeqIO.parse(input_file, "fasta"):
            data_set.append((str(seq_record.id), str(seq_record.seq)))
        chunk_size = 2000
        data_set_list = [data_set[i:i + chunk_size] for i in range(0, len(data_set), chunk_size)]


        for data_set_id, one_data_set in enumerate(data_set_list):
            model.eval()
            model.cuda()
            seq_encodings   = []
            seq_all_hiddens = []
            seq_ids         = []
            sequences       = []  

            for i in range(0, len(one_data_set), batch_size):

                print(i, "out of", len(one_data_set), "; ", data_set_id + 1, "out of", len(data_set_list))
                batch = one_data_set[i:i+batch_size] if i + batch_size <= len(one_data_set) else one_data_set[i:]
                batch_labels, batch_strs, batch_tokens = batch_converter(batch)
                print("batch_tokens.size(): ", batch_tokens.size())
                batch_tokens = batch_tokens.cuda()
                seq_ids     += batch_labels
                sequences   += batch_strs
                
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33])
                results = results["representations"][33].cpu().detach()

                print("results.size(): ", results.size())
                sequence_representations = []
                for i, ( _ , seq ) in enumerate(batch):
                    seq_all_hiddens.append(results[i, 1 : len(seq) + 1].numpy())
                    sequence_representations.append(results[i, 1 : len(seq) + 1].mean(0))
                seq_encodings.append(np.stack(sequence_representations))

            seq_embeddings = np.concatenate(seq_encodings)
            print("seq_embeddings.shape: ", seq_embeddings.shape)
            seq_embedding_output = {"seq_embeddings": seq_embeddings, "seq_ids": seq_ids, "seq_all_hiddens": seq_all_hiddens, "sequences": sequences}
            with open( Path(str(output_file).replace(".p", "_"+ str(data_set_id) +".p")), "wb" ) as h:
                pickle.dump( seq_embedding_output, h )

        print("done")


    ###################################################################################################################
    #              `7MM"""YMM   .M"""bgd `7MMM.     ,MMF'                        pd""b.   `7MM"""Yp,                  #
    #                MM    `7  ,MI    "Y   MMMb    dPMM                         (O)  `8b    MM    Yb                  #
    #                MM   d    `MMb.       M YM   ,M MM         pd*"*b.              ,89    MM    dP                  #
    #                MMmmMM      `YMMNq.   M  Mb  M' MM        (O)   j8            ""Yb.    MM"""bg.                  #
    #                MM   Y  , .     `MM   M  YM.P'  MM  mmmmm     ,;j9 mmmmm         88    MM    `Y                  #
    #                MM     ,M Mb     dM   M  `YM'   MM         ,-='            (O)  .M'    MM    ,9                  #
    #              .JMMmmmmMMM P"Ybmmd"  .JML. `'  .JMML.      Ammmmmmm          bmmmd'   .JMMmmmd9                   #
    ###################################################################################################################
    if model_select == "ESM_2_3B":

        # model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t36_3B_UR50D")
        model, alphabet   = esm.pretrained.esm2_t36_3B_UR50D()
        batch_converter   = alphabet.get_batch_converter()

        data_set = []
        for seq_record in SeqIO.parse(input_file, "fasta"):
            data_set.append((str(seq_record.id), str(seq_record.seq)))
        chunk_size = 1000
        data_set_list = [data_set[i:i + chunk_size] for i in range(0, len(data_set), chunk_size)]

        for data_set_id, one_data_set in enumerate(data_set_list):
            model.eval()
            model.cuda()
            seq_encodings   = []
            seq_all_hiddens = []
            seq_ids         = []
            sequences       = []  

            for i in range(0, len(one_data_set), batch_size):

                print(i, "out of", len(one_data_set), "; ", data_set_id + 1, "out of", len(data_set_list))
                batch = one_data_set[i:i+batch_size] if i + batch_size <= len(one_data_set) else one_data_set[i:]
                batch_labels, batch_strs, batch_tokens = batch_converter(batch)
                print("batch_tokens.size(): ", batch_tokens.size())
                batch_tokens = batch_tokens.cuda()
                seq_ids     += batch_labels
                sequences   += batch_strs

                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[36])
                results = results["representations"][36].cpu().detach()

                print("results.size(): ", results.size())
                sequence_representations = []
                for i, ( _ , seq ) in enumerate(batch):
                    seq_all_hiddens.append(results[i, 1 : len(seq) + 1].numpy())
                    sequence_representations.append(results[i, 1 : len(seq) + 1].mean(0))
                seq_encodings.append(np.stack(sequence_representations))

            seq_embeddings = np.concatenate(seq_encodings)
            print("seq_embeddings.shape: ", seq_embeddings.shape)
            seq_embedding_output = {"seq_embeddings": seq_embeddings, "seq_ids": seq_ids, "seq_all_hiddens": seq_all_hiddens, "sequences": sequences}
            with open( Path(str(output_file).replace(".p", "_"+ str(data_set_id) +".p")), "wb" ) as h:
                pickle.dump( seq_embedding_output, h )

        print("done")



    ###################################################################################################################
    #              `7MM"""YMM   .M"""bgd `7MMM.     ,MMF'                             M******  `7MM"""Yp,             #
    #                MM    `7  ,MI    "Y   MMMb    dPMM                        __,   .M          MM    Yb             #
    #                MM   d    `MMb.       M YM   ,M MM         pd*"*b.       `7MM   |bMMAg.     MM    dP             #
    #                MMmmMM      `YMMNq.   M  Mb  M' MM        (O)   j8         MM        `Mb    MM"""bg.             #
    #                MM   Y  , .     `MM   M  YM.P'  MM  mmmmm     ,;j9 mmmmm   MM         jM    MM    `Y             #
    #                MM     ,M Mb     dM   M  `YM'   MM         ,-='            MM   (O)  ,M9    MM    ,9             #
    #              .JMMmmmmMMM P"Ybmmd"  .JML. `'  .JMML.      Ammmmmmm       .JMML.  6mmm9    .JMMmmmd9              #
    ###################################################################################################################
    if model_select == "ESM_2_15B":

        # model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t48_15B_UR50D")
        model, alphabet   = esm.pretrained.esm2_t48_15B_UR50D()
        batch_converter   = alphabet.get_batch_converter()

        data_set = []
        for seq_record in SeqIO.parse(input_file, "fasta"):
            data_set.append((str(seq_record.id), str(seq_record.seq)))
        chunk_size = 500
        data_set_list = [data_set[i:i + chunk_size] for i in range(0, len(data_set), chunk_size)]

        for data_set_id, one_data_set in enumerate(data_set_list):
            model.eval()
            model.cuda()
            seq_encodings   = []
            seq_all_hiddens = []
            seq_ids         = []
            sequences       = []  

            for i in range(0, len(one_data_set), batch_size):

                print(i, "out of", len(one_data_set), "; ", data_set_id + 1, "out of", len(data_set_list))
                batch = one_data_set[i:i+batch_size] if i + batch_size <= len(one_data_set) else one_data_set[i:]
                batch_labels, batch_strs, batch_tokens = batch_converter(batch)
                print("batch_tokens.size(): ", batch_tokens.size())
                batch_tokens = batch_tokens.cuda()
                seq_ids     += batch_labels
                sequences   += batch_strs

                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[48])
                results = results["representations"][48].cpu().detach()

                print("results.size(): ", results.size())
                sequence_representations = []
                for i, ( _ , seq ) in enumerate(batch):
                    seq_all_hiddens.append(results[i, 1 : len(seq) + 1].numpy())
                    sequence_representations.append(results[i, 1 : len(seq) + 1].mean(0))
                seq_encodings.append(np.stack(sequence_representations))

            seq_embeddings = np.concatenate(seq_encodings)
            print("seq_embeddings.shape: ", seq_embeddings.shape)
            seq_embedding_output = {"seq_embeddings": seq_embeddings, "seq_ids": seq_ids, "seq_all_hiddens": seq_all_hiddens, "sequences": sequences}
            with open( Path(str(output_file).replace(".p", "_"+ str(data_set_id) +".p")), "wb" ) as h:
                pickle.dump( seq_embedding_output, h )

        print("done")

    ##########################################################################################################                                               
    #                                     ,,                                                                 #
    #        db                `7MM     `7MM            `7MMF'                                               #
    #       ;MM:                 MM       MM              MM                                                 #
    #      ,V^MM.   `7MMpMMMb.   MM  ,MP' MMpMMMb.        MM         ,6"Yb. `7Mb,od8 .P"Ybmmm .gP"Ya         #
    #     ,M  `MM     MM    MM   MM ;Y    MM    MM        MM        8)   MM   MM' "':MI  I8  ,M'   Yb        #
    #     AbmmmqMA    MM    MM   MM;Mm    MM    MM        MM      ,  ,pm9MM   MM     WmmmP"  8M""""""        #
    #    A'     VML   MM    MM   MM `Mb.  MM    MM        MM     ,M 8M   MM   MM    8M       YM.    ,        #  
    #  .AMA.   .AMMA.JMML  JMML.JMML. YA.JMML  JMML.    .JMMmmmmMMM `Moo9^Yo.JMML.   YMMMMMb  `Mbmmd'        #
    #                                                                               6'     dP                #
    #                                                                               Ybmmmd'                  #
    ##########################################################################################################                                           
    if model_select == "Ankh_Large":
        
        model, tokenizer = ankh.load_large_model()
        
        data_set = []
        for seq_record in SeqIO.parse(input_file, "fasta"):
            data_set.append((str(seq_record.id), str(seq_record.seq)))
        chunk_size = 2000
        data_set_list = [data_set[i:i + chunk_size] for i in range(0, len(data_set), chunk_size)]


        for data_set_id, one_data_set in enumerate(data_set_list):
            model.eval()
            model.cuda()
            seq_encodings   = [] 
            seq_all_hiddens = [] 
            seq_ids         = [] 
            sequences       = [] 

            for i in range(0, len(one_data_set), batch_size):
                print(i, "out of", len(one_data_set), "; ", data_set_id, "out of", len(data_set_list))

                batch = one_data_set[i:i + batch_size] if i + batch_size <= len(one_data_set) else one_data_set[i:]
                ids_batch, seqs_batch = zip(*batch)                 # split tuples
                seq_ids              += ids_batch
                sequences            += seqs_batch

                outputs = tokenizer.batch_encode_plus(seqs_batch, add_special_tokens=True, padding=True, return_tensors="pt")

                with torch.no_grad():
                    embeddings = model(outputs["input_ids"].cuda(), attention_mask=outputs["attention_mask"].cuda())
                results = embeddings[0].cpu().detach()              # (B, L, D)

                print("results.size(): ", results.size())
                sequence_representations = []
                for j, seq in enumerate(seqs_batch):
                    # drop <s> and </s>, keep per‑residue hidden states
                    seq_all_hiddens.append(results[j, 1: len(seq) + 1].numpy())
                    sequence_representations.append(results[j, 1: len(seq) + 1].mean(0))
                seq_encodings.append(np.stack(sequence_representations))

            seq_embeddings = np.concatenate(seq_encodings)
            print("seq_embeddings.shape:", seq_embeddings.shape)
            seq_embedding_output = {
                "seq_embeddings"  :   seq_embeddings  , 
                "seq_ids"         :   seq_ids         , 
                "sequences"       :   sequences       , 
                "seq_all_hiddens" :   seq_all_hiddens , 
            }

            out_path = Path(str(output_file).replace(".p", f"_{data_set_id}.p"))
            with open(out_path, "wb") as h:
                pickle.dump(seq_embedding_output, h)

        print("done")
        
     ######################################################################################################                                                                                        
     #                                     ,,                                                             #
     #        db                `7MM     `7MM            `7MM"""Yp,                                       #
     #       ;MM:                 MM       MM              MM    Yb                                       #
     #      ,V^MM.   `7MMpMMMb.   MM  ,MP' MMpMMMb.        MM    dP  ,6"Yb.  ,pP"Ybd  .gP"Ya              #
     #     ,M  `MM     MM    MM   MM ;Y    MM    MM        MM"""bg. 8)   MM  8I   `" ,M'   Yb             #
     #     AbmmmqMA    MM    MM   MM;Mm    MM    MM        MM    `Y  ,pm9MM  `YMMMa. 8M""""""             #
     #    A'     VML   MM    MM   MM `Mb.  MM    MM        MM    ,9 8M   MM  L.   I8 YM.    ,             #
     #  .AMA.   .AMMA.JMML  JMML.JMML. YA.JMML  JMML.    .JMMmmmd9  `Moo9^Yo.M9mmmP'  `Mbmmd'             #
     #                                                                                                    #
     ######################################################################################################  
     
    if model_select == "Ankh_Base":
         
        model, tokenizer = ankh.load_base_model()
         
        data_set = []
        for seq_record in SeqIO.parse(input_file, "fasta"):
            data_set.append((str(seq_record.id), str(seq_record.seq)))
        chunk_size = 2000
        data_set_list = [data_set[i:i + chunk_size] for i in range(0, len(data_set), chunk_size)]


        for data_set_id, one_data_set in enumerate(data_set_list):
            model.eval()
            model.cuda()
            seq_encodings   = [] 
            seq_all_hiddens = [] 
            seq_ids         = [] 
            sequences       = [] 

            for i in range(0, len(one_data_set), batch_size):
                print(i, "out of", len(one_data_set), "; ", data_set_id, "out of", len(data_set_list))

                batch = one_data_set[i:i + batch_size] if i + batch_size <= len(one_data_set) else one_data_set[i:]
                ids_batch, seqs_batch = zip(*batch)                 # split tuples
                seq_ids              += ids_batch
                sequences            += seqs_batch

                outputs = tokenizer.batch_encode_plus(seqs_batch, add_special_tokens=True, padding=True, return_tensors="pt")

                with torch.no_grad():
                    embeddings = model(outputs["input_ids"].cuda(), attention_mask=outputs["attention_mask"].cuda())
                results = embeddings[0].cpu().detach()              # (B, L, D)

                print("results.size(): ", results.size())
                sequence_representations = []
                for j, seq in enumerate(seqs_batch):
                    # drop <s> and </s>, keep per‑residue hidden states
                    seq_all_hiddens.append(results[j, 1: len(seq) + 1].numpy())
                    sequence_representations.append(results[j, 1: len(seq) + 1].mean(0))
                seq_encodings.append(np.stack(sequence_representations))

            seq_embeddings = np.concatenate(seq_encodings)
            print("seq_embeddings.shape:", seq_embeddings.shape)
            seq_embedding_output = {
                "seq_embeddings"  :   seq_embeddings  , 
                "seq_ids"         :   seq_ids         , 
                "sequences"       :   sequences       , 
                "seq_all_hiddens" :   seq_all_hiddens , 
            }

            out_path = Path(str(output_file).replace(".p", f"_{data_set_id}.p"))
            with open(out_path, "wb") as h:
                pickle.dump(seq_embedding_output, h)

        print("done")
         
    ########################################################################################################                        
    #                                                                                                      #
    #       .g8"""bgd     db     `7MM"""Mq. `7MM"""Mq.    .6*"                      `7MMM.     ,MMF'       #
    #     .dP'     `M    ;MM:      MM   `MM.  MM   `MM. ,M'                           MMMb    dPMM         #
    #     dM'       `   ,V^MM.     MM   ,M9   MM   ,M9 ,Mbmmm.        ,AM   ,pP""Yq.  M YM   ,M MM         #
    #     MM           ,M  `MM     MMmmdM9    MMmmdM9  6M'  `Mb.     AVMM  6W'    `Wb M  Mb  M' MM         #
    #     MM.          AbmmmqMA    MM  YM.    MM mmmmm MI     M8   ,W' MM  8M      M8 M  YM.P'  MM         #
    #     `Mb.     ,' A'     VML   MM   `Mb.  MM       WM.   ,M9 ,W'   MM  YA.    ,A9 M  `YM'   MM         #
    #       `"bmmmd'.AMA.   .AMMA.JMML. .JMM.JMML.      WMbmmd9  AmmmmmMMmm `Ybmmd9'.JML. `'  .JMML.       #
    #                                                                  MM#                                 #
    #                                                                  MM                                  #
    ########################################################################################################
    
    if model_select == "CARP_640M":

        model, collater = load_model_and_alphabet("carp_640M")

        data_set = []
        for seq_record in SeqIO.parse(input_file, "fasta"):
            data_set.append((str(seq_record.id), str(seq_record.seq)))
        chunk_size = 1000
        data_set_list = [data_set[i:i + chunk_size] for i in range(0, len(data_set), chunk_size)]

        for data_set_id, one_data_set in enumerate(data_set_list):
            model.eval()
            model.cuda()
            seq_encodings   = []
            seq_all_hiddens = []
            seq_ids         = []
            sequences       = []  

            for i in range(0, len(one_data_set), batch_size):

                print(i, "out of", len(one_data_set), "; ", data_set_id + 1, "out of", len(data_set_list))
                batch = one_data_set[i : i + batch_size]
                batch_ids  = [p[0] for p in batch]
                batch_seqs = [p[1] for p in batch]
                seq_ids   += batch_ids
                sequences += batch_seqs

                # collater returns (tokens, lengths); we want only tokens
                batch_tokens = collater(batch_seqs)[0].cuda()

                with torch.no_grad():
                    out = model(batch_tokens)
                hidden = out["representations"][56].cpu().detach()  # layer 56

                sequence_representations = []
                for j, seq in enumerate(batch_seqs):
                    seq_all_hiddens.append(hidden[j, 1 : len(seq) + 1].numpy())
                    sequence_representations.append(hidden[j, 1 : len(seq) + 1].mean(0))
                seq_encodings.append(np.stack(sequence_representations))

            seq_embeddings = np.concatenate(seq_encodings)
            print("seq_embeddings.shape:", seq_embeddings.shape)

            seq_embedding_output = {
                "seq_embeddings":  seq_embeddings,
                "seq_ids":         seq_ids,
                "seq_all_hiddens": seq_all_hiddens,
                "sequences":       sequences,        # ← new
            }

            out_path = Path(str(output_file).replace(".p", f"_{data_set_id}.p"))
            with open(out_path, "wb") as h:
                pickle.dump(seq_embedding_output, h)

        print("done")
    ###################################################################################################################
    ###################################################################################################################
    return seq_embedding_output



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#   `7MMM.     ,MMF'      db      `7MMF'`7MN.   `7MF'                 M             M             M                                                    #
#     MMMb    dPMM       ;MM:       MM    MMN.    M                   M             M             M                                                    #
#     M YM   ,M MM      ,V^MM.      MM    M YMb   M                   M             M             M                                                    #
#     M  Mb  M' MM     ,M  `MM      MM    M  `MN. M               `7M'M`MF'     `7M'M`MF'     `7M'M`MF'                                                #
#     M  YM.P'  MM     AbmmmqMA     MM    M   `MM.M                 VAMAV         VAMAV         VAMAV                                                  #
#     M  `YM'   MM    A'     VML    MM    M     YMM                  VVV           VVV           VVV                                                   #
#   .JML. `'  .JMML..AMA.   .AMMA..JMML..JML.    YM                   V             V             V                                                    #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
if __name__ == "__main__":
    #====================================================================================================#
    # Args
    Step_code = "N03_"
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

    
    dataset_nme          = dataset_nme_list[17]
    data_folder          = Path("N_DataProcessing/")
    input_seqs_fasta_file = "N00_" + dataset_nme + ".fasta"
    #====================================================================================================#
    #====================================================================================================#
    # List Index:          [0]     [1]      [2]       [3]       [4]     [5]     [6]       [7]      [8]
    models_list      = ["TAPE", "TAPE_FT", "BERT", "ALBERT", "Electra", "T5", "Xlnet", "ESM_1B", "ESM_1V", 
    #                        [9]         [10]        [11]        [12]    [13]       [14]         [15]         [16]
                        "ESM_2_650", "ESM_2_3B", "ESM_2_15B", "Unirep", "T5XL", "Ankh_Large", "Ankh_Base", "CARP_640M"]
    # Select model using index. ( ##### !!!!! models_list[3] Electra deprecated ! )
    model_select     = models_list[7] 
    pretraining_name = "X01_" + dataset_nme + "_FT_inter_epoch5_trial_training.pt"
    #====================================================================================================#
    output_file_name_header = Step_code + dataset_nme + "_embedding_"
    #====================================================================================================#
    batch_size    = 20
    xlnet_mem_len = 512
    
    #====================================================================================================#
    # ArgParser
    parser = argparse.ArgumentParser(
        description="Preprocesses the sequence datafile.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dataset_nme"              , type = str  , default = dataset_nme             , help = "dataset_nme."                            )
    parser.add_argument("--model_select"             , type = str  , default = model_select            , help = "model_select."                           )
    parser.add_argument("--data_folder"              , type = Path , default = data_folder             , help = "Path to the dir contains the datasets."  )
    parser.add_argument("--input_seqs_fasta_file"    , type = str  , default = input_seqs_fasta_file   , help = "input_seqs_fasta_file."                  )
    parser.add_argument("--output_file_name_header"  , type = str  , default = output_file_name_header , help = "output_file_name_header."                )
    parser.add_argument("--pretraining_name"         , type = str  , default = pretraining_name        , help = "pretraining_name."                       )
    parser.add_argument("--batch_size"               , type = int  , default = batch_size              , help = "Batch size."                             )
    parser.add_argument("--xlnet_mem_len"            , type = int  , default = 512                     , help = "xlnet_mem_len=512."                      )

    args = parser.parse_args()

    #====================================================================================================#
    # If dataset_nme is specified, use default pretrained model name and output name.
    vars_dict = vars(args)
    vars_dict["input_seqs_fasta_file"]    = "N00_"    + vars_dict["dataset_nme"] + ".fasta"
    vars_dict["pretraining_name"]         = "N01_"    + vars_dict["dataset_nme"] + "_FT_inter_epoch5_trial_training.pt"
    vars_dict["output_file_name_header"]  = Step_code + vars_dict["dataset_nme"] + "_embedding_"

    print(vars_dict)

    #====================================================================================================#
    # Main
    N03_embedding_LM(**vars_dict)

    print("*"*50)
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






