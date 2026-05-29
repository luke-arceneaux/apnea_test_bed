''' Defines Pytorch dataset + dataloader for Apnea detection'''
import numpy as np
import pandas as pd
import os
import sys
import random
import torch 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import wandb
import utils
import traceback


'''
Pytorch dataset for Apnea detection
'''
class ApneaDataset(Dataset):
    def __init__(self, cfg, data):
        self.cfg = cfg
        self.data = data 

    ''' return size of dataset '''
    def __len__(self):
        return len(self.data)

    ''' retrieve one sample '''
    def __getitem__(self, idx):
        label = self.data[idx][0]
        seq = self.data[idx][2]
        return seq, label

    ''' split data into train, test'''
    def get_splits(self, test_frac):
        # determine sizes
        test_size = round(test_frac * len(self.data))
        train_size = len(self.data) - test_size
        # calculate the split
        print(f"Train size: {train_size}, Test size: {test_size}")
        return random_split(self, [train_size, test_size])



''' Dataloader '''
class ApneaDataloader(DataLoader):
    ''' defines train/val datasets, batch size'''
    def __init__(self, cfg, data):

        # dataset 
        self.dataset = ApneaDataset(cfg, data)

        # batch size to use when loading samples
        self.batch_size = int(cfg.batch_size)
        print(self.batch_size)

        # splits dataset into train/validation datasets based on test_frac
        self.train_data, self.val_data = self.dataset.get_splits(float(cfg.test_frac))


    # Gets train, validation dataloader
    def get_data(self):
        # dataloader for training dataset 
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       drop_last=True)
        
        # dataloader for validation dataset 
        self.val_loader = DataLoader(self.val_data,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     drop_last=True)

        return self.train_loader, self.val_loader

