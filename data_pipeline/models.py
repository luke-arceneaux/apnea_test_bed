'''
This script defines various model architectures for apnea detection
'''
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm1d
import numpy as np


''' 
Inception-Net based 1D CNN 

@Input: sequence (batch_size, 1, 120)
@Output: binary logits (0 for no apnea, 1 for apnea)
'''
class CNN(nn.Module):

    ''' Instantiate layers '''
    def __init__(self, input_size=None, 
                       output_size=None):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.block1 = nn.Sequential(

                nn.Conv1d(in_channels=input_size,
                out_channels=128,
                kernel_size=64, # num steps to slide across at each pass
                stride=32,
                bias=True),
             
                nn.BatchNorm1d(128),
                # nn.MaxPool1d(kernel_size=4, stride=2),
                nn.Flatten()
        )


        self.block2 = nn.Sequential(

                nn.Conv1d(in_channels=input_size,
                out_channels=64,
                kernel_size=32,
                stride=2,
                bias=True),
             
                nn.BatchNorm1d(64),

                nn.MaxPool1d(kernel_size=4, stride=2),

                nn.Flatten()
        )

        self.block3 = nn.Sequential(

                nn.Conv1d(in_channels=input_size,
                out_channels=8,
                kernel_size=4,
                stride=2,
                bias=True),
             
                nn.BatchNorm1d(8),

                nn.MaxPool1d(kernel_size=4, stride=2),

                nn.Flatten()
        )
        
        self.block4 = nn.Sequential(

                nn.Conv1d(in_channels=input_size,
                out_channels=4,
                kernel_size=2,
                stride=1,
                bias=True),
             
                nn.BatchNorm1d(4),

                nn.MaxPool1d(kernel_size=4, stride=2),

                nn.Flatten()
        )
        
        self.flatten = nn.Flatten()

        fc1_input_size = self.calculate_fc1_input_size(input_size)
        self.fc1 = nn.Linear(fc1_input_size, 512)

        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()

        # self.dropout = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(128, self.output_size) 
        self.softmax = nn.Softmax(-1) # softmax to get logits

    def calculate_fc1_input_size(self, input_size):
        # Dynamically calculate the input size for fc1 based on the network architecture
        with torch.no_grad():
            dummy_input = torch.randn(16, input_size, 120)
            x = self.block1(dummy_input)
            xx = self.block2(dummy_input)
            xxx = self.block3(dummy_input)
            xxxx = self.block4(dummy_input)
            flat_inp = self.flatten(dummy_input)
            concatenated_size = x.size(1) + xx.size(1) + xxx.size(1) + xxxx.size(1) + flat_inp.size(1)
        return concatenated_size

    ''' forward call to model '''
    def forward(self, inp):
        inp = inp.view(-1, self.input_size, 120)  # B, C, 120

        # propagate inputs through blocks for feature learning
        x = self.block1(inp)  
        xx = self.block2(inp)
        xxx = self.block3(inp)  
        xxxx = self.block4(inp)

        flat_inp = self.flatten(inp)

        # concatenate flattened outputs of each block with flattened original input
        x = torch.cat([x, xx, xxx, xxxx, flat_inp], -1)
        # fully connected layers
        x = self.fc1(x)
#         except:
#             x = self.fc1_1(x)

        x = self.bn1(x)
        x = self.relu1(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # x = self.dropout(x)
        out = self.fc3(x)
        out = self.softmax(out)
        return out


'''LSTM architecture''' 
class LSTM(nn.Module):
    
    # Define model architecture
    def __init__(self,input_size=1, 
                    hidden_size=64,
                    num_layers=4,
                    output_size=2,
                    dropout=0.1):
        super(LSTM,self).__init__()

        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,dropout=dropout,batch_first=True)
        self.relu = nn.ReLU(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size,)
        self.softmax = nn.Softmax(-1)
        
    # Propagate input through model
    def forward(self,inp):
        x, _ = self.lstm(inp)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = x[:, -1, :].squeeze()
        x = self.softmax(x)
        return x