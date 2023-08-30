#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 15:21:15 2022

@author: user
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from efficientnet_pytorch import EfficientNet

class EffNet(nn.Module):
    def __init__(self, n_labels, dtype=torch.FloatTensor, device=torch.device("cpu")):
        """
        Initializes the EffNet nn model class. This is the class used for the transcoders effnet_b0 and effnet_b7 in exp_train_model/main_doce_training.py. 

        Args:
        - mels_tr: The Mel transform used for converting audio into Mel spectrograms. Here it just serves to retrieve the number of labels
                    that corresponds to the classifier outputs (527 for PANN, 521 for YamNet)
        - effnet_type: effnet_b0 or effnet_b7
        - dtype: The data type for the model (default: torch.FloatTensor).
        - device: The device to run the model on (default: torch.device("cpu")).
        """
        super().__init__()
        
        ###############
        #models loading
        self.model = EfficientNet.from_name('efficientnet-b0', num_classes=n_labels)
        # state_dict = torch.load("./efficient_net/efficientnet-b0-355c32eb.pth")
        # state_dict.pop('_fc.weight')
        # state_dict.pop('_fc.bias')
        # self.model.load_state_dict(state_dict, strict=False)

        # modify input conv layer to accept 1x101x64 input
        self.model._conv_stem = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)

        self.model.to(device)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)

        #x = F.interpolate(x, size=(64, 33), mode='nearest')

        y_pred = self.model(x)
        #clamp gave better results than sigmoid function
        y_pred = torch.sigmoid(y_pred)
        #y_pred = torch.clamp(y_pred, min=0, max=1)
        return y_pred

class MLP(nn.Module):
    def __init__(self, input_shape, output_shape, dtype=torch.FloatTensor, 
                 hl_1=100, hl_2=50):
        super().__init__()
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hl_1 = hl_1
        self.hl_2 = hl_2
        self.input_fc = nn.Linear(input_shape, hl_1)
        self.hidden_fc = nn.Linear(hl_1, hl_2)
        self.output_fc = nn.Linear(hl_2, output_shape)
        self.dtype = dtype

    def forward(self, x):

        # x = [batch size, height, width]

        # MT: useless lines (maybe when 2d spectrogramms given ?)
        #batch_size = x.shape[0]
        #x = x.view(batch_size, -1)

        # x = [batch size, height * width]
        
        x = torch.squeeze(x, dim=-1)

        h_1 = F.relu(self.input_fc(x))

        # h_1 = [batch size, 250]

        h_2 = F.relu(self.hidden_fc(h_1))

        # h_2 = [batch size, 100]

        y_pred = self.output_fc(h_2)
        
        y_pred = torch.sigmoid(y_pred)

        # y_pred = torch.reshape(y_pred, (y_pred.shape[0], self.output_shape[0], self.output_shape[1]))

        # y_pred = [batch size, output dim]

        return y_pred