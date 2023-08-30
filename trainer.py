import torch
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
import math
import csv
import re
import numpy as np
import os
import shutil
import time
from torch.autograd import Variable
import torch.nn as nn
import librosa as lr
from pathlib import Path
import sys
import copy
from sklearn.metrics import log_loss
import yaml
import pickle
from sklearn.model_selection import KFold
import models as md
import h5py
from torch.utils.data import WeightedRandomSampler

#for CNN + PINV
class MelTrainer:
    def __init__(self, model, models_path, model_name, split_id, train_dataset=None, 
                 eval_dataset=None, out_path='./outputs/', out_name='dataset_inference', learning_rate=1e-3, dtype=torch.FloatTensor, classifier='PANN'):
        """
        Initializes the MelTrainer class. This class trains a model only on Mel spectrogram values.

        Args:
        - setting_data: The setting data for the model.
        - model: The model architecture.
        - models_path: The path to save the trained model.
        - transcoder: The transcoder type (cnn_pinv, mlp, mlp_pinv)
        - model_name: The name of the model.
        - train_dataset: The training dataset.
        - valid_dataset: The validation dataset.
        - eval_dataset: The evaluation dataset.
        - learning_rate: The learning rate for optimization.
        - dtype: The data type for the model (default: torch.FloatTensor).
        - classifier: The type of classifier (default: 'PANN').
        """
        self.dtype = dtype
        self.classifier = classifier

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        self.train_duration = 0

        # for dataset in [train_dataset, valid_dataset, eval_dataset]:
        #     if dataset != None:
        #         self.flen = dataset.flen
        #         self.hlen = dataset.hlen
        #         self.sr = dataset.sr
        #         break
        
        self.model = model
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)
        
        self.models_path = models_path
        self.model_name = model_name

        self.split_id = split_id
        self.out_path = out_path
        self.out_name = out_name
        print('TRAINED MODEL')
        # ut.count_parameters(self.model)
        
    def train(self, batch_size=64, epochs=10, device=torch.device("cpu")):
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        losses_train = []
        losses_valid = []
        losses_eval = []

        self.model.train()
        #self.loss_function = nn.MSELoss()
        self.loss_function = nn.CrossEntropyLoss()
        self.model = self.model.to(self.device)

        # #fist validation
        # loss_valid = self.validate(self.valid_dataset, 0, batch_size=batch_size, device=self.device, forced=True)
        # losses_valid.append(loss_valid)
        
        # loss_eval = self.validate(self.eval_dataset, 0, batch_size=batch_size, device=self.device, label='EVALUATION')
        # losses_eval.append(loss_eval)
        
        #validation on evaluation
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False, drop_last=False)
        cur_loss = 0
        for cur_epoch in range(self.epochs):
            tqdm_it=tqdm(self.train_dataloader, 
                         desc='TRAINING: Epoch {}, Chunk {}/{}, loss: {:.4f}'
                         .format(cur_epoch+1, 0, 0, cur_loss))
            for (idx,_,x,y) in tqdm_it:

                start_time = time.time()

                x = x.type(self.dtype)
                y = y.type(self.dtype)
                
                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                y_pred = self.model(x)

                cur_loss = self.loss_function(y_pred,y)
                
                cur_loss.backward()

                batch_duration = time.time() - start_time
                self.train_duration += batch_duration
                
                self.optimizer.step()
                
                cur_loss = float(cur_loss.data)

                losses_train.append(cur_loss)
                
                tqdm_it.set_description('TRAINING: Epoch {}, Chunk {}/{}, loss: {:.4f}'
                                        .format(cur_epoch+1,0,0,cur_loss))                    
                
            #Validation
            # loss_valid = self.validate(self.valid_dataset, cur_epoch, batch_size=batch_size, device=self.device)
            # losses_valid.append(loss_valid)
            
            # loss_eval = self.validate(self.eval_dataset, cur_epoch, batch_size=batch_size, device=self.device, label='EVALUATION')
            # losses_eval.append(loss_eval)
        
        # losses = {
        #         'losses_train': np.array(losses_train),
        #         'losses_valid': np.array(losses_valid),
        #         'losses_eval': np.array(losses_eval)
        #     }
        self.best_state_dict = copy.deepcopy(self.model.state_dict())
        losses_train = np.array(losses_train)
        return(losses_train)
    
    def validate(self, dataset, cur_epoch, batch_size=64, device=torch.device("cpu"), label='VALIDATION', forced=False):
        self.model.eval
        losses_valid = []
        loss_function = nn.MSELoss()
        
        valid_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
        tqdm_it=tqdm(valid_dataloader, desc=label+': Chunk {}/{}'.format(0,0))

        for (idx,x,y,_) in tqdm_it:
            x = x.type(self.dtype)
            y = y.type(self.dtype)

            x = x.to(device)
            y = y.to(device)

            y_pred = self.model(x)

            cur_loss = loss_function(y_pred,y)

            losses_valid.append(cur_loss.detach())
                
        losses_valid = torch.Tensor(losses_valid)
        loss_valid = torch.mean(losses_valid)
        print(" => Validation loss at epoch {} is {:.4f}".format(cur_epoch+1, loss_valid))
        
        if forced == True:
            self.best_loss = loss_valid
            self.best_state_dict = copy.deepcopy(self.model.state_dict())
            self.best_epoch = cur_epoch
        else:
            if label == 'VALIDATION':
                if loss_valid <= self.best_loss:
                    self.best_loss = loss_valid
                    self.best_state_dict = copy.deepcopy(self.model.state_dict())
                    self.best_epoch = cur_epoch
        
        return loss_valid.detach().cpu().numpy()
    
    def load_model(self, device):
        self.model = self.model.to(device)
        state_dict = torch.load(self.models_path + self.model_name, map_location=device)
        self.model.load_state_dict(state_dict)
        
    def save_model(self):
    
        """
        SAVE MODEL
        """
    
        torch.save(self.best_state_dict, self.models_path + self.model_name)
    
        # """
        # "SAVE MODEL SETTINGS"
        # """
        
        # transcoder = self.transcoder
        # input_shape = self.model.input_shape
        # output_shape = self.model.output_shape
        
        # cnn_kernel_size = None
        # cnn_dilation = None
        # cnn_nb_layers = None
        # cnn_nb_channels = None
        
        # mlp_hl_1 = None
        # mlp_hl_2 = None
        
        # if 'cnn' in transcoder:
        #     cnn_kernel_size = self.model.kernel_size
        #     cnn_dilation = self.model.dilation
        #     cnn_nb_layers = self.model.nb_layers
        #     cnn_nb_channels = self.model.nb_channels
            
        # if 'mlp' in transcoder:
        #     mlp_hl_1 = self.model.hl_1
        #     mlp_hl_2 = self.model.hl_2

        # model_settings = {
        #     "model_type": transcoder,
        #   "input_shape": input_shape,
        #   "output_shape": output_shape,
        #   "cnn_kernel_size": cnn_kernel_size,
        #   "cnn_dilation": cnn_dilation,
        #   "cnn_nb_layers": cnn_nb_layers,
        #   "cnn_nb_channels": cnn_nb_channels,
        #   "mlp_hl_1": mlp_hl_1,
        #   "mlp_hl_2": mlp_hl_2,
        #   "mels_type": self.classifier,
        #   "batch_size": self.batch_size,
        #   "epochs": self.epochs,
        #   "settings": self.setting_data
        # }

        # with open(self.models_path + self.model_name + '_settings.yaml', 'w') as file:
        #     yaml.dump(model_settings, file)
    def evaluate(self, batch_size=64, device=torch.device("cpu")):
        self.model.eval
                
        self.eval_dataloader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
        tqdm_it=tqdm(self.eval_dataloader, desc='EVALUATION: Chunk {}/{}'.format(0,0))
        
        #save the output of the model in a .dat file. This avoids havind memory issues
        # output_logits = np.memmap(self.outputs_path['logits']+'.dat', dtype=np.float64,
        #               mode='w+', shape=(self.eval_dataset.n_tho_frames, self.classif_inference.n_labels))

        with h5py.File(self.out_path + self.out_name + '.h5', 'a') as hf:
            group = f'Singer{self.split_id+1}'
            if group in hf:
                del hf[group]
                print(f"Group {group} deleted.")
            split_group = hf.create_group(group)
            eval_group = split_group.create_group('inference')
            eval_group.create_dataset('audio_name', shape=((self.eval_dataset.len_dataset,)), dtype='S200')
            eval_group.create_dataset('groundtruth', shape=((self.eval_dataset.len_dataset,)), dtype='S200')
            eval_group.create_dataset('inference', shape=((self.eval_dataset.len_dataset,)), dtype='S200')

            for (idx,name,x,y) in tqdm_it:
                x = x.type(self.dtype)
                y = y.type(self.dtype)
                
                x = x.to(device)
                y = y.to(device)
                
                y_pred = self.model(x)
                
                name = list(name)

                argm_pred = torch.argmax(y_pred, dim=1)
                argm = torch.argmax(y, dim=1)

                eval_group['audio_name'][idx] = name
                eval_group['inference'][idx] = [self.eval_dataset.gt_list_to_str[key] for key in argm_pred.detach().cpu().tolist()]
                eval_group['groundtruth'][idx] = [self.eval_dataset.gt_list_to_str[key] for key in argm.detach().cpu().tolist()]
        
        return()

#for CNN + PINV
class MelRegularTrainer:
    def __init__(self, model, models_path, model_name, split_id, train_dataset=None, valid_dataset=None,
                 eval_dataset=None, out_path='./outputs/', out_name='dataset_inference', learning_rate=1e-3, dtype=torch.FloatTensor, classifier='PANN'):
        """
        Initializes the MelTrainer class. This class trains a model only on Mel spectrogram values.

        Args:
        - setting_data: The setting data for the model.
        - model: The model architecture.
        - models_path: The path to save the trained model.
        - transcoder: The transcoder type (cnn_pinv, mlp, mlp_pinv)
        - model_name: The name of the model.
        - train_dataset: The training dataset.
        - valid_dataset: The validation dataset.
        - eval_dataset: The evaluation dataset.
        - learning_rate: The learning rate for optimization.
        - dtype: The data type for the model (default: torch.FloatTensor).
        - classifier: The type of classifier (default: 'PANN').
        """
        self.dtype = dtype
        self.classifier = classifier

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.eval_dataset = eval_dataset
        
        self.train_duration = 0
        self.split_id = split_id

        # for dataset in [train_dataset, valid_dataset, eval_dataset]:
        #     if dataset != None:
        #         self.flen = dataset.flen
        #         self.hlen = dataset.hlen
        #         self.sr = dataset.sr
        #         break
        
        self.model = model
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)
        
        self.models_path = models_path
        self.model_name = model_name

        self.out_path = out_path
        self.out_name = out_name
        if self.train_dataset:
            self.train_sampler = WeightedRandomSampler(self.train_dataset.samples_weight, len(self.train_dataset.samples_weight))
        if self.valid_dataset:
            self.valid_sampler = WeightedRandomSampler(self.valid_dataset.samples_weight, len(self.valid_dataset.samples_weight))
        if self.eval_dataset:
            self.eval_sampler = WeightedRandomSampler(self.eval_dataset.samples_weight, len(self.eval_dataset.samples_weight))

        print('TRAINED MODEL')
        # ut.count_parameters(self.model)
        
    def train(self, batch_size=64, epochs=10, device=torch.device("cpu")):
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        losses_train = []
        losses_valid = []
        losses_eval = []

        self.model.train()
        #self.loss_function = nn.MSELoss()
        self.loss_function = nn.CrossEntropyLoss()
        self.model = self.model.to(self.device)

        #fist validation
        loss_valid = self.validate(self.valid_dataset, 0, batch_size=batch_size, device=self.device, forced=True)
        losses_valid.append(loss_valid)
        
        loss_eval = self.validate(self.eval_dataset, 0, batch_size=batch_size, device=self.device, label='EVALUATION')
        losses_eval.append(loss_eval)
        
        
        #validation on evaluation
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, num_workers=8, pin_memory=False, drop_last=False, sampler=self.train_sampler)
        cur_loss = 0
        for cur_epoch in range(self.epochs):
            tqdm_it=tqdm(self.train_dataloader, 
                         desc='TRAINING: Epoch {}, Chunk {}/{}, loss: {:.4f}'
                         .format(cur_epoch+1, 0, 0, cur_loss))
            for (idx,_,x,y) in tqdm_it:

                start_time = time.time()

                x = x.type(self.dtype)
                y = y.type(self.dtype)
                
                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                y_pred = self.model(x)

                cur_loss = self.loss_function(y_pred,y)
                
                cur_loss.backward()

                batch_duration = time.time() - start_time
                self.train_duration += batch_duration
                
                self.optimizer.step()
                
                cur_loss = float(cur_loss.data)

                losses_train.append(cur_loss)
                
                tqdm_it.set_description('TRAINING: Epoch {}, Chunk {}/{}, loss: {:.4f}'
                                        .format(cur_epoch+1,0,0,cur_loss))                    
                
            #Validation
            loss_valid = self.validate(self.valid_dataset, cur_epoch, batch_size=batch_size, device=self.device)
            losses_valid.append(loss_valid)
            
            loss_eval = self.validate(self.eval_dataset, cur_epoch, batch_size=batch_size, device=self.device, label='EVALUATION')
            losses_eval.append(loss_eval)
        
        losses = {
                'losses_train': np.array(losses_train),
                'losses_valid': np.array(losses_valid),
                'losses_eval': np.array(losses_eval)
            }
        
        # self.best_state_dict = copy.deepcopy(self.model.state_dict())
        # losses_train = np.array(losses_train)
        return(losses)
    
    def validate(self, dataset, cur_epoch, batch_size=64, device=torch.device("cpu"), label='VALIDATION', forced=False):
        self.model.eval
        losses_valid = []
        loss_function = nn.CrossEntropyLoss()

        if dataset.split_type == 'valid':
            sampler = self.valid_sampler
        if dataset.split_type == 'eval':
            sampler = self.eval_sampler
        
        valid_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=False, drop_last=False, sampler=sampler)
        # valid_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
        tqdm_it=tqdm(valid_dataloader, desc=label+': Chunk {}/{}'.format(0,0))

        for (idx,_, x,y) in tqdm_it:
            x = x.type(self.dtype)
            y = y.type(self.dtype)

            x = x.to(device)
            y = y.to(device)

            y_pred = self.model(x)

            cur_loss = loss_function(y_pred,y)

            losses_valid.append(cur_loss.detach())
                
        losses_valid = torch.Tensor(losses_valid)
        loss_valid = torch.mean(losses_valid)
        print(" => Validation loss at epoch {} is {:.4f}".format(cur_epoch+1, loss_valid))
        
        if forced == True:
            self.best_loss = loss_valid
            self.best_state_dict = copy.deepcopy(self.model.state_dict())
            self.best_epoch = cur_epoch
        else:
            if label == 'VALIDATION':
                if loss_valid <= self.best_loss:
                    self.best_loss = loss_valid
                    self.best_state_dict = copy.deepcopy(self.model.state_dict())
                    self.best_epoch = cur_epoch
        
        return loss_valid.detach().cpu().numpy()
    
    def load_model(self, device):
        self.model = self.model.to(device)
        state_dict = torch.load(self.models_path + self.model_name, map_location=device)
        self.model.load_state_dict(state_dict)
        
    def save_model(self):
    
        """
        SAVE MODEL
        """
    
        torch.save(self.best_state_dict, self.models_path + self.model_name)
    
        # """
        # "SAVE MODEL SETTINGS"
        # """
        
        # transcoder = self.transcoder
        # input_shape = self.model.input_shape
        # output_shape = self.model.output_shape
        
        # cnn_kernel_size = None
        # cnn_dilation = None
        # cnn_nb_layers = None
        # cnn_nb_channels = None
        
        # mlp_hl_1 = None
        # mlp_hl_2 = None
        
        # if 'cnn' in transcoder:
        #     cnn_kernel_size = self.model.kernel_size
        #     cnn_dilation = self.model.dilation
        #     cnn_nb_layers = self.model.nb_layers
        #     cnn_nb_channels = self.model.nb_channels
            
        # if 'mlp' in transcoder:
        #     mlp_hl_1 = self.model.hl_1
        #     mlp_hl_2 = self.model.hl_2

        # model_settings = {
        #     "model_type": transcoder,
        #   "input_shape": input_shape,
        #   "output_shape": output_shape,
        #   "cnn_kernel_size": cnn_kernel_size,
        #   "cnn_dilation": cnn_dilation,
        #   "cnn_nb_layers": cnn_nb_layers,
        #   "cnn_nb_channels": cnn_nb_channels,
        #   "mlp_hl_1": mlp_hl_1,
        #   "mlp_hl_2": mlp_hl_2,
        #   "mels_type": self.classifier,
        #   "batch_size": self.batch_size,
        #   "epochs": self.epochs,
        #   "settings": self.setting_data
        # }

        # with open(self.models_path + self.model_name + '_settings.yaml', 'w') as file:
        #     yaml.dump(model_settings, file)
    def evaluate(self, batch_size=64, device=torch.device("cpu")):
        self.model.eval
                
        self.eval_dataloader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
        tqdm_it=tqdm(self.eval_dataloader, desc='EVALUATION: Chunk {}/{}'.format(0,0))
        
        #save the output of the model in a .dat file. This avoids havind memory issues
        # output_logits = np.memmap(self.outputs_path['logits']+'.dat', dtype=np.float64,
        #               mode='w+', shape=(self.eval_dataset.n_tho_frames, self.classif_inference.n_labels))

        with h5py.File(self.out_path + self.out_name + '.h5', 'a') as hf:
            group = f'split_k_{self.split_id}'
            if group in hf:
                del hf[group]
                print(f"Group {group} deleted.")
            split_group = hf.create_group(group)
            eval_group = split_group.create_group('inference')
            eval_group.create_dataset('audio_name', shape=((self.eval_dataset.len_dataset,)), dtype='S200')
            eval_group.create_dataset('groundtruth', shape=((self.eval_dataset.len_dataset,)), dtype='S200')
            eval_group.create_dataset('inference', shape=((self.eval_dataset.len_dataset,)), dtype='S200')

            for (idx,name,x,y) in tqdm_it:
                x = x.type(self.dtype)
                y = y.type(self.dtype)
                
                x = x.to(device)
                y = y.to(device)
                
                y_pred = self.model(x)
                
                name = list(name)

                argm_pred = torch.argmax(y_pred, dim=1)
                argm = torch.argmax(y, dim=1)

                eval_group['audio_name'][idx] = name
                eval_group['inference'][idx] = [self.eval_dataset.gt_list_to_str[key] for key in argm_pred.detach().cpu().tolist()]
                eval_group['groundtruth'][idx] = [self.eval_dataset.gt_list_to_str[key] for key in argm.detach().cpu().tolist()]
        
        return()