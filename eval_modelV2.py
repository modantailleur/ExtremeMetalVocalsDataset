from data_generator import SingerSplitDataset, RegularSplitDataset
from models import EffNet 
from trainer import MelTrainer, MelRegularTrainer
import torch
import numpy as np

dataset_name = "dataset_normal_split"
data_path = 'mel_dataset/'
n_splits = 4

train_datasets = []
eval_datasets = []

#manage gpu
force_cpu=False
useCuda = torch.cuda.is_available() and not force_cpu
if useCuda:
    print('Using CUDA.')
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor
    #MT: add
    device = torch.device("cuda:0")
else:
    print('No CUDA available.')
    dtype = torch.FloatTensor
    ltype = torch.LongTensor
    #MT: add
    device = torch.device("cpu")


models = [EffNet(n_labels=4, device=device) for k in range(n_splits)]

for split_id in range(n_splits):
    train_datasets.append(RegularSplitDataset(hdf5_path=data_path+dataset_name+'.h5', split_id=split_id, split_type='train'))
    eval_datasets.append(RegularSplitDataset(hdf5_path=data_path+dataset_name+'.h5', split_id=split_id, split_type='eval'))

for split_id in range(n_splits):
    trainer = MelRegularTrainer(model=models[split_id], models_path='./model/', model_name=f'model_technique_classif_split{split_id+1}', split_id=split_id, train_dataset=train_datasets[split_id], eval_dataset=eval_datasets[split_id], out_name='dataset_technique_classif')
    trainer.load_model(device=device)
    trainer.evaluate(device=device)