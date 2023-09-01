from data_generator import SingerSplitDataset, RegularSplitDataset
from models import EffNet, MLP
from trainer import MelTrainer, MelRegularTrainer
import torch
import numpy as np

dataset_name = "dataset_normal_split"
data_path = 'mel_dataset/'
n_splits = 4

train_datasets = []
valid_datasets = []
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
#models = [MLP(input_shape=128, output_shape=4) for k in range(27)]

for split_id in range(n_splits):
    train_datasets.append(RegularSplitDataset(hdf5_path=data_path+dataset_name+'.h5', split_id=split_id, split_type='train'))
    valid_datasets.append(RegularSplitDataset(hdf5_path=data_path+dataset_name+'.h5', split_id=split_id, split_type='valid'))
    eval_datasets.append(RegularSplitDataset(hdf5_path=data_path+dataset_name+'.h5', split_id=split_id, split_type='eval'))

for split_id in range(n_splits):
    trainer = MelRegularTrainer(model=models[split_id], models_path='./model/', model_name=f'model_technique_classif_split{split_id+1}', split_id=split_id, train_dataset=train_datasets[split_id], valid_dataset=valid_datasets[split_id], eval_dataset=eval_datasets[split_id])
    loss_train = trainer.train(device=device, batch_size=256, epochs=20)
    np.save('losses/loss_'+trainer.model_name+'.npy', loss_train)
    trainer.save_model()
