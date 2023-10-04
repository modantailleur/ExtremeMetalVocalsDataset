import pandas as pd
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import librosa
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from data_generator import SingerSplitDataset, TechniqueClassificationDataset, BinaryTechniqueClassificationDataset,  SingerClassificationDataset
from models import EffNet, MLP
from trainer import MelTrainer, MelRegularTrainer
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from torchaudio.transforms import MelSpectrogram
import torch
import matplotlib as mpl

#fix the random seed
torch.manual_seed(0)

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def divide_chunks_2D(l, n):
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[:, i:i + n]

def create_mel_dataset(dataset_name='default', distorsion_in_train=True):

    file_meta = "metadata_files.csv"
    file_kfolds = "split_kfolds.csv"

    #read df and filter
    df_meta = pd.read_csv(file_meta)
    df_k_folds = pd.read_csv(file_kfolds)
    df_meta = df_meta.merge(df_k_folds, on='file_name')

    #keep only rows where there is a split_k value
    split_k_columns = [col for col in df_meta.columns if 'split' in col]
    mask = df_meta[split_k_columns] != "None"
    df_meta = df_meta[mask.all(axis=1)]
    df_meta.reset_index(drop=True, inplace=True)

    n_splits = sum(1 for col in df_meta.columns if 'split' in col)

    #######################
    ###################
    # DROP DISTORSION TECHNIQUES FROM TRAIN AND VALID
    #################
    #####################

    if not distorsion_in_train:
        for col in split_k_columns:
            df_meta[col] = df_meta.apply(lambda row: 'out' if row[col] == 'train' and row['name'] != 'ClearVoice' else row[col], axis=1)

    #######################
    ###################
    # CREATE MEL DATASET FROM DF
    #################
    #####################

    f_names = df_meta['file_name'].to_numpy()
    f_gts = df_meta['name'].to_numpy()
    f_singers = df_meta['singer_id'].to_numpy()
    f_splits =  df_meta.filter(like='split', axis=1).to_numpy()

    out_path = './mel_dataset/'

    mels_n_num = 0
    # 0: train, 1: valid, 2: test
    mels_n_num_per_split = np.zeros((n_splits, 3), dtype=int)
    tve_dict = {
        'train':0,
        'valid':1, 
        'eval':2
    }
    n_frames = 192
    n_fft=1024
    hop_length = 256
    n_mels = 128
    fmin=20
    fmax=8000
    window = 'hann'
    norm = "slaney"
    mel_scale = "slaney"

    melspec_layer = MelSpectrogram(
        n_mels=n_mels,
        sample_rate=48000,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        f_min=fmin,
        f_max=fmax,
        center=True,
        power=2.0,
        mel_scale=mel_scale,
        norm=norm,
        normalized=True,
        pad_mode="constant",
    )

    #have to first calculate the melspectrograms, in order to get their shape to create the h5 files. Takes a bit of time but makes things easier 
    # for training and evaluating (only 1 h5 file for a dataset)
    for idx, (f_name, f_singer, f_gt, f_split) in enumerate(zip(f_names, f_singers, f_gts, f_splits)):
        
        audio, sr = librosa.load('CTED/'+f_name, sr=48000)
        audio = librosa.util.normalize(audio)

        x_wave = torch.Tensor(audio).unsqueeze(0)
        torch_mels = melspec_layer(x_wave)
        torch_mels = 10 * torch.log10(torch_mels + 1e-10)
        torch_mels = torch.clamp((torch_mels + 100) / 100, min=0.0)
        mels = torch_mels.squeeze(0).numpy()

        # mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
        # mels = np.log(mels + 10e-10)

        mels = np.array([mels[:, i:i + n_frames] for i in range(0, mels.shape[1], n_frames) if mels[:, i:i + n_frames].shape[1] == n_frames])

        mels_n_num = mels_n_num + mels.shape[0]

        for split in range(n_splits):
            if f_split[split] == 'train':
                mels_n_num_per_split[split][0] += mels.shape[0]
            if f_split[split] == 'valid':
                mels_n_num_per_split[split][1] += mels.shape[0]
            if f_split[split] == 'eval':
                mels_n_num_per_split[split][2] += mels.shape[0]

        print(f'\rCOMPUTED: {idx} / {len(f_names)}')

    create_folder(os.path.dirname('./mel_dataset/'))
    with h5py.File(out_path + dataset_name+ '.h5', 'w') as hf:
        split_groups = []
        for split in range(n_splits):  

            split_group = hf.create_group(f'split{split}')
            split_groups.append(split_group)

            train_group = split_group.create_group('train')
            valid_group = split_group.create_group('valid')
            eval_group = split_group.create_group('eval')

            train_group.create_dataset('audio_name', shape=((mels_n_num_per_split[split][0],)), dtype='S200')
            train_group.create_dataset('mel_spectrogram', shape=((mels_n_num_per_split[split][0], n_mels, n_frames)), dtype=np.float32)
            train_group.create_dataset('technique', shape=((mels_n_num_per_split[split][0],)), dtype='S200')
            train_group.create_dataset('singer', shape=((mels_n_num_per_split[split][0],)), dtype=np.int16)

            valid_group.create_dataset('audio_name', shape=((mels_n_num_per_split[split][1],)), dtype='S200')
            valid_group.create_dataset('mel_spectrogram', shape=((mels_n_num_per_split[split][1], n_mels, n_frames)), dtype=np.float32)
            valid_group.create_dataset('technique', shape=((mels_n_num_per_split[split][1],)), dtype='S200')
            valid_group.create_dataset('singer', shape=((mels_n_num_per_split[split][1],)), dtype=np.int16)

            eval_group.create_dataset('audio_name', shape=((mels_n_num_per_split[split][2],)), dtype='S200')
            eval_group.create_dataset('mel_spectrogram', shape=((mels_n_num_per_split[split][2], n_mels, n_frames)), dtype=np.float32)
            eval_group.create_dataset('technique', shape=((mels_n_num_per_split[split][2],)), dtype='S200')
            eval_group.create_dataset('singer', shape=((mels_n_num_per_split[split][2],)), dtype=np.int16)

        mels_n_num = np.zeros((n_splits, 3), dtype=int)

        for idx, (f_name, f_singer, f_gt, f_split) in enumerate(zip(f_names, f_singers, f_gts, f_splits)):

            audio, sr = librosa.load('CTED/'+f_name, sr=48000)
            audio = librosa.util.normalize(audio)

            x_wave = torch.Tensor(audio).unsqueeze(0)
            torch_mels = melspec_layer(x_wave)
            torch_mels = 10 * torch.log10(torch_mels + 1e-10)
            torch_mels = torch.clamp((torch_mels + 100) / 100, min=0.0)
            mels = torch_mels.squeeze(0).numpy()

            # mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
            # mels = np.log(mels + 10e-10)

            mels = np.array([mels[:, i:i + n_frames] for i in range(0, mels.shape[1], n_frames) if mels[:, i:i + n_frames].shape[1] == n_frames])

            for split in range(n_splits):
                try:
                    tve_idx = tve_dict[f_split[split]]
                except KeyError:
                    continue

                hf[f'split{split}'][f_split[split]]['audio_name'][mels_n_num[split][tve_idx]:mels_n_num[split][tve_idx]+mels.shape[0]] = [f_name + '___' + str(k) for k in range(mels.shape[0])]
                hf[f'split{split}'][f_split[split]]['mel_spectrogram'][mels_n_num[split][tve_idx]:mels_n_num[split][tve_idx]+mels.shape[0]] = mels
                hf[f'split{split}'][f_split[split]]['technique'][mels_n_num[split][tve_idx]:mels_n_num[split][tve_idx]+mels.shape[0]] = [f_gt for k in range(mels.shape[0])]
                hf[f'split{split}'][f_split[split]]['singer'][mels_n_num[split][tve_idx]:mels_n_num[split][tve_idx]+mels.shape[0]] = [f_singer for k in range(mels.shape[0])]

                mels_n_num[split][tve_idx] += mels.shape[0]
                
                print(f'\rCOMPUTED: {idx} / {len(f_names)}')

def train_model(dataset_name='default', model_prefix='default', groundtruth='technique', epochs=20, batch_size=32):

    if groundtruth == 'technique':
        Dataset = TechniqueClassificationDataset
        n_labels = 4

    if groundtruth == 'technique_binary':
        Dataset = BinaryTechniqueClassificationDataset
        n_labels = 2

    if groundtruth == 'singer':
        Dataset = SingerClassificationDataset
        n_labels = 27

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


    models = [EffNet(n_labels=n_labels, device=device) for k in range(n_splits)]
    #models = [MLP(input_shape=128, output_shape=4) for k in range(27)]

    for split_id in range(n_splits):
        train_datasets.append(Dataset(hdf5_path=data_path+dataset_name+'.h5', split_id=split_id, split_type='train'))
        valid_datasets.append(Dataset(hdf5_path=data_path+dataset_name+'.h5', split_id=split_id, split_type='valid'))
        eval_datasets.append(Dataset(hdf5_path=data_path+dataset_name+'.h5', split_id=split_id, split_type='eval'))

    for split_id in range(n_splits):
        trainer = MelRegularTrainer(model=models[split_id], models_path='./model/', model_name=f'{model_prefix}_{split_id+1}', split_id=split_id, train_dataset=train_datasets[split_id], valid_dataset=valid_datasets[split_id], eval_dataset=eval_datasets[split_id])
        loss_train = trainer.train(device=device, batch_size=batch_size, epochs=epochs)
        np.save('losses/loss_'+trainer.model_name+'.npy', loss_train)
        trainer.save_model()

def eval_model(dataset_name='default', model_prefix='default', groundtruth='technique', out_name='default'):

    if groundtruth == 'technique':
        Dataset = TechniqueClassificationDataset
        n_labels = 4

    if groundtruth == 'technique_binary':
        Dataset = BinaryTechniqueClassificationDataset
        n_labels = 2

    if groundtruth == 'singer':
        Dataset = SingerClassificationDataset
        n_labels = 27

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


    models = [EffNet(n_labels=n_labels, device=device) for k in range(n_splits)]

    for split_id in range(n_splits):
        train_datasets.append(Dataset(hdf5_path=data_path+dataset_name+'.h5', split_id=split_id, split_type='train'))
        eval_datasets.append(Dataset(hdf5_path=data_path+dataset_name+'.h5', split_id=split_id, split_type='eval'))

    for split_id in range(n_splits):
        trainer = MelRegularTrainer(model=models[split_id], models_path='./model/', model_name=f'{model_prefix}_{split_id+1}', split_id=split_id, train_dataset=train_datasets[split_id], eval_dataset=eval_datasets[split_id], out_name=out_name)
        trainer.load_model(device=device)
        trainer.evaluate(device=device)

def calculate_metric(out_name='default', exp_type='technique'):
    out_path = './outputs/'
    n_splits = 4
    inference = []
    groundtruth = []
    name = []

    with h5py.File(out_path + out_name+ '.h5', 'r') as hf:
        for split_id in range(n_splits):
            inference_dataset = hf[f'split{split_id}']['inference']['inference']
            groundtruth_dataset = hf[f'split{split_id}']['inference']['groundtruth']
            name += [item.decode('utf-8') for item in hf[f'split{split_id}']['inference']['audio_name']]
            inference += [item.decode('utf-8') for item in inference_dataset]
            groundtruth += [item.decode('utf-8') for item in groundtruth_dataset]

    data = {'Name': name, 'Inference': inference, 'GroundTruth': groundtruth}
    df = pd.DataFrame(data)
    df['Name'] = df['Name'].str.split('__').str[0]

    inference_one_hot = pd.get_dummies(df['Inference'], prefix='')
    groundtruth_one_hot = pd.get_dummies(df['GroundTruth'], prefix='')

    inference_one_hot = pd.concat([df['Name'], inference_one_hot], axis=1)
    groundtruth_one_hot = pd.concat([df['Name'], groundtruth_one_hot], axis=1)

    inference_avg = inference_one_hot.groupby('Name').mean().reset_index()
    groundtruth_avg = groundtruth_one_hot.groupby('Name').mean().reset_index()

    # Apply thresholding: below 0.5 becomes 0, 0.5 or above becomes 1
    one_hot_columns_inf = inference_avg.columns[1:]  # Assuming one-hot columns start from the 4th column
    inference_avg['Max'] = inference_avg[one_hot_columns_inf].idxmax(axis=1)
    one_hot_columns_gt = groundtruth_avg.columns[1:]  # Assuming one-hot columns start from the 4th column
    groundtruth_avg['Max'] = groundtruth_avg[one_hot_columns_gt].idxmax(axis=1)

    # score calculation at the scale of a mel-spectrogram frame
    inference = np.array(inference)
    groundtruth = np.array(groundtruth)
    scores = inference == groundtruth

    #score calculation at the scale of 
    inference_file = inference_avg['Max'].to_numpy()
    groundtruth_file = groundtruth_avg['Max'].to_numpy()
    scores_file = inference_file == groundtruth_file

    if exp_type == 'technique':
        labels = ['_ClearVoice', '_BlackShriek', '_DeathGrowl', '_HardcoreScream']
        conf_mat = confusion_matrix(inference_file, groundtruth_file, normalize='true', labels=labels)
        # mpl.rcParams['font.family'] = 'Times New Roman'
        # mpl.rcParams['font.size'] = 20
        plt.figure(figsize=(8, 6))
        # sns.set_theme(font='Times New Roman', font_scale=1)
        sns.set(font='Times New Roman', font_scale=1.8)
        sns.heatmap(conf_mat*100, annot=True, cmap="magma_r", fmt=".0f",
            xticklabels=['Clear Voice', 'Black Shriek', 'Death Growl', 'Hardcore Scream'], 
            yticklabels=['Clear Voice', 'Black Shriek', 'Death Growl', 'Hardcore Scream'], annot_kws={"weight": "bold"})
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0, ha='right')
        plt.subplots_adjust(left=0.3, right=1.0, top=0.9, bottom=0.3)
        # plt.xlabel("Predicted")
        # plt.ylabel("Actual")
        # plt.title("Confusion Matrix")
        plt.show()
    else:
        conf_mat = confusion_matrix(inference_file, groundtruth_file, normalize='true')

    print('MICRO ACCURACY')
    print(np.mean(scores_file))
    print('MACRO ACCURACY')
    print(np.mean(np.diag(conf_mat)))
    print(conf_mat)