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

np.random.seed(0)

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def divide_chunks_2D(l, n):
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[:, i:i + n]

file_meta = "metadata_files.xlsx"
file_kfolds = "split_kfolds.xlsx"
file_meta_singers = "metadata_singers.xlsx"

#read df and filter
df_meta = pd.read_excel(file_meta)
df_k_folds = pd.read_excel(file_kfolds)
df_meta = df_meta.merge(df_k_folds, on='file_name')

#keep only rows where there is a split_k value
split_k_columns = [col for col in df_meta.columns if 'split_k_' in col]
mask = df_meta[split_k_columns] != "None"
df_meta = df_meta[mask.all(axis=1)]
df_meta.reset_index(drop=True, inplace=True)

n_splits = sum(1 for col in df_meta.columns if 'split_k_' in col)

#######################
###################
# DROP DISTORSION TECHNIQUES FROM TRAIN AND VALID
#################
#####################

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
# f_splits = df_meta['split'].to_numpy()
f_splits =  df_meta.filter(like='split_k_', axis=1).to_numpy()

l_audio_length = []
out_path = './mel_dataset/'
out_name = 'dataset_no_distorsion_split'

mels_n_num = 0
mels_n_num_per_subject = np.zeros(27, dtype=int)
# 0: train, 1: valid, 2: test
mels_n_num_per_split = np.zeros((n_splits, 3), dtype=int)
mels_n_num_per_non_subject = np.zeros(27, dtype=int)
n_frames = 64
n_fft=1024
hop_length = 256
n_mels = 128
fmin=20
fmax=8000
threshold = 0.05

for idx, (f_name, f_singer, f_gt) in enumerate(zip(f_names, f_singers, f_gts)):
    audio, sr = librosa.load('CTED/'+f_name, sr=48000)
    audio = librosa.util.normalize(audio)
    audio_len = len(audio)/48000
    l_audio_length.append(audio_len)
    print(f'\rCOMPUTED: {idx} / {len(f_names)}')

df_meta['l_audio_length'] = l_audio_length

sum_by_name = df_meta.groupby('name')['l_audio_length'].sum()
smallest_value = sum_by_name.min()
perc_by_name = ((sum_by_name - smallest_value) / sum_by_name).reset_index()
undersample_dict = perc_by_name.set_index('name')['l_audio_length'].to_dict()


for idx, (f_name, f_singer, f_gt, f_split) in enumerate(zip(f_names, f_singers, f_gts, f_splits)):
    
    audio, sr = librosa.load('CTED/'+f_name, sr=48000)
    audio = librosa.util.normalize(audio)
    # audio_len = len(audio)/48000
    # l_audio_length.append(audio_len)
    mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mels = np.array([mels[:, i:i + n_frames] for i in range(0, mels.shape[1], n_frames) if mels[:, i:i + n_frames].shape[1] == n_frames])
    
    #MT: REACTIVE IF DROP ELEMENTS IS WANTED
    #drop elements
    # level = mels.mean(axis=(1,2))
    # indices_too_low = np.where(level < threshold)
    # mels = np.delete(mels, indices_too_low, axis=0)
    # num_elements_undersample = int(mels.shape[0]*undersample_dict[f_gt])
    # indices_undersample = np.random.choice(mels.shape[0], num_elements_undersample, replace=False)
    # mels = np.delete(mels, indices_undersample, axis=0)
    ###########################################
    ###########################################

    mels_n_num = mels_n_num + mels.shape[0]
    # mels_n_num_per_subject[f_singer-1] = mels_n_num_per_subject[f_singer-1] + mels.shape[0]
    for split in range(n_splits):
        if f_split[split] == 'train':
            mels_n_num_per_split[split][0] += mels.shape[0]
        if f_split[split] == 'valid':
            mels_n_num_per_split[split][1] += mels.shape[0]
        if f_split[split] == 'eval':
            mels_n_num_per_split[split][2] += mels.shape[0]

    print(f'\rCOMPUTED: {idx} / {len(f_names)}')

mels_n_num_per_non_subject = np.sum(mels_n_num_per_subject) - mels_n_num_per_subject

create_folder(os.path.dirname('./mel_dataset/'))
with h5py.File(out_path + out_name+ '.h5', 'w') as hf:
    split_groups = []
    for split in range(n_splits):  # Replace num_splits with the actual number of splits
        # for split_id in range(27):  # Replace num_splits with the actual number of splits
        split_group = hf.create_group(f'split_k_{split}')
        split_groups.append(split_group)

        # f_names = df_meta[df_meta['singer_id']==split_id+1]['file_name'].to_numpy()
        # f_gt = df_meta[df_meta['singer_id']==split_id+1]['name'].to_numpy()
        train_group = split_group.create_group('train')
        valid_group = split_group.create_group('valid')
        eval_group = split_group.create_group('eval')

        train_group.create_dataset('audio_name', shape=((mels_n_num_per_split[split][0],)), dtype='S200')
        train_group.create_dataset('mel_spectrogram', shape=((mels_n_num_per_split[split][0], n_mels, n_frames)), dtype=np.float32)
        train_group.create_dataset('groundtruth', shape=((mels_n_num_per_split[split][0],)), dtype='S200')
        train_group.create_dataset('singer', shape=((mels_n_num_per_split[split][0],)), dtype=np.int16)

        valid_group.create_dataset('audio_name', shape=((mels_n_num_per_split[split][1],)), dtype='S200')
        valid_group.create_dataset('mel_spectrogram', shape=((mels_n_num_per_split[split][1], n_mels, n_frames)), dtype=np.float32)
        valid_group.create_dataset('groundtruth', shape=((mels_n_num_per_split[split][1],)), dtype='S200')
        valid_group.create_dataset('singer', shape=((mels_n_num_per_split[split][1],)), dtype=np.int16)

        eval_group.create_dataset('audio_name', shape=((mels_n_num_per_split[split][2],)), dtype='S200')
        eval_group.create_dataset('mel_spectrogram', shape=((mels_n_num_per_split[split][2], n_mels, n_frames)), dtype=np.float32)
        eval_group.create_dataset('groundtruth', shape=((mels_n_num_per_split[split][2],)), dtype='S200')
        eval_group.create_dataset('singer', shape=((mels_n_num_per_split[split][2],)), dtype=np.int16)

    mels_n_num = np.zeros((n_splits, 3), dtype=int)

    tve_dict = {
        'train':0,
        'valid':1, 
        'eval':2
    }

    for idx, (f_name, f_singer, f_gt, f_split) in enumerate(zip(f_names, f_singers, f_gts, f_splits)):

        audio, sr = librosa.load('CTED/'+f_name, sr=48000)
        audio = librosa.util.normalize(audio)
        # audio_len = len(audio)/48000
        # l_audio_length.append(audio_len)
        mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
        mels = np.array([mels[:, i:i + n_frames] for i in range(0, mels.shape[1], n_frames) if mels[:, i:i + n_frames].shape[1] == n_frames])

        #MT: REACTIVE IF DROP ELEMENTS IS WANTED
        #drop elements
        # level = mels.mean(axis=(1,2))
        # indices_too_low = np.where(level < threshold)
        # mels = np.delete(mels, indices_too_low, axis=0)
        # num_elements_to_drop = int(mels.shape[0]*undersample_dict[f_gt])
        # indices_to_drop = np.random.choice(mels.shape[0], num_elements_to_drop, replace=False)
        # mels = np.delete(mels, indices_to_drop, axis=0)
        ###########################################
        ###########################################

        # for split_id in range(27):
        #     #if this is not the current singer id
        #     if split_id+1 != f_singer:
        #         te = 'train'
        #     else:
        #         te = 'eval'
        


        for split in range(n_splits):
            try:
                tve_idx = tve_dict[f_split[split]]
            except KeyError:
                continue

            hf[f'split_k_{split}'][f_split[split]]['audio_name'][mels_n_num[split][tve_idx]:mels_n_num[split][tve_idx]+mels.shape[0]] = [f_name + '___' + str(k) for k in range(mels.shape[0])]
            hf[f'split_k_{split}'][f_split[split]]['mel_spectrogram'][mels_n_num[split][tve_idx]:mels_n_num[split][tve_idx]+mels.shape[0]] = mels
            hf[f'split_k_{split}'][f_split[split]]['groundtruth'][mels_n_num[split][tve_idx]:mels_n_num[split][tve_idx]+mels.shape[0]] = [f_gt for k in range(mels.shape[0])]
            hf[f'split_k_{split}'][f_split[split]]['singer'][mels_n_num[split][tve_idx]:mels_n_num[split][tve_idx]+mels.shape[0]] = [f_singer for k in range(mels.shape[0])]

            mels_n_num[split][tve_idx] += mels.shape[0]
            
            print(f'\rCOMPUTED: {idx} / {len(f_names)}')
