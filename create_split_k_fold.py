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
file_meta_singers = "metadata_singers.xlsx"

#read df and filter
df_meta = pd.read_excel(file_meta)
df_meta = df_meta[df_meta['type']=='Technique']
df_meta = df_meta[df_meta['authors_rank']!='C']
df_meta = df_meta[df_meta['name']!='GrindInhale']
df_meta = df_meta.sort_values(by=['singer_id'])
n_splits = 4

######################
########## TRAIN TEST VALID SPLIT
######################

# Calculate the number of samples for each split
total_samples = len(df_meta)
train_samples = int(0.7 * total_samples)
valid_samples = int(0.15 * total_samples)
test_samples = total_samples - train_samples - valid_samples

# Create train split
train_df, remaining_df = train_test_split(df_meta, train_size=train_samples, random_state=42)

# Create valid and test splits
valid_df, eval_df = train_test_split(remaining_df, train_size=valid_samples, random_state=42)

# Add a new column with split information
train_df['split'] = 'train'
valid_df['split'] = 'valid'
eval_df['split'] = 'eval'

# Concatenate all splits back into a single DataFrame
df_meta = pd.concat([train_df, valid_df, eval_df])
df_meta = df_meta.reset_index(drop=True)

######################
########## K FOLDS
######################

kf = KFold(n_splits=4)

split_gen = kf.split(df_meta.index.values.tolist())
train_splits = []
valid_splits = []
eval_splits = []

for idx, (fulltrain_index, eval_index) in enumerate(split_gen):
    train_samples = int(0.8 * len(fulltrain_index))
    train_index, valid_index = train_test_split(fulltrain_index, train_size=train_samples, random_state=42)

    train_splits.append(train_index)
    valid_splits.append(valid_index)
    eval_splits.append(eval_index)

train_splits = np.array(train_splits)
valid_splits = np.array(valid_splits)
eval_splits = np.array(eval_splits)

for split_idx in range(n_splits):
    split_name = f'split_k_{split_idx}'
    
    df_meta[split_name] = 'train'
    df_meta.loc[valid_splits[split_idx], split_name] = 'valid'
    df_meta.loc[eval_splits[split_idx], split_name] = 'eval'


#######################
###################
# SAVE SPLIT
#################
#####################

df = pd.read_excel(file_meta)
cols_to_use = df_meta.columns.difference(df.columns)

# Merge the DataFrames on the 'file_name' column using a left join
merged_df = df.merge(df_meta.drop(columns=['singer_id', 'type', 'name', 'range', 'vowel', 'authors_rank', 'duration(s)']), on='file_name', how='left')

# Replace NaN values with 'None'
merged_df = merged_df.fillna('None')

print(df_meta)
print(df)
print(merged_df)

# Save the merged DataFrame as an Excel file
merged_df.to_excel('metadata_files_kfolds.xlsx', index=False)

#####################
####################