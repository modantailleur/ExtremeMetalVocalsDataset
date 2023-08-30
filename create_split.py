import pandas as pd
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import librosa

np.random.seed(0)

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def divide_chunks_2D(l, n):
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[:, i:i + n]

#file_meta = "metadata-files.xlsx"
file_meta = "metadata_files.xlsx"
file_meta_singers = "metadata_singers.xlsx"

#read df and filter
df = pd.read_excel(file_meta)

#subset is recalculated, so dropped if already exists
if 'subset' in df.columns:
    df.drop(columns=['subset'], inplace=True)

# df_meta.to_excel("metadata_files_audio_length.xlsx")  
df_meta = df[df['type']=='Technique']
df_meta = df_meta[df_meta['authors_rank']!='C']
df_meta = df_meta[df_meta['name']!='GrindInhale']
df_meta = df_meta.sort_values(by=['singer_id'])

df_meta = pd.DataFrame(df_meta)

df_temp = df_meta.drop(['range', 'vowel', 'authors_rank'], axis=1)
df_temp = df_temp.groupby(['singer_id', 'name'])['duration(s)'].sum().reset_index(name='total_audio_length')
df_temp = df_temp.sort_values(by=['singer_id', 'name', 'total_audio_length'])

print(df_temp)

# Initialize dictionaries to keep track of assigned techniques for each subset
train_assigned = {}
test_assigned = {}

# Iterate through the sorted DataFrame
subset_column = []

#assign each technique to a subset
for _, row in df_temp.iterrows():
    singer_id = row['singer_id']
    name = row['name']
    audio_length = row['total_audio_length']
    
    # Check if technique is already assigned to train or test
    if (singer_id, name) in train_assigned:
        subset_column.append('train')
        train_assigned[(singer_id, name)] -= audio_length
    elif (singer_id, name) in test_assigned:
        subset_column.append('test')
        test_assigned[(singer_id, name)] -= audio_length
    else:
        # Calculate the remaining audio length to distribute
        remaining_audio_length = audio_length
        
        # Calculate the remaining percentage to distribute
        if (len(train_assigned) + len(test_assigned)) != 0:
            remaining_percentage = 0.7 if len(train_assigned) / (len(train_assigned) + len(test_assigned)) <= 0.7 else 0.3
        else:
            remaining_percentage = 0.7
        
        # Decide whether to assign to train or test
        if np.random.rand() < remaining_percentage:
            subset_column.append('train')
            train_assigned[(singer_id, name)] = remaining_audio_length
        else:
            subset_column.append('test')
            test_assigned[(singer_id, name)] = remaining_audio_length

# Add the 'subset' column to the DataFrame
df_temp['subset'] = subset_column

# Perform an outer join on 'singer_id' and 'name'
result_df = df.merge(df_temp, on=['singer_id', 'name'], how='outer')

# Fill missing values with '-'
result_df.fillna('-', inplace=True)
result_df.drop(['total_audio_length'], axis=1, inplace=True)

result_df.to_excel('metadata_files_subset.xlsx', index=False)

gp = result_df.groupby(['subset'])['duration(s)'].sum()

print(gp)