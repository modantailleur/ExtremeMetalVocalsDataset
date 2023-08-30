import h5py
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

out_path = './outputs/'
out_name = 'dataset_singer_classif'
n_splits = 4
inference = []
groundtruth = []
name = []

with h5py.File(out_path + out_name+ '.h5', 'r') as hf:
    for split_id in range(n_splits):
        inference_dataset = hf[f'split_k_{split_id}']['inference']['inference']
        groundtruth_dataset = hf[f'split_k_{split_id}']['inference']['groundtruth']
        name += [item.decode('utf-8') for item in hf[f'split_k_{split_id}']['inference']['audio_name']]
        inference += [item.decode('utf-8') for item in inference_dataset]
        groundtruth += [item.decode('utf-8') for item in groundtruth_dataset]

# Create a dictionary from the lists
data = {'Name': name, 'Inference': inference, 'GroundTruth': groundtruth}
# Create a pandas DataFrame
df = pd.DataFrame(data)
# Print the DataFrame
df['Name'] = df['Name'].str.split('__').str[0]
print(df)
inference_one_hot = pd.get_dummies(df['Inference'], prefix='')
groundtruth_one_hot = pd.get_dummies(df['GroundTruth'], prefix='')

inference_one_hot = pd.concat([df['Name'], inference_one_hot], axis=1)
groundtruth_one_hot = pd.concat([df['Name'], groundtruth_one_hot], axis=1)

inference_avg = inference_one_hot.groupby('Name').mean().reset_index()
groundtruth_avg = groundtruth_one_hot.groupby('Name').mean().reset_index()

print('before')
print(inference_avg)
print(groundtruth_avg)

# Apply thresholding: below 0.5 becomes 0, 0.5 or above becomes 1
one_hot_columns_inf = inference_avg.columns[1:]  # Assuming one-hot columns start from the 4th column
inference_avg['Max'] = inference_avg[one_hot_columns_inf].idxmax(axis=1)
one_hot_columns_gt = groundtruth_avg.columns[1:]  # Assuming one-hot columns start from the 4th column
groundtruth_avg['Max'] = groundtruth_avg[one_hot_columns_gt].idxmax(axis=1)

print('and after')
print(inference_avg)
print(groundtruth_avg)

inference = np.array(inference)
groundtruth = np.array(groundtruth)
scores = inference == groundtruth

print(scores)
print(len(inference))
print(len(groundtruth))
print(np.mean(scores))
print(confusion_matrix(inference, groundtruth, normalize='true'))

inference_macro = inference_avg['Max'].to_numpy()
groundtruth_macro = groundtruth_avg['Max'].to_numpy()
scores_macro = inference_macro == groundtruth_macro


print(scores)
print(len(inference_macro))
print(len(groundtruth_macro))
print(np.mean(scores_macro))
print(confusion_matrix(inference_macro, groundtruth_macro, normalize='true'))