import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import librosa
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
import pandas as pd
from sklearn.utils import resample

hdf5_path = './mel_dataset/mel_dataset_kfold.h5'

keep_idx = 0
l_mel_spectro = []
l_gt = []

with h5py.File(hdf5_path, 'r') as hf:
    for split_type in ['train', 'valid', 'eval']:
        for idx, audio_name in enumerate(hf[f'split_k_{0}'][split_type]['audio_name'][:]):
            mel_spectrogram =  hf[f'split_k_{0}'][split_type]['mel_spectrogram'][idx]
            groundtruth =  hf[f'split_k_{0}'][split_type]['technique'][idx]
            groundtruth = groundtruth.decode('utf-8')

            if (np.mean(mel_spectrogram)) > 0:
                if keep_idx < 1000:
                    if groundtruth != 'BlackShriek':
                        mfccs = librosa.feature.mfcc(S=mel_spectrogram*100-100, n_mfcc=13)
                        l_mel_spectro.append(np.mean(mfccs, axis=1))
                        l_gt.append(groundtruth)
                    keep_idx += 1

l_gt = np.array(l_gt)
l_mel_spectro = np.array(l_mel_spectro)
# l_mel_spectro = l_mel_spectro.reshape(l_mel_spectro.shape[0], -1)  # Flatten to (num_samples, num_mel_bins * num_frames)
print(l_mel_spectro.shape)

# Create a dictionary to map categories to colors
category_to_color = {
    'ClearVoice': 'blue',
    'BlackShriek': 'red',
    'DeathGrowl': 'green',
    'HardcoreScream': 'yellow'
}

colors = [category_to_color[category] for category in l_gt]

# Standardize the data
scaler = StandardScaler()
mel_spectrograms_std = scaler.fit_transform(l_mel_spectro)

# Perform PCA with 3 components
pca = PCA(n_components=3)
pca_result = pca.fit_transform(mel_spectrograms_std)

# Create a DataFrame for easier class-wise resampling
data_df = pd.DataFrame({'PCA1': pca_result[:, 0], 'PCA2': pca_result[:, 1], 'PCA3': pca_result[:, 2], 'Groundtruth': l_gt})
data_df['Colors'] = data_df['Groundtruth'].map(category_to_color)

# Calculate the number of samples to keep per class (adjust as needed)
samples_per_class = data_df['Groundtruth'].value_counts().min()

# Resample the data to have the same number of points per class
resampled_data = data_df.groupby('Groundtruth').apply(lambda x: resample(x, n_samples=samples_per_class, random_state=42)).reset_index(drop=True)

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(resampled_data['PCA1'], resampled_data['PCA2'], resampled_data['PCA3'], c=resampled_data['Colors'], cmap='viridis')

ax.set_title('3D PCA of Mel Spectrograms with Balanced Classes')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

plt.show()
