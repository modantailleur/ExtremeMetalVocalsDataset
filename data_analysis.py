import pandas as pd
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_meta = "metadata-files.xlsx"
file_meta_singers = "metadata_singers.xlsx"
df_meta = pd.read_excel(file_meta)
df_meta_singers = pd.read_excel(file_meta_singers)
f_names = df_meta['file_name'].to_numpy()
l_audio_length = []

# df_total = df_meta.merge(df_meta_singers, on="singer_id", how="inner")  # You can change "how" to other options if needed (e.g., "outer", "left", "right")
# #just to check if there are any mistakes (0 put at techniques where there is an audio file)
# df_total_mistakes = df_total[df_total.iloc[:, -14:].eq(0).any(axis=1)]

for f_name in f_names:
    audio, sr = librosa.load('CTED/'+f_name, sr=48000)
    audio_len = len(audio)/48000
    l_audio_length.append(audio_len)

f_names_in_folder = []
for subdir, dirs, files in os.walk('CTED/'):
    for file in files:
        f = os.path.join(subdir, file)
        f_names_in_folder.append(file)

#check if there are any differences between the names in the audio folder and the name in the metadatas
differences = [item for item in f_names_in_folder if item not in f_names]

print('GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG')
print(len(differences))
print(len(f_names))
print(len(f_names_in_folder))
print(len(f_names)+len(differences))
print(differences)

df_meta['audio_len'] = l_audio_length
df_meta = df_meta[df_meta['type']=='Technique']
df_meta = df_meta[df_meta['authors_rank']!='C']

print(df_meta)

print('Total seconds')
print(np.sum(df_meta['audio_len'].to_numpy()))

print('Seconds of clear Voice')
print(np.sum(df_meta[df_meta['name']=='ClearVoice']['audio_len'].to_numpy()))

df_bar = df_meta.groupby(['name']).agg(total_audio_len=('audio_len', 'sum'),
                                    count_audio=('audio_len', 'count')).reset_index()

# Set up the plot
plt.figure(figsize=(10, 6))

# Create a bar plot
sns.barplot(data=df_bar, x='name', y='total_audio_len', color='grey')
sns.set_color_codes("muted")

plt.xlabel('Technique')
plt.ylabel('Audio length (in s)')
plt.title('Audio length for each technique')
#plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

df_bar_vowel = df_meta.groupby(['name', 'vowel']).agg(total_audio_len=('audio_len', 'sum'),
                                    count_audio=('audio_len', 'count')).reset_index()

# Set up the plot
plt.figure(figsize=(10, 6))

# Create a bar plot
sns.barplot(data=df_bar_vowel, x='name', y='total_audio_len', hue='vowel', palette='Set1')

plt.xlabel('Technique')
plt.ylabel('Audio length (in s)')
plt.title('Audio length for each technique and each vowel')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Vowel')
plt.tight_layout()
plt.show()

df_bar_range = df_meta.groupby(['name', 'range']).agg(total_audio_len=('audio_len', 'sum'),
                                    count_audio=('audio_len', 'count')).reset_index()

# Set up the plot
plt.figure(figsize=(10, 6))

# Create a bar plot
sns.barplot(data=df_bar_range, x='name', y='total_audio_len', hue='range', palette='Set1')

plt.xlabel('Technique')
plt.ylabel('Audio length (in s)')
plt.title('Audio length for each technique and each range')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Vowel')
plt.tight_layout()
plt.show()

df_bar_singer = df_meta.groupby('name')['singer_id'].nunique().reset_index()
df_bar_singer.columns = ['name', 'num_singers']

# Set up the plot
plt.figure(figsize=(10, 6))

# Create a bar plot
sns.barplot(data=df_bar_singer, x='name', y='num_singers', palette='Set1')

plt.xlabel('Technique')
plt.ylabel('Audio length (in s)')
plt.title('Number of singers for each technique')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Vowel')
plt.tight_layout()
plt.show()